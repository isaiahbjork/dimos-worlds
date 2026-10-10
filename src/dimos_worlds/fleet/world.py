"""Several walking robots in one MuJoCo world, each on the physics step its policy was trained with.

Why not one MjModel: a MuJoCo model has one timestep, and DimOS's walking policies were trained at different ones
(Go1/Go2 5 ms, G1 2 ms; both act every 20 ms). Run a policy on the wrong physics step and its gait changes. So each
robot gets its own MjModel/MjData: the same scene, the robot itself at its own timestep, and every OTHER robot as a
kinematic proxy (mocap bodies with that robot's collision and visual geoms). One world tick is 20 ms of sim time for
everyone: proxies are posed from the other robots' state at the start of the tick, then each robot steps its own
substeps (Go2: 4 x 5 ms, G1: 10 x 2 ms). Robots collide with the scene in full physics and with each other one tick
late, as kinematic obstacles; head cameras see the other robots.

The scene must be static (no joints): DimOS's policy controllers read qpos[7:] as the robot's joints. Its ground
plane must be a geom named "floor" (DimOS's G1 model has explicit foot/floor contact pairs).

Timing (why a run reproduces exactly, whatever else the machine is doing): sim time is the only clock in the
stepping. Wall clock only paces the loop (`speed` x real time, best effort; a busy machine makes sim time run
slower, never a tick longer). Commands from other threads land between ticks, under the world lock, and are logged
with their tick, plus a state hash every STATE_EVERY_TICKS; dimos_worlds.replay re-issues the commands at the same
ticks and must reach the same hashes. Three things keep a busy machine from changing what the robots do:
  * a route is ONE command (walk_route): the controller moves to the next corner inside the tick, so no corner
    waits on a caller's thread (sending legs one at a time made every run under load walk a different path)
  * the velocity command reaching a policy is rate-limited per robot (acc_vx, acc_wz): a step in forward speed or
    yaw rate is a jolt these policies were not trained for (it made the G1 fall)
  * each ONNX policy runs on one thread without spin-waiting (a thread pool's spinning fights the world thread for
    cores and gains nothing on a ~100-wide MLP)

The policies are DimOS's own classes (Go1OnnxController, G1OnnxController) loading DimOS's ONNX files; only their
inference session is replaced by a single-threaded one.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import logging
import math
import threading
import time
from typing import Any
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from dimos_worlds.fleet.robots import RobotSpec, policy_path, robot_assets
from dimos_worlds.replay.log import FORMAT, RunLog

log = logging.getLogger("dimos_worlds.fleet.world")

CTRL_DT = 0.02  # one world tick = one policy step for every robot
STATE_EVERY_TICKS = 250  # a state hash in the run log every 5 s of sim
LAG_LOG_S = 30.0
RECOVER_S = 1.0  # a fallen robot is set back on its feet (a sim reset of its pose) after this long

# Route controller (port of the warehouse G1 go-to controller; same numbers for every body).
ARRIVE_M = 0.25  # a goal counts as reached this close ...
TRACK_ALPHA = 0.1  # velocity tracking: low-pass per tick
TRACK_KP = 0.0
TRACK_KI = 0.6  # per second
TRACK_I_MAX = 0.2
HOLD_KP = 0.5  # zero command: per second, pulls back to the held pose ...
HOLD_MAX = 0.1  # ... with at most this command
HOLD_SETTLED = 0.1  # m/s: the held pose is taken once the body has slowed below this
# No braking command while walking forward (velocity tracking): above FWD_V m/s forward, the vx command stays at or
# above NEUTRAL_VX, the command the G1 policy reads as zero forward speed (DimOS feeds it 2 x command - 0.18, the
# 0.18 being its drift compensation for a standing G1). A zero or negative command at walking speed is a request to
# walk backwards: the G1 leans forward, speeds up and falls (seen when settling after an overshoot, after an in-place
# turn that wound up the tracking integral, and in hold). From a neutral command it slows down on its own.
FWD_V = 0.2  # m/s
NEUTRAL_VX = 0.09  # policy command units
LEAVE_M = 0.6  # ... and stays reached unless the robot drifts out past this
TURN_FIRST_M = 1.2  # inside this distance of the goal, turn to face it before stepping (walking while turning orbits)
VIA_M = 0.45  # a route corner counts as passed this close
FINAL_TURN_TOL = 0.2
FINAL_TURN_MAX_S = 8.0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _quat_yaw(q: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _single_thread_session(path: str) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


class _Command:
    """What the policy is asked for: the rate-limited command the policy reads, and the target it ramps toward."""

    def __init__(self) -> None:
        self.value = np.zeros(3, dtype=np.float32)  # what get_command returns (policy units: vx, vy, wz)
        self.goal = np.zeros(3, dtype=np.float64)

    def get_command(self) -> np.ndarray:  # DimOS InputController protocol
        return self.value.copy()

    def stop(self) -> None:  # DimOS InputController protocol
        pass


def _make_policy(spec: RobotSpec, default_angles: np.ndarray, cmd: _Command) -> Any:
    from dimos.simulation.mujoco.policy import G1OnnxController, Go1OnnxController

    params: dict[str, Any] = {
        "policy_path": str(policy_path(spec)),
        "default_angles": default_angles,
        "n_substeps": round(spec.ctrl_dt / spec.sim_dt),
        "action_scale": 0.5,
        "input_controller": cmd,
        "ctrl_dt": spec.ctrl_dt,
    }
    if spec.mujoco == "unitree_g1":
        policy = G1OnnxController(**params, drift_compensation=[-0.18, 0.0, -0.09])
    elif spec.mujoco == "unitree_go1":
        policy = Go1OnnxController(**params)
    else:
        raise ValueError(f"no DimOS walking policy for {spec.mujoco}")
    policy._policy = _single_thread_session(params["policy_path"])  # noqa: SLF001 - see the module docstring
    return policy


@dataclass
class Route:
    """A walk along corners [(x, y, yaw), ...] to the last one's pose; done/error set by the world."""

    legs: list[tuple[float, float, float]]
    speed: float
    index: int = 0
    reached: bool = False
    turn_t: float = 0.0
    done: bool = False
    error: str | None = None
    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    def finish(self, error: str | None = None) -> None:
        if not self.done:
            self.done = True
            self.error = error
            self._event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)


class Body:
    """One robot: its own model (scene + itself + proxies of the others), data, DimOS policy and controller."""

    def __init__(self, spec: RobotSpec, model: mujoco.MjModel, proxies: dict[str, list[tuple[int, str]]]) -> None:
        self.spec = spec
        self.model = model
        self.data = mujoco.MjData(model)
        self.n_substeps = round(spec.ctrl_dt / spec.sim_dt)
        if abs(self.n_substeps * spec.sim_dt - CTRL_DT) > 1e-12 or abs(spec.ctrl_dt - CTRL_DT) > 1e-12:
            raise ValueError(f"{spec.id}: policy period must be {CTRL_DT} s in whole physics steps")
        model.opt.timestep = spec.sim_dt
        key = model.keyframe("home").id
        mujoco.mj_resetDataKeyframe(model, self.data, key)
        self.home_qpos = np.array(model.key_qpos[key], dtype=np.float64)
        self.home_ctrl = np.array(model.key_ctrl[key], dtype=np.float64)
        self.cmd = _Command()
        self.vel = np.zeros(3)  # measured body-frame vx, vy, wz, low-passed (velocity tracking)
        self.track_i = np.zeros(3)  # velocity tracking integrator, command units
        self.hold = np.full(3, np.nan)  # velocity tracking: x, y, yaw held while the command is zero
        self.policy = _make_policy(spec, self.home_qpos[7:].copy(), self.cmd)
        self.proxies = proxies  # other robot id -> [(mocap id here, that robot's body name)]
        self.route: Route | None = None
        self.mode = "walk"  # walk | down
        self.down_t = 0.0
        self.falls = 0
        self.place(spec.x, spec.y, spec.yaw)

    # ---- state
    def place(self, x: float, y: float, yaw: float) -> None:
        d = self.data
        d.qpos[:] = self.home_qpos
        d.qpos[0:2] = [x, y]
        d.qpos[3:7] = _yaw_quat(yaw)
        d.qvel[:] = 0
        d.ctrl[:] = self.home_ctrl
        self.vel[:] = 0
        self.track_i[:] = 0
        self.hold[:] = np.nan
        self.policy._last_action[:] = 0  # noqa: SLF001
        self.policy._counter = 0  # noqa: SLF001
        if hasattr(self.policy, "_phase"):
            self.policy._phase = np.array([0.0, np.pi])  # noqa: SLF001
        mujoco.mj_forward(self.model, d)

    def pose(self) -> tuple[float, float, float]:
        d = self.data
        return float(d.qpos[0]), float(d.qpos[1]), _quat_yaw(d.qpos[3:7])

    def state_bytes(self) -> bytes:
        d = self.data
        return (d.qpos.tobytes() + d.qvel.tobytes() + d.ctrl.tobytes() + self.cmd.value.tobytes()
                + self.cmd.goal.tobytes() + self.vel.tobytes() + self.track_i.tobytes()
                + self.hold.tobytes())

    # ---- control (world lock held)
    def tick(self, world: SharedWorld) -> None:
        s = self.spec
        if self.mode == "down":
            self.down_t += CTRL_DT
            if self.down_t >= RECOVER_S:
                x, y, yaw = self.pose()
                self.place(x, y, yaw)
                self.mode = "walk"
                world.event(s.id, "recovered", x=x, y=y)
        elif float(self.data.qpos[2]) < s.fall_z:
            self.falls += 1
            self.mode = "down"
            self.down_t = 0.0
            self.cmd.goal[:] = 0
            self.cmd.value[:] = 0
            x, y, yaw = self.pose()
            if self.route is not None:
                self.route.finish("fell over")
                self.route = None
            world.event(s.id, "fell", x=x, y=y, yaw=yaw)
        else:
            if self.route is not None:
                self._steer(self.route)
                if self.route.done:
                    self.route = None
            goal = self._tracked_goal() if (s.comp.track_velocity and self.route is None) else self.cmd.goal
            lim = np.array([s.acc_vx, s.acc_vx, s.acc_wz]) * CTRL_DT
            self.cmd.value[:] = self.cmd.value + np.clip(goal - self.cmd.value, -lim, lim)
        for _ in range(self.n_substeps):
            self.policy.get_control(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

    def _tracked_goal(self) -> np.ndarray:
        out = np.array(self._tracked_goal_raw(), dtype=np.float64)  # (may be cmd.goal itself: never edit in place)
        if self.vel[0] > FWD_V:
            out[0] = max(out[0], NEUTRAL_VX)  # see NEUTRAL_VX
        return out

    def _tracked_goal_raw(self) -> np.ndarray:
        """cmd.goal as a wanted body velocity: policy command = goal / cmd_gain + PI on the measured velocity.
        A zero command holds the pose it was given at (the G1 policy otherwise creeps ~0.5 m per 100 s standing)."""
        s = self.spec
        d = self.data
        yaw = _quat_yaw(d.qpos[3:7])
        c, sn = math.cos(yaw), math.sin(yaw)
        meas = np.array([c * d.qvel[0] + sn * d.qvel[1], -sn * d.qvel[0] + c * d.qvel[1], d.qvel[5]])
        self.vel += TRACK_ALPHA * (meas - self.vel)  # the gait sways; average over ~0.2 s
        want = self.cmd.goal
        if not want.any():
            self.track_i[:] = 0
            x, y = float(d.qpos[0]), float(d.qpos[1])
            if np.isnan(self.hold[0]):
                if math.hypot(self.vel[0], self.vel[1]) > HOLD_SETTLED or abs(self.vel[2]) > 2.5 * HOLD_SETTLED:
                    return want  # still coming to a stop: hold where it stops, not where the command was
                self.hold[:] = [x, y, yaw]
            ex, ey = self.hold[0] - x, self.hold[1] - y
            hold = np.array([HOLD_KP * (c * ex + sn * ey) / s.cmd_gain, HOLD_KP * (-sn * ex + c * ey) / s.cmd_gain,
                             HOLD_KP * _wrap(float(self.hold[2]) - yaw)])
            return np.clip(hold, -HOLD_MAX, HOLD_MAX)
        self.hold[:] = np.nan
        err = want - self.vel
        self.track_i[:] = np.clip(self.track_i + TRACK_KI * err * CTRL_DT, -TRACK_I_MAX, TRACK_I_MAX)
        ff = want / np.array([s.cmd_gain, s.cmd_gain, 1.0])
        out = ff + TRACK_KP * err + self.track_i
        return np.clip(out, [-s.max_vx, -s.max_vx, -s.max_wz], [s.max_vx, s.max_vx, s.max_wz])

    def _steer(self, r: Route) -> None:
        """Go-to controller over the policy command: walk while turning toward the corner, turn first near the goal,
        pass corners without stopping, then turn to the final heading."""
        s = self.spec
        x, y, yaw = self.pose()
        tx, ty, tyaw = r.legs[r.index]
        last = r.index == len(r.legs) - 1
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        if not last and dist < VIA_M:
            r.index += 1  # a corner: on to the next one in this same tick, no stop
            r.reached = False
            self._steer(r)
            return
        if r.reached and dist > LEAVE_M:
            r.reached = False
        if not r.reached and last and dist < ARRIVE_M:
            r.reached = True
            r.turn_t = 0.0
        if not r.reached:
            err = _wrap(math.atan2(dy, dx) - yaw)
            if abs(err) > (0.9 if dist > TURN_FIRST_M else 0.35):
                vx = 0.0
            else:
                vx = float(np.clip(0.4 * dist, 0.08, r.speed)) * max(0.0, math.cos(err))
            wz = float(np.clip(1.2 * err, -s.max_wz, s.max_wz))
            cvx = min(vx / s.cmd_gain, s.max_vx)
            if cvx > 0:
                cvx = max(cvx, s.comp.min_vx)
            if abs(err) > 0.1:
                wz = math.copysign(max(abs(wz), s.comp.min_wz), wz)
            self.cmd.goal[:] = [cvx, 0.0, wz]
            return
        err = _wrap(tyaw - yaw)
        r.turn_t += CTRL_DT
        if abs(err) < FINAL_TURN_TOL or r.turn_t > FINAL_TURN_MAX_S:
            self.cmd.goal[:] = 0
            r.finish()
            return
        w = min(0.4, s.max_wz)
        wz = float(np.clip(1.0 * err, -w, w))
        self.cmd.goal[:] = [0.0, 0.0, math.copysign(max(abs(wz), s.comp.min_wz), wz)]


# ---------------------------------------------------------------- model building

def _robot_root(scene_xml: str, mujoco_name: str) -> ET.Element:
    root = ET.fromstring(scene_xml)
    root.set("model", f"{mujoco_name}_shared")
    root.insert(0, ET.Element("include", file=f"{mujoco_name}.xml"))
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    map_elem = visual.find("map")
    if map_elem is None:
        map_elem = ET.SubElement(visual, "map")
    map_elem.set("znear", "0.01")
    map_elem.set("zfar", "1000")
    return root


def _solo_model(mujoco_name: str, assets: dict[str, bytes]) -> mujoco.MjModel:
    # DimOS's G1 has foot/floor contact pairs, so the solo model needs a geom named "floor" (not copied to proxies).
    xml = (f'<mujoco model="{mujoco_name}_solo"><include file="{mujoco_name}.xml"/>'
           '<worldbody><geom name="floor" type="plane" size="1 1 0.1"/></worldbody></mujoco>')
    return mujoco.MjModel.from_xml_string(xml, assets=assets)


def _add_proxy(spec: mujoco.MjSpec, rid: str, solo: mujoco.MjModel) -> list[tuple[str, str]]:
    """Add every body of `solo` (another robot) as a mocap body with copies of its geoms. Returns
    [(proxy body name, solo body name)]."""
    meshes: dict[int, str] = {}
    out = []
    for b in range(1, solo.nbody):
        bname = solo.body(b).name or f"body{b}"
        pname = f"proxy/{rid}/{bname}"
        body = spec.worldbody.add_body(name=pname, mocap=True)
        for g in np.flatnonzero(solo.geom_bodyid == b):
            g = int(g)
            gtype = int(solo.geom_type[g])
            kw: dict[str, Any] = {
                "type": mujoco.mjtGeom(gtype),
                "size": solo.geom_size[g].copy(),
                "pos": solo.geom_pos[g].copy(),
                "quat": solo.geom_quat[g].copy(),
                "contype": int(solo.geom_contype[g]),
                "conaffinity": int(solo.geom_conaffinity[g]),
                "group": int(solo.geom_group[g]),
                "rgba": (solo.mat_rgba[solo.geom_matid[g]] if solo.geom_matid[g] >= 0 else solo.geom_rgba[g]).copy(),
                "condim": int(solo.geom_condim[g]),
                "friction": solo.geom_friction[g].copy(),
            }
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(solo.geom_dataid[g])
                if mid not in meshes:
                    va, vn = solo.mesh_vertadr[mid], solo.mesh_vertnum[mid]
                    fa, fn = solo.mesh_faceadr[mid], solo.mesh_facenum[mid]
                    mname = f"proxy/{rid}/mesh{mid}"
                    spec.add_mesh(name=mname, uservert=solo.mesh_vert[va:va + vn].flatten().tolist(),
                                  userface=solo.mesh_face[fa:fa + fn].flatten().tolist())
                    meshes[mid] = mname
                kw["meshname"] = meshes[mid]
            body.add_geom(**kw)
        out.append((pname, bname))
    return out


def build_body_models(scene_xml: str, specs: tuple[RobotSpec, ...]) -> dict[str, tuple[mujoco.MjModel, dict]]:
    """robot id -> (its model, {other id: [(mocap id, other's body name)]})."""
    assets = robot_assets({s.mujoco for s in specs})
    solos = {m: _solo_model(m, assets) for m in {s.mujoco for s in specs}}
    out = {}
    for spec in specs:
        root = _robot_root(scene_xml, spec.mujoco)
        xml = ET.tostring(root, encoding="unicode")
        mspec = mujoco.MjSpec.from_string(xml, include={k: v for k, v in assets.items() if k.endswith(".xml")},
                                          assets=assets)
        names: dict[str, list[tuple[str, str]]] = {}
        for other in specs:
            if other.id != spec.id:
                names[other.id] = _add_proxy(mspec, other.id, solos[other.mujoco])
        model = mspec.compile()
        proxies = {
            oid: [(int(model.body_mocapid[model.body(p).id]), b) for p, b in rows] for oid, rows in names.items()
        }
        out[spec.id] = (model, proxies)
    return out


# ---------------------------------------------------------------- the world

class SharedWorld:
    """The robots in `specs` in one scene, stepped together tick by tick on sim time."""

    def __init__(self, specs: tuple[RobotSpec, ...], scene_xml: str, *, scene_name: str = "custom",
                 run_log: RunLog | None = None, speed: float = 1.0) -> None:
        ids = [s.id for s in specs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"robot ids must be unique: {ids}")
        self.specs = tuple(specs)
        self.scene_xml = scene_xml
        self.scene_name = scene_name
        self.speed = speed
        self.lock = threading.RLock()
        self.ticks = 0
        self.log = run_log if run_log is not None else RunLog()
        models = build_body_models(scene_xml, self.specs)
        self.bodies: dict[str, Body] = {s.id: Body(s, *models[s.id]) for s in self.specs}
        # other robot's body ids, by proxy rows
        self._proxy_src = {
            rid: {oid: [(mid, self.bodies[oid].model.body(bname).id) for mid, bname in rows]
                  for oid, rows in body.proxies.items()}
            for rid, body in self.bodies.items()
        }
        self._pose_proxies()
        for body in self.bodies.values():
            mujoco.mj_forward(body.model, body.data)
        self.on_tick: list[Callable[[SharedWorld], None]] = []
        self.wall_lag_s = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.log.write({"t": "header", "format": FORMAT, "world": self.describe(), "versions": _versions()})

    # ---- description (what replay needs to rebuild this world)
    def describe(self) -> dict[str, Any]:
        return {
            "scene": self.scene_name,
            "scene_sha256": hashlib.sha256(self.scene_xml.encode()).hexdigest(),
            "robots": [s.to_dict() for s in self.specs],
            "ctrl_dt": CTRL_DT,
            "state_every_ticks": STATE_EVERY_TICKS,
        }

    @property
    def sim_time(self) -> float:
        return self.ticks * CTRL_DT

    def state_hash(self) -> str:
        h = hashlib.sha256()
        for rid in sorted(self.bodies):
            h.update(rid.encode())
            h.update(self.bodies[rid].state_bytes())
        return h.hexdigest()[:16]

    # ---- log
    def record(self, robot: str, op: str, **args: Any) -> None:
        """Log a command with the tick it lands on (lock held: it takes effect before tick `ticks` + 1)."""
        self.log.write({"t": "cmd", "tick": self.ticks, "robot": robot, "op": op, "args": args})

    def event(self, robot: str, what: str, **info: Any) -> None:
        log.info("%s %s at tick %d %s", robot, what, self.ticks, info)
        self.log.write({"t": "event", "tick": self.ticks, "robot": robot, "what": what, **info})

    # ---- commands (any thread)
    def _body(self, robot: str) -> Body:
        try:
            return self.bodies[robot]
        except KeyError:
            raise KeyError(f"no robot {robot!r}; have {sorted(self.bodies)}") from None

    def set_velocity(self, robot: str, vx: float, vy: float, wz: float) -> None:
        """Velocity command in policy units (DimOS cmd_vel semantics); cancels a route. Rate-limited."""
        vx, vy, wz = float(vx), float(vy), float(wz)
        with self.lock:
            body = self._body(robot)
            s = body.spec
            mpt = s.comp.min_pure_turn
            if mpt and math.hypot(vx, vy) < 0.05 and 0.05 < abs(wz) < mpt:
                wz_eff = math.copysign(mpt, wz)  # the logged command stays what was sent
            else:
                wz_eff = wz
            goal = [float(np.clip(vx, -s.max_vx, s.max_vx)), float(np.clip(vy, -s.max_vx, s.max_vx)),
                    float(np.clip(wz_eff, -max(s.max_wz, mpt), max(s.max_wz, mpt)))]
            if body.route is None and body.cmd.goal.tolist() == goal:
                return  # a stream repeating the same command changes nothing: not logged (replays the same)
            self.record(robot, "velocity", vx=vx, vy=vy, wz=wz)
            if body.route is not None:
                body.route.finish("superseded")
                body.route = None
            body.cmd.goal[:] = goal

    def walk_route(self, robot: str, legs: list[tuple[float, float, float]], speed: float = 0.5) -> Route:
        """Walk corners [(x, y, yaw), ...] ending at the last pose, as ONE command (see the module docstring)."""
        legs = [(float(x), float(y), float(yaw)) for x, y, yaw in legs]
        route = Route(legs=legs, speed=float(speed))
        with self.lock:
            body = self._body(robot)
            self.record(robot, "walk_route", legs=[list(leg) for leg in legs], speed=float(speed))
            if not legs:
                route.finish("empty route")
                return route
            if body.route is not None:
                body.route.finish("superseded")
            body.route = route
        return route

    def walk_to(self, robot: str, x: float, y: float, yaw: float, speed: float = 0.5) -> Route:
        """A* route on the warehouse floor plan (cell.route) from where the robot is, sent as one walk_route."""
        from dimos_worlds.warehouse.cell import route as plan

        with self.lock:
            legs = plan(self._body(robot).pose(), (float(x), float(y), float(yaw)))
            return self.walk_route(robot, legs, speed)

    def stop(self, robot: str) -> None:
        with self.lock:
            body = self._body(robot)
            self.record(robot, "stop")
            if body.route is not None:
                body.route.finish("stopped")
                body.route = None
            body.cmd.goal[:] = 0  # ramps down, like the robots' own stop

    def apply(self, cmd: dict[str, Any]) -> None:
        """Re-issue a logged command (replay)."""
        robot, op, a = cmd["robot"], cmd["op"], cmd.get("args", {})
        if op == "velocity":
            self.set_velocity(robot, a["vx"], a["vy"], a["wz"])
        elif op == "walk_route":
            self.walk_route(robot, [tuple(leg) for leg in a["legs"]], a["speed"])
        elif op == "stop":
            self.stop(robot)
        else:
            raise ValueError(f"unknown command {op!r}")

    # ---- queries
    def pose(self, robot: str) -> tuple[float, float, float]:
        with self.lock:
            return self._body(robot).pose()

    def poses(self) -> dict[str, tuple[float, float, float]]:
        with self.lock:
            return {rid: b.pose() for rid, b in self.bodies.items()}

    # ---- stepping
    def _pose_proxies(self) -> None:
        for rid, body in self.bodies.items():
            for oid, rows in self._proxy_src[rid].items():
                src = self.bodies[oid].data
                for mid, bid in rows:
                    body.data.mocap_pos[mid] = src.xpos[bid]
                    body.data.mocap_quat[mid] = src.xquat[bid]

    def tick(self) -> None:
        """One tick: proxies from everyone's state at the start, then each robot's own substeps."""
        with self.lock:
            self._pose_proxies()
            for rid in sorted(self.bodies):
                self.bodies[rid].tick(self)
            self.ticks += 1
            if self.ticks % STATE_EVERY_TICKS == 0:
                self.log.write({"t": "state", "tick": self.ticks, "hash": self.state_hash()})
        # Outside the lock: a callback that publishes must never block a command thread waiting on the lock. Only
        # this thread steps the world, so the state callbacks read here is the state this tick left.
        for fn in self.on_tick:
            fn(self)

    def snapshot(self, robot: str, into: mujoco.MjData) -> None:
        """Copy a robot's MjData (for rendering on another thread without holding up the world)."""
        with self.lock:
            body = self._body(robot)
            mujoco.mj_copyData(into, body.model, body.data)

    def step(self, n: int) -> None:
        for _ in range(n):
            self.tick()

    # ---- real-time loop
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="shared-world", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        """Pace ticks to `speed` x real time. Behind (a busy machine): tick as fast as possible, never skip or
        stretch a tick, so sim time just runs slower than asked."""
        next_wall = time.monotonic()
        w0, s0 = next_wall, self.sim_time
        while not self._stop.is_set():
            self.tick()
            next_wall += CTRL_DT / self.speed
            now = time.monotonic()
            lag = next_wall - now
            if lag > 0:
                time.sleep(lag)
            else:
                self.wall_lag_s = -lag
                if -lag > 1.0:
                    next_wall = now  # fell behind: do not catch up in a burst
            if now - w0 >= LAG_LOG_S:
                rate = (self.sim_time - s0) / (now - w0)
                if rate < 0.9 * self.speed:
                    log.info("world behind: %.2fx real time over %.0f s (asked %.1fx); sim steps unchanged", rate,
                             now - w0, self.speed)
                w0, s0 = now, self.sim_time

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.log.close()


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    import platform

    out = {"mujoco": mujoco.__version__, "platform": f"{platform.system()}-{platform.machine()}"}
    for dist in ("dimos", "onnxruntime", "numpy", "dimos-worlds"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            pass
    return out


def warehouse_world(specs: tuple[RobotSpec, ...] | None = None, *, run_log: RunLog | None = None,
                    speed: float = 1.0) -> SharedWorld:
    """The warehouse scene package with the given robots (default: Go2 + G1, robots.warehouse_fleet())."""
    from dimos_worlds.fleet.robots import warehouse_fleet
    from dimos_worlds.warehouse.scene import SCENE_XML

    return SharedWorld(specs or warehouse_fleet(), SCENE_XML.read_text(), scene_name="warehouse", run_log=run_log,
                       speed=speed)
