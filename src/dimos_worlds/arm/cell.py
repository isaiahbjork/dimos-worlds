"""The warehouse arm cell: a KUKA iiwa 14 on the pedestal by the conveyor, picking and placing one tote.

The arm is MuJoCo Menagerie's `kuka_iiwa_14` (BSD-3-Clause, from Drake), loaded from the Menagerie copy DimOS already
fetches (robots.menagerie_dir()); nothing of it is vendored here. It stands on the scene's pedestal (layout.json
`arm_cell`), base fixed: a static body in the warehouse, not a walking robot. A suction tool (a cylinder, `tool_m`
long) hangs below the flange.

Its own MuJoCo model and thread, not part of the shared world (fleet.world): the scene there must have no joints, and
the walking robots keep out of the arm cell anyway (it is a stay-out in places.json). So the walking robots do not
see the arm move, and arm runs are not in the shared world's run log.

Motion: positional IK (damped least squares on the tool tip position, tool axis held straight down, a pull toward
the home posture in the null space), then minimum-jerk joint interpolation between IK solutions, tracked by the
Menagerie model's own joint position servos with gravity compensation. The suction grip is kinematic: while gripped
the tote follows the tool rigidly; released, it drops the last centimetre and settles under physics.

    cell = ArmCell()
    cell.pick_place("conveyor", "pick_table")   # tote-09 from the conveyor end onto the pick table
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import threading

import mujoco
import numpy as np

from dimos_worlds.warehouse import cell as wh

ARM_PREFIX = "iiwa/"
HOME_Q = np.array([0.0, 0.785398, 0.0, -1.5708, 0.0, 0.0, 0.0])  # Menagerie keyframe "home"
SIM_DT = 0.002
JOINT_SPEED = 0.8  # rad/s, the fastest joint during a move
GRIP_GAP_M = 0.005  # the tool stops this far above a tote top to grip ...
PLACE_DROP_M = 0.01  # ... and lets go with the tote's base this high over the surface
TARGETS: dict[str, tuple[float, float, float]] = {k: (float(v[0]), float(v[1]), float(v[2]))
                                                    for k, v in wh.LAYOUT["arm_cell"]["reach_targets"].items()}


def iiwa_xml_path():
    from dimos_worlds.fleet.robots import menagerie_dir

    path = menagerie_dir() / "kuka_iiwa_14" / "iiwa14.xml"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: MuJoCo Menagerie's kuka_iiwa_14 is missing from DimOS's Menagerie copy")
    return path


def build_model(tote_id: str = "tote-09") -> mujoco.MjModel:
    """Warehouse scene + iiwa on the pedestal + `tote_id` made free (it rests where the scene puts it)."""
    from dimos_worlds.warehouse.scene import SCENE_XML

    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    spec.option.timestep = SIM_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    tote = spec.body(tote_id)
    if tote is None:
        raise KeyError(f"no tote {tote_id!r} in the scene")
    tote.add_freejoint(name=f"{tote_id}_free")
    for g in tote.geoms:
        g.mass = wh.TOTE_MASS_KG / max(1, len(tote.geoms))

    arm = mujoco.MjSpec.from_file(str(iiwa_xml_path()))
    for light in list(arm.lights):
        arm.delete(light)
    l7 = arm.body("link7")
    half = wh.ARM_TOOL_M / 2
    l7.add_geom(name="suction", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.025, half, 0],
                pos=[0, 0, 0.045 + half], rgba=[0.15, 0.15, 0.15, 1], contype=0, conaffinity=0, group=2)
    l7.add_site(name="tool_tip", pos=[0, 0, 0.045 + wh.ARM_TOOL_M], size=[0.01, 0, 0])
    for b in arm.bodies:
        b.gravcomp = 1.0
    bx, by, bz = wh.ARM_BASE
    yaw = math.radians(float(wh.LAYOUT["arm_cell"]["base_yaw_deg"]))
    frame = spec.worldbody.add_frame(pos=[bx, by, bz], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    spec.attach(arm, prefix=ARM_PREFIX, frame=frame)
    spec.add_exclude(bodyname1=f"{ARM_PREFIX}link7", bodyname2=tote_id)
    spec.add_exclude(bodyname1=f"{ARM_PREFIX}link6", bodyname2=tote_id)
    return spec.compile()


def _min_jerk(s: float) -> float:
    return s * s * s * (10 - 15 * s + 6 * s * s)


@dataclass
class Move:
    q0: np.ndarray
    q1: np.ndarray
    duration: float
    t: float = 0.0


class ArmCell:
    """The arm and one free tote in their own model. step() advances SIM_DT; the motion helpers step until done."""

    def __init__(self, tote_id: str = "tote-09") -> None:
        self.tote_id = tote_id
        self.model = build_model(tote_id)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joints = [m.joint(f"{ARM_PREFIX}joint{i}").id for i in range(1, 8)]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.joints])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.joints])
        self.acts = [m.actuator(f"{ARM_PREFIX}actuator{i}").id for i in range(1, 8)]
        self.lo = m.jnt_range[self.joints, 0]
        self.hi = m.jnt_range[self.joints, 1]
        self.tip = m.site(f"{ARM_PREFIX}tool_tip").id
        self.tote_body = m.body(tote_id).id
        fj = m.joint(f"{tote_id}_free").id
        self.tote_q = int(m.jnt_qposadr[fj])
        self.tote_v = int(m.jnt_dofadr[fj])
        self.data.qpos[self.qadr] = HOME_Q
        self.data.ctrl[self.acts] = HOME_Q
        mujoco.mj_forward(m, self.data)
        self._ik_data = mujoco.MjData(m)
        self.lock = threading.RLock()
        self.grip: tuple[np.ndarray, np.ndarray] | None = None  # tote pose in the tool frame (pos, quat)
        self.move: Move | None = None
        self.on_step: list[Callable[[ArmCell], None]] = []
        self.sim_time = 0.0
        self.settle(0.5)

    # ---- state
    def q(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    def tip_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tip].copy()

    def tote_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.qpos[self.tote_q:self.tote_q + 3].copy(), self.data.qpos[self.tote_q + 3:self.tote_q + 7].copy()

    # ---- IK
    def ik(self, target: np.ndarray, q_start: np.ndarray | None = None, iters: int = 300,
           tol: float = 1e-3, yaw: float = 0.0) -> tuple[np.ndarray, float]:
        """Joint angles putting the tool tip at `target` (world) with the tool pointing straight down and turned to
        `yaw` about the vertical (so a gripped tote keeps its heading). Returns (q, position error in m)."""
        m, d = self.model, self._ik_data
        d.qpos[:] = self.data.qpos
        q = (self.q() if q_start is None else np.array(q_start, dtype=float)).copy()
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        c, sn = math.cos(yaw), math.sin(yaw)
        want = np.array([[c, sn, 0.0], [sn, -c, 0.0], [0.0, 0.0, -1.0]])  # columns: tool x, y, z (z down)
        lam = 0.05
        err_p = np.inf
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.tip]
            R = d.site_xmat[self.tip].reshape(3, 3)
            e = np.concatenate([target - p, 0.5 * sum(np.cross(R[:, i], want[:, i]) for i in range(3))])
            err_p = float(np.linalg.norm(e[:3]))
            if err_p < tol and np.linalg.norm(e[3:]) < 1e-2:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, self.tip)
            J = np.vstack([jacp[:, self.dadr], 0.5 * jacr[:, self.dadr]])
            e[3:] *= 0.5
            JJt = J @ J.T + lam * lam * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            null = np.eye(7) - np.linalg.pinv(J) @ J
            dq += null @ (0.1 * (HOME_Q - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo + 1e-3, self.hi - 1e-3)
        return q, err_p

    # ---- stepping
    def step(self) -> None:
        with self.lock:
            mv = self.move
            if mv is not None:
                mv.t += SIM_DT
                s = min(1.0, mv.t / mv.duration)
                self.data.ctrl[self.acts] = mv.q0 + (mv.q1 - mv.q0) * _min_jerk(s)
                if s >= 1.0:
                    self.move = None
            mujoco.mj_step(self.model, self.data)
            if self.grip is not None:
                self._carry()
            self.sim_time += SIM_DT
        for fn in self.on_step:
            fn(self)

    def _carry(self) -> None:
        d = self.data
        rel_p, rel_q = self.grip
        tip_p = d.site_xpos[self.tip]
        tip_q = np.zeros(4)
        mujoco.mju_mat2Quat(tip_q, d.site_xmat[self.tip])
        p = tip_p + d.site_xmat[self.tip].reshape(3, 3) @ rel_p
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, tip_q, rel_q)
        d.qpos[self.tote_q:self.tote_q + 3] = p
        d.qpos[self.tote_q + 3:self.tote_q + 7] = q
        d.qvel[self.tote_v:self.tote_v + 6] = 0
        mujoco.mj_forward(self.model, d)

    def run(self, seconds: float) -> None:
        for _ in range(int(round(seconds / SIM_DT))):
            self.step()

    def settle(self, seconds: float) -> None:
        self.run(seconds)

    def move_to_q(self, q1: np.ndarray, *, wait: bool = True) -> None:
        q0 = self.data.ctrl[self.acts].copy()
        duration = max(0.8, float(np.max(np.abs(q1 - q0))) / JOINT_SPEED * 1.875)  # min-jerk peak = 1.875 x mean
        with self.lock:
            self.move = Move(q0=q0, q1=np.array(q1, dtype=float), duration=duration)
        if wait:
            while self.move is not None:
                self.step()
            self.run(0.3)  # let the servos catch up

    def solve(self, target: tuple[float, float, float] | np.ndarray, yaw: float = 0.0) -> tuple[np.ndarray, float]:
        """IK from a few starts, tool at `yaw` or `yaw` + pi (a tote is the same turned half way round): the
        solution with the least error, then the least joint motion from where the arm is commanded now."""
        now = self.data.ctrl[self.acts].copy()
        best: tuple[float, float, np.ndarray] | None = None
        for start in (now, HOME_Q):
            for y in (yaw, yaw + math.pi):
                q, err = self.ik(np.asarray(target, dtype=float), q_start=start, yaw=y)
                key = (round(err, 3), float(np.max(np.abs(q - now))), q)
                if best is None or key[:2] < best[:2]:
                    best = key
        assert best is not None
        return best[2], best[0]

    def move_tip(self, target: tuple[float, float, float] | np.ndarray, yaw: float = 0.0) -> float:
        """Tool tip to `target` (tool down). Returns the remaining tip error in m."""
        q1, err = self.solve(target, yaw)
        if err > 0.01:
            raise ValueError(f"target {tuple(np.round(target, 3))} out of reach (IK error {err:.3f} m)")
        self.move_to_q(q1)
        return float(np.linalg.norm(self.tip_pos() - np.asarray(target)))

    # ---- grip
    def grip_on(self) -> None:
        d = self.data
        tip_p = d.site_xpos[self.tip].copy()
        R = d.site_xmat[self.tip].reshape(3, 3).copy()
        tip_q = np.zeros(4)
        mujoco.mju_mat2Quat(tip_q, d.site_xmat[self.tip])
        tp, tq = self.tote_pose()
        if np.linalg.norm(tp[:2] - tip_p[:2]) > 0.08 or abs(tp[2] + wh.TOTE_SIZE[2] / 2 - tip_p[2]) > 0.03:
            raise RuntimeError(f"no tote under the tool (tote at {np.round(tp, 3)}, tool tip at {np.round(tip_p, 3)})")
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, tip_q)
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, inv, tq)
        with self.lock:
            self.grip = (R.T @ (tp - tip_p), rel_q)

    def grip_off(self) -> None:
        with self.lock:
            self.grip = None

    # ---- the task
    def pick_place(self, src: str, dst: str, *, log: Callable[[str], None] | None = None) -> np.ndarray:
        """Move the tote from reach target `src` to `dst` (layout.json arm_cell.reach_targets). Returns the tote's
        final position."""
        say = log or (lambda s: None)
        for name in (src, dst):
            if name not in TARGETS:
                raise KeyError(f"no reach target {name!r}; have {sorted(TARGETS)}")
            x, y, z = TARGETS[name]
            if not wh.arm_can_reach(x, y, z):
                raise ValueError(f"{name} is out of the arm's reach")
        clear = wh.ARM_CLEARANCE_M
        tp, _ = self.tote_pose()
        sx, sy, _ = TARGETS[src]
        top = tp[2] + wh.TOTE_SIZE[2] / 2
        if math.hypot(tp[0] - sx, tp[1] - sy) > 0.05:
            raise RuntimeError(f"the tote is not at {src} (it is at {np.round(tp, 3)})")
        dx, dy, dz = TARGETS[dst]
        bx, by, bz = wh.ARM_BASE
        tx, ty, tz = wh.ARM_TRANSIT
        say(f"above {src}")
        self.move_tip((tp[0], tp[1], top + clear))
        say(f"down onto the tote at {src}")
        self.move_tip((tp[0], tp[1], top + GRIP_GAP_M))
        self.grip_on()
        say("gripped; lifting")
        self.move_tip((tp[0], tp[1], top + clear))
        self.move_tip((bx + tx, by + ty, bz + tz))
        say(f"above {dst}")
        self.move_tip((dx, dy, dz + clear))
        self.move_tip((dx, dy, dz + PLACE_DROP_M))
        self.grip_off()
        say(f"released at {dst}")
        self.move_tip((dx, dy, dz + clear))
        self.move_to_q(HOME_Q)
        self.run(1.0)
        final, _ = self.tote_pose()
        say(f"tote at ({final[0]:.3f}, {final[1]:.3f}, {final[2]:.3f})")
        return final
