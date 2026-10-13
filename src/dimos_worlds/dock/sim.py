"""Run the Dock 3-class station and its M4D-class quadcopter in MuJoCo: covers, launch/landing, dock camera feeds.

    from dimos_worlds.dock.sim import DockSim
    sim = DockSim()                  # dock.xml (generate with python -m dimos_worlds.dock.gen_mjcf)
    sim.open(); sim.run_until_idle()  # 5 s cover travel (spec cover.travel_s)
    sim.takeoff(3.0); sim.land()
    jpeg = sim.feed_jpeg("dock_internal")

Cover state machine: closed -> opening -> open -> closing -> closed. A command in the middle of travel reverses
from where it is.
Linear actuators: a closed chain (cylinder hinge + rod slide + equality connect to the cover pin) follows the
cover in physics; set_cover_deg() poses it analytically for stills.
Drone: a point-mass-like quadrotor held by an external force/torque controller (thrust along body Z + attitude
PD), enough to launch from and land on the pad with real contact physics; it is not a flight-dynamics model.
Feeds: dock_internal, dock_external (equidistant fisheye, 151 deg diagonal, as DJI specifies; lens="stream"
gives the dewarped rectilinear view the livestream shows, spec cameras.*.stream) and drone_wide (rectilinear,
DJI 84 deg diagonal at 4:3). The fisheye is made by remapping a wide pinhole render.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from . import geometry as G

HERE = Path(__file__).resolve().parent
DEFAULT_XML = HERE / "dock.xml"
CLEAR_ALT_M = 3.0  # spec.json ops.close_clear_alt_m
FEEDS = ("dock_internal", "dock_external", "drone_wide")


def load_model(xml: Path | str = DEFAULT_XML) -> mujoco.MjModel:
    """Load a dock model, rebuilding the generated meshes first if the spec or shape code changed."""
    from .gen_mjcf import ensure_assets

    ensure_assets()
    return mujoco.MjModel.from_xml_path(str(xml))


@dataclass
class DockState:
    state: str  # closed | opening | open | closing
    cover_deg: float  # mean of both covers, 0 closed .. 90 open
    drone: str  # parked | airborne | landing | taking_off
    drone_pos: tuple[float, float, float]


class DockSim:
    def __init__(self, xml: Path | str = DEFAULT_XML, prefix: str = "", model: mujoco.MjModel | None = None) -> None:
        self.m = model if model is not None else load_model(xml)
        self.d = mujoco.MjData(self.m)
        self.spec = G.load_spec()
        p = prefix
        self.p = p
        self.j_r = self.m.joint(f"{p}cover_r_hinge").id
        self.j_l = self.m.joint(f"{p}cover_l_hinge").id
        self.j_act = {s: (self.m.joint(f"{p}act_{s}_swing").id, self.m.joint(f"{p}act_{s}_slide").id) for s in ("r", "l")}
        self.a_r = self.m.actuator(f"{p}cover_r_act").id
        self.a_l = self.m.actuator(f"{p}cover_l_act").id
        self.speed = float(self.m.actuator_ctrlrange[self.a_r][1])
        self.open_rad = float(self.m.actuator_actrange[self.a_r][1])
        self.drone = self.m.body(f"{p}drone").id
        self.mass = float(self.m.body_subtreemass[self.drone])
        jfree = self.m.joint(f"{p}drone_free")
        self.qadr = int(jfree.qposadr[0])
        self.vadr = int(jfree.dofadr[0])
        self.target_rad = 0.0
        self.ref: np.ndarray | None = None  # drone position reference (None = motors off)
        self.ref_v = np.zeros(3)
        mujoco.mj_forward(self.m, self.d)
        self.pad_pos = self.d.xpos[self.drone].copy()  # parked pose = initial pose
        R = self.d.xmat[self.drone].reshape(3, 3)
        self.yaw_ref = math.atan2(R[1, 0], R[0, 0])  # the dock's yaw: the drone keeps its parked heading
        self.drone_geoms = {g for g in range(self.m.ngeom) if self._in_drone(g)}
        self.max_pen = 0.0  # deepest drone contact seen (m, positive = penetration)
        self.bad_contacts: set[str] = set()  # drone contacts with anything but the pad / contacts / ground
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    @classmethod
    def from_model(cls, model: mujoco.MjModel, prefix: str = "") -> "DockSim":
        """A dock attached into another env (fixture.attach_dock) and compiled."""
        return cls(prefix=prefix, model=model)

    # ------------------------------------------------------------ helpers
    def _in_drone(self, g: int) -> bool:
        b = self.m.geom_bodyid[g]
        while b != 0:
            if b == self.drone:
                return True
            b = self.m.body_parentid[b]
        return False

    @property
    def cover_rad(self) -> tuple[float, float]:
        return float(self.d.qpos[self.m.jnt_qposadr[self.j_r]]), float(self.d.qpos[self.m.jnt_qposadr[self.j_l]])

    def state(self) -> DockState:
        r, left = self.cover_rad
        mean = (r + left) / 2
        moving = abs(self.d.ctrl[self.a_r]) > 0 or abs(self.d.ctrl[self.a_l]) > 0
        if moving:
            st = "opening" if self.d.ctrl[self.a_r] > 0 else "closing"
        else:
            st = "open" if mean > self.open_rad - math.radians(1) else ("closed" if mean < math.radians(1) else "partial")
        pos = tuple(float(v) for v in self.d.xpos[self.drone])
        if self.ref is None:
            dr = "parked" if np.linalg.norm(np.array(pos) - self.pad_pos) < 0.03 else "landed_off_pad"
        else:
            dr = "airborne"
        return DockState(st, math.degrees(mean), dr, pos)

    # ------------------------------------------------------------ covers
    def open(self) -> None:
        self.target_rad = self.open_rad

    def close(self) -> None:
        r, _ = self.cover_rad
        if self.ref is not None and self.d.xpos[self.drone][2] < self.pad_pos[2] + CLEAR_ALT_M and \
                abs(self.d.xpos[self.drone][0] - self.pad_pos[0]) < 1.2:
            raise RuntimeError("drone is in the cover sweep; refusing to close")
        self.target_rad = 0.0

    def _cover_ctrl(self) -> None:
        for j, a in ((self.j_r, self.a_r), (self.j_l, self.a_l)):
            act = float(self.d.act[self.m.actuator_actadr[a]])
            err = self.target_rad - act
            if abs(err) < 1e-9:
                self.d.ctrl[a] = 0.0
            else:
                # full speed until the last step, then exactly what is left: travel time = angle / speed
                self.d.ctrl[a] = math.copysign(min(self.speed, abs(err) / self.m.opt.timestep), err)

    # ------------------------------------------------------------ drone
    def _drone_ctrl(self) -> None:
        d, b = self.d, self.drone
        d.xfrc_applied[b] = 0.0
        if self.ref is None:
            return
        pos = d.xpos[b]
        vel = d.qvel[self.vadr:self.vadr + 3]
        g = -self.m.opt.gravity[2]
        acc = 6.0 * (self.ref - pos) + 4.5 * (self.ref_v - vel)
        acc[2] += g
        acc = np.clip(acc, [-4, -4, 2.0], [4, 4, 16.0])
        force = self.mass * acc
        # attitude: keep level, hold yaw (a real quad tilts to make lateral force; this proxy applies it directly)
        R = d.xmat[b].reshape(3, 3)
        w = d.qvel[self.vadr + 3:self.vadr + 6]
        z_body = R[:, 2]
        e_tilt = np.cross(z_body, [0, 0, 1])
        yaw = math.atan2(R[1, 0], R[0, 0])
        e_yaw = math.atan2(math.sin(self.yaw_ref - yaw), math.cos(self.yaw_ref - yaw))
        w_world = R @ w  # free-joint angular velocity is in the body frame; xfrc torque is in the world frame
        torque = 2.0 * e_tilt + np.array([0, 0, 0.6 * e_yaw]) - 0.25 * w_world
        d.xfrc_applied[b, :3] = force  # MuJoCo applies xfrc at the body's centre of mass
        d.xfrc_applied[b, 3:] = torque

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._cover_ctrl()
            self._drone_ctrl()
            mujoco.mj_step(self.m, self.d)
            self._watch_contacts()

    def _watch_contacts(self) -> None:
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in self.drone_geoms or g2 in self.drone_geoms:
                self.max_pen = max(self.max_pen, -float(c.dist))
                other = g2 if g1 in self.drone_geoms else g1
                name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                mine = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g1 if g1 in self.drone_geoms else g2) or ""
                ground = self.m.geom_type[other] == mujoco.mjtGeom.mjGEOM_PLANE
                if not (name.endswith("pad_plate") or "charge_contact" in name or ground) or \
                        "foot" not in mine:
                    self.bad_contacts.add(f"{mine}<->{name}")

    def run(self, seconds: float) -> None:
        self.step(int(round(seconds / self.m.opt.timestep)))

    def run_until_idle(self, timeout_s: float = 30.0) -> float:
        """Step until both covers reach their target. Returns the time it took (s)."""
        t0 = self.d.time
        while self.d.time - t0 < timeout_s:
            self.step(10)
            if all(abs(float(self.d.act[self.m.actuator_actadr[a]]) - self.target_rad) < 1e-6
                   for a in (self.a_r, self.a_l)):
                self.d.ctrl[self.a_r] = self.d.ctrl[self.a_l] = 0.0
                self.step(250)  # let the hinge settle on the activation target
                return self.d.time - t0 - 250 * self.m.opt.timestep
        raise TimeoutError("covers did not reach their target")

    def _fly_to(self, target: np.ndarray, speed: float, tol: float = 0.03, timeout: float = 60.0) -> None:
        if self.ref is None:
            self.ref = self.d.xpos[self.drone].copy()
        start = self.ref.copy()
        dist = float(np.linalg.norm(target - start))
        dur = max(dist / speed, 0.5)
        t0 = self.d.time
        while self.d.time - t0 < dur:
            s = (self.d.time - t0) / dur
            sm = s * s * (3 - 2 * s)  # smoothstep
            self.ref = start + (target - start) * sm
            self.ref_v = (target - start) * (6 * s * (1 - s)) / dur
            self.step(5)
        self.ref, self.ref_v = target.copy(), np.zeros(3)
        while np.linalg.norm(self.d.xpos[self.drone] - target) > tol:
            if self.d.time - t0 > timeout:
                raise TimeoutError("drone did not reach its reference")
            self.step(5)

    def takeoff(self, alt_above_pad: float = 3.0, speed: float = 1.5) -> None:
        if self.state().state != "open":
            raise RuntimeError("covers are not open: the drone cannot leave the dock")
        self._fly_to(self.pad_pos + np.array([0, 0, alt_above_pad]), speed)

    def fly_to(self, x: float, y: float, z: float, speed: float = 4.0) -> None:
        self._fly_to(np.array([x, y, z], dtype=float), speed)

    def land(self, speed: float = 0.6) -> None:
        if self.state().state != "open":
            raise RuntimeError("covers are not open: the drone cannot land in the dock")
        above = self.pad_pos + np.array([0, 0, 1.0])
        self._fly_to(above, 2.0)
        self._fly_to(self.pad_pos + np.array([0, 0, 0.02]), speed, tol=0.025)
        # final touchdown: lower the reference slowly until the feet carry the weight, then motors off
        t0 = self.d.time
        while self.d.time - t0 < 3.0:
            self.ref = self.ref - np.array([0, 0, 0.05 * self.m.opt.timestep * 5])
            self.step(5)
            if self.d.xpos[self.drone][2] < self.pad_pos[2] + 0.004:
                break
        self.ref = None
        self.run(1.0)

    # ------------------------------------------------------------ feeds
    def _renderer(self, w: int, h: int) -> mujoco.Renderer:
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.m, height=h, width=w)
        return self._renderers[key]

    def render(self, feed: str, width: int = 1280, height: int = 720, lens: str | None = None) -> np.ndarray:
        """A camera frame. Dock cameras: lens "fisheye" (DJI's 151 deg equidistant sensor image, the default) or
        "stream" (the dewarped rectilinear view the livestream shows, spec cameras.<feed>.stream)."""
        cam = self.m.camera(f"{self.p}{feed}").id
        if feed == "drone_wide":
            r = self._renderer(width, int(width * 3 / 4))
            r.update_scene(self.d, camera=cam)
            return r.render()
        c = self.spec["cameras"][feed]
        if (lens or "fisheye") == "stream":
            hfov = c["stream"]["hfov_deg"]
            fovy = math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) * height / width))
            old = float(self.m.cam_fovy[cam])
            self.m.cam_fovy[cam] = fovy
            r = self._renderer(width, height)
            r.update_scene(self.d, camera=cam)
            img = r.render()
            self.m.cam_fovy[cam] = old
            return img
        return self._fisheye(cam, width, height, dfov_deg=c["dfov_deg"])

    def _fisheye(self, cam: int, w: int, h: int, dfov_deg: float) -> np.ndarray:
        """Equidistant fisheye with the given diagonal FOV, remapped from a wide pinhole render."""
        th_max = math.radians(dfov_deg / 2)
        r_diag = math.hypot(w, h) / 2
        # pinhole wide enough to hold the fisheye corners
        cx = math.tan(th_max) * (w / 2) / r_diag  # direction tangent at the corner, x part
        cy = math.tan(th_max) * (h / 2) / r_diag
        pw, ph = min(self.m.vis.global_.offwidth, 2560), min(self.m.vis.global_.offheight, 1440)
        tan_y = max(cy, cx * ph / pw) * 1.02
        fovy = math.degrees(2 * math.atan(tan_y))
        old = float(self.m.cam_fovy[cam])
        self.m.cam_fovy[cam] = fovy
        r = self._renderer(pw, ph)
        r.update_scene(self.d, camera=cam)
        img = r.render()
        self.m.cam_fovy[cam] = old
        u, v = np.meshgrid(np.arange(w) - (w - 1) / 2, np.arange(h) - (h - 1) / 2)
        rr = np.hypot(u, v)
        th = rr / r_diag * th_max
        phi = np.arctan2(v, u)
        t = np.tan(np.minimum(th, math.radians(89)))
        px = t * np.cos(phi)
        py = t * np.sin(phi)
        f = (ph / 2) / tan_y
        sx = np.clip((px * f + (pw - 1) / 2).round().astype(int), 0, pw - 1)
        sy = np.clip((py * f + (ph - 1) / 2).round().astype(int), 0, ph - 1)
        return img[sy, sx]

    def feed_jpeg(self, feed: str, width: int = 1280, height: int = 720, quality: int = 85,
                  lens: str | None = None) -> bytes:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(self.render(feed, width, height, lens)).save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    def linkage(self, deg: float, side: str = "r") -> tuple[float, float]:
        """Actuator swing (rad) and rod stroke (m) that put the rod's eye on the cover pin at a cover angle."""
        bp, cp = G.actuator_pins(self.spec, side)
        _, _, d1 = G.actuator_pose(self.spec, deg, side)
        d0 = G._norm(G._sub(cp, bp))
        swing = math.atan2(d1[0], d1[2]) - math.atan2(d0[0], d0[2])
        _, cpw, _ = G.actuator_pose(self.spec, deg, side)
        stroke = math.dist(cpw, bp) - math.dist(cp, bp)
        return swing, stroke

    def set_cover_deg(self, deg: float) -> None:
        """Teleport both covers (stills, tests): joint angle, actuator activation and the linkage pose."""
        rad = math.radians(deg)
        for j, a in ((self.j_r, self.a_r), (self.j_l, self.a_l)):
            self.d.qpos[self.m.jnt_qposadr[j]] = rad
            self.d.act[self.m.actuator_actadr[a]] = rad
        for side, (js, jl) in self.j_act.items():
            swing, stroke = self.linkage(deg, side)
            self.d.qpos[self.m.jnt_qposadr[js]] = swing
            self.d.qpos[self.m.jnt_qposadr[jl]] = stroke
        self.target_rad = rad
        mujoco.mj_forward(self.m, self.d)
