"""Non-blocking launch / fly / land through the dock's covers, for a module that steps the sim in real time.

`DockSim.takeoff()` / `land()` block until done; `DockPilot` does the same sequences one control tick at a time, so
a DimOS module can keep publishing odometry and camera frames, and take new commands, while the drone flies:

    pilot = DockPilot(DockSim())
    pilot.launch(4.0)                  # open covers -> climb to 4 m above the pad -> close covers once 3 m clear
    while pilot.phase != "flying": pilot.tick()
    pilot.velocity(1.0, 0.0, 0.0)      # world-frame m/s (vx, vy, vz) + yaw rate; a dead-man: 0.5 s without one = hover
    pilot.goto(4.0, 2.0, 3.5)          # or a straight move to a point
    pilot.land()                       # back over the pad at 3 m -> open covers -> descend -> touch down -> close

Phases: parked -> opening -> climbing -> flying -> returning -> holding -> descending -> touchdown -> settling
-> parked. A launch climbs at least 3.5 m, so the covers can close under the drone.
The covers follow: they open for a launch or a landing, and close once the drone is 3 m above the pad (launch) or
on the pad with its motors off (landing). `DockSim.close()` refuses while the drone is in the cover sweep.

`stopped(True)` is the safety latch (the safety watchdog's `safety_stopped`): the drone holds where it is (velocity
and any goto dropped), and launch, goto and velocity are refused until it clears. Landing stays allowed: bringing
the drone home is the safe way out. Refusals raise `PilotError` with the reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .sim import CLEAR_ALT_M, DockSim

CTRL_STEPS = 5  # physics steps per control tick (10 ms at the dock's 2 ms)
MIN_ALT_M = 1.0  # flying: the reference never goes below this above the pad
MAX_RANGE_M = 60.0  # flying: the reference stays inside this radius of the pad (a geofence)
MAX_SPEED = (5.0, 2.0, 1.0)  # m/s horizontal, m/s vertical, rad/s yaw
CMD_TIMEOUT_S = 0.5  # sim seconds without a velocity command: hover
CLIMB_MPS, RETURN_MPS, DESCEND_MPS, TOUCH_MPS = 1.5, 3.0, 0.6, 0.05


class PilotError(RuntimeError):
    pass


@dataclass
class _Move:
    start: np.ndarray
    target: np.ndarray
    t0: float
    dur: float


class DockPilot:
    def __init__(self, sim: DockSim, launch_alt_m: float = 4.0) -> None:
        self.sim = sim
        self.phase = "parked"
        self.launch_alt = launch_alt_m
        self.latched = False
        self.events: list[str] = []  # what happened, newest last (the module publishes them)
        self._move: _Move | None = None
        self._vel = np.zeros(3)
        self._wz = 0.0
        self._vel_t = -1e9
        self._t_phase = 0.0
        self._close_after_launch = False
        self._dock_yaw = sim.yaw_ref

    # ------------------------------------------------------------ state
    @property
    def t(self) -> float:
        return float(self.sim.d.time)

    @property
    def pos(self) -> np.ndarray:
        return self.sim.d.xpos[self.sim.drone].copy()

    def quat(self) -> tuple[float, float, float, float]:
        """Drone orientation (x, y, z, w)."""
        w, x, y, z = (float(v) for v in self.sim.d.xquat[self.sim.drone])
        return x, y, z, w

    def state(self) -> dict:
        s = self.sim.state()
        p = self.pos
        return {"t": round(self.t, 3), "phase": self.phase, "covers": s.state, "cover_deg": round(s.cover_deg, 1),
                "drone": [round(float(v), 3) for v in p],
                "alt_above_pad": round(float(p[2] - self.sim.pad_pos[2]), 3),
                "pad": [round(float(v), 3) for v in self.sim.pad_pos], "safety_stopped": self.latched}

    def _event(self, text: str) -> None:
        self.events.append(text)

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self._t_phase = self.t
        self._event(f"phase {phase}")

    # ------------------------------------------------------------ commands
    def launch(self, alt_m: float | None = None) -> None:
        if self.latched:
            raise PilotError("safety stop is latched: reset it before a launch")
        if self.phase != "parked":
            raise PilotError(f"launch needs the drone parked (phase {self.phase})")
        self.launch_alt = max(float(alt_m if alt_m is not None else self.launch_alt), CLEAR_ALT_M + 0.5)
        self.sim.open()
        self._set_phase("opening")

    def land(self) -> None:
        if self.phase in ("parked", "returning", "holding", "descending", "touchdown"):
            raise PilotError(f"nothing to land (phase {self.phase})")
        hold = self.sim.pad_pos + np.array([0.0, 0.0, CLEAR_ALT_M])
        self._start_move(hold, RETURN_MPS)
        self._vel[:] = 0.0
        self._wz = 0.0
        self.sim.yaw_ref = self._dock_yaw  # land square to the dock
        self._set_phase("returning")

    def goto(self, x: float, y: float, z: float, speed: float = 2.0) -> None:
        self._need_flying("goto")
        self._start_move(self._fence(np.array([x, y, z], dtype=float)), min(speed, MAX_SPEED[0]))

    def velocity(self, vx: float, vy: float, vz: float = 0.0, wz: float = 0.0) -> None:
        """World-frame velocity (m/s) and yaw rate (rad/s). Holds for CMD_TIMEOUT_S of sim time, then hover."""
        moving = any(abs(v) > 1e-6 for v in (vx, vy, vz, wz))
        if not moving:  # zero is always accepted (it is what a latched watchdog sends)
            self._vel[:] = 0.0
            self._wz = 0.0
            return
        self._need_flying("velocity")
        h = math.hypot(vx, vy)
        k = min(1.0, MAX_SPEED[0] / h) if h > 0 else 1.0
        self._vel = np.array([vx * k, vy * k, float(np.clip(vz, -MAX_SPEED[1], MAX_SPEED[1]))])
        self._wz = float(np.clip(wz, -MAX_SPEED[2], MAX_SPEED[2]))
        self._vel_t = self.t
        self._move = None

    def open(self) -> None:
        self.sim.open()

    def close(self) -> None:
        if self.phase in ("opening", "holding", "descending", "touchdown"):
            raise PilotError(f"the covers are in use (phase {self.phase})")
        self.sim.close()  # raises while the drone is in the cover sweep

    def stopped(self, latched: bool) -> None:
        """The safety latch. On: hold position (drop velocity and goto)."""
        if latched and not self.latched:
            self._vel[:] = 0.0
            self._wz = 0.0
            if self.phase == "flying":
                self._move = None
                if self.sim.ref is not None:
                    self.sim.ref = self.pos
                    self.sim.ref_v = np.zeros(3)
            self._event("safety stop: holding")
        elif not latched and self.latched:
            self._event("safety stop cleared")
        self.latched = latched

    def _need_flying(self, what: str) -> None:
        if self.latched:
            raise PilotError(f"safety stop is latched: {what} refused")
        if self.phase != "flying":
            raise PilotError(f"{what} needs the drone flying (phase {self.phase})")

    def _fence(self, p: np.ndarray) -> np.ndarray:
        pad = self.sim.pad_pos
        d = p[:2] - pad[:2]
        r = float(np.linalg.norm(d))
        if r > MAX_RANGE_M:
            p[:2] = pad[:2] + d * (MAX_RANGE_M / r)
        p[2] = max(p[2], pad[2] + MIN_ALT_M)
        return p

    def _start_move(self, target: np.ndarray, speed: float) -> None:
        if self.sim.ref is None:
            self.sim.ref = self.pos
        start = self.sim.ref.copy()
        self._move = _Move(start, target.astype(float), self.t, max(float(np.linalg.norm(target - start)) / speed, 0.5))

    # ------------------------------------------------------------ the loop
    def _follow(self) -> bool:
        """Advance the current move's reference. True once the drone is at the target."""
        mv = self._move
        if mv is None:
            return True
        s = min(1.0, (self.t - mv.t0) / mv.dur)
        self.sim.ref = mv.start + (mv.target - mv.start) * (s * s * (3 - 2 * s))
        self.sim.ref_v = (mv.target - mv.start) * (6 * s * (1 - s)) / mv.dur
        if s >= 1.0 and np.linalg.norm(self.pos - mv.target) < 0.05:
            self._move = None
            return True
        return False

    def tick(self) -> None:
        """One control tick: phase logic, then CTRL_STEPS physics steps."""
        sim, ph = self.sim, self.phase
        st = sim.state().state
        if ph == "opening" and st == "open":
            self._start_move(sim.pad_pos + np.array([0.0, 0.0, self.launch_alt]), CLIMB_MPS)
            self._close_after_launch = True  # once the drone is clear of the sweep
            self._set_phase("climbing")
        elif ph == "climbing":
            if self._follow():
                self._set_phase("flying")
        elif ph == "flying":
            if self._move is not None:
                self._follow()
            elif sim.ref is not None:
                if self.t - self._vel_t > CMD_TIMEOUT_S:
                    self._vel[:] = 0.0
                    self._wz = 0.0
                dt = sim.m.opt.timestep * CTRL_STEPS
                sim.ref = self._fence(sim.ref + self._vel * dt)
                sim.ref_v = self._vel.copy()
                sim.yaw_ref += self._wz * dt
        elif ph == "returning":
            if self._follow():
                sim.open()
                self._set_phase("holding")
        elif ph == "holding" and st == "open":
            self._start_move(sim.pad_pos + np.array([0.0, 0.0, 0.02]), DESCEND_MPS)
            self._set_phase("descending")
        elif ph == "descending":
            if self._follow() or (self._move is not None and self.t - self._move.t0 > self._move.dur + 2.0):
                self._move = None
                self._set_phase("touchdown")
        elif ph == "touchdown":
            sim.ref = sim.ref - np.array([0.0, 0.0, TOUCH_MPS * sim.m.opt.timestep * CTRL_STEPS])
            if self.pos[2] < sim.pad_pos[2] + 0.004 or self.t - self._t_phase > 3.0:
                sim.ref = None  # motors off: the feet carry it
                self._set_phase("settling")
        elif ph == "settling" and self.t - self._t_phase > 1.0:
            sim.close()
            self._set_phase("parked")
        if self._close_after_launch and ph in ("climbing", "flying"):
            try:
                sim.close()
                self._close_after_launch = False
                self._event("covers closing behind the drone")
            except RuntimeError:
                pass  # still in the sweep: try again next tick
        sim.step(CTRL_STEPS)

    def run_until(self, phase: str, timeout_s: float = 60.0) -> float:
        """Tick until `phase` (tests, scripts). Returns the sim time it took."""
        t0 = self.t
        while self.phase != phase:
            if self.t - t0 > timeout_s:
                raise TimeoutError(f"still {self.phase} after {timeout_s} s (waiting for {phase})")
            self.tick()
        return self.t - t0
