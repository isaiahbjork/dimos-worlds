"""Safety watchdog rules: what stops a robot, how the stop latches, and what a reset needs. Pure Python.

The watchdog sits between whoever commands a robot's velocity (a planner, teleop, a coordinator) and the robot.
Commands pass through unchanged until one of these trips it:

  fall       roll or pitch beyond `max_tilt_rad` for `tilt_hold_s` (from the IMU, e.g. Unitree lowstate
             imu_state.rpy; without an IMU, from the odometry orientation)
  link_lost  the last command asked for motion and no command has come for `cmd_timeout_s` (a dead-man: a sender
             that wants the robot to keep moving keeps sending, as DimOS's local planner does at 10 Hz)
  odom_stale no odometry for `odom_timeout_s` (once odometry has been seen)
  imu_stale  no IMU message for `imu_timeout_s` (once an IMU message has been seen)

A trip latches: from then on every command is replaced by zero velocity, whatever arrives, until `reset()`.
A reset is refused while the robot is still tilted past `upright_rad`, or odometry or the IMU are stale. After a
reset the last command is forgotten: the robot moves again only on a command sent after the reset.

Every trip, reset and refused reset is an `Alert`, given to each `on_alert` callback and returned by the call that
raised it. Time is the caller's (`now` in seconds): the DimOS module uses the wall clock, tests use sim time.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Any, Literal

AlertKind = Literal["fall", "link_lost", "odom_stale", "imu_stale", "reset", "reset_refused"]
TRIPS: tuple[str, ...] = ("fall", "link_lost", "odom_stale", "imu_stale")
ZERO = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class WatchdogConfig:
    max_tilt_rad: float = math.radians(40.0)  # roll or pitch past this is a fall ...
    tilt_hold_s: float = 0.1  # ... once it lasts this long (a single spiky sample is not)
    upright_rad: float = math.radians(15.0)  # a reset needs roll and pitch back under this
    cmd_timeout_s: float = 0.5  # motion asked for, then no command this long: link lost
    odom_timeout_s: float = 0.5
    imu_timeout_s: float = 0.5
    moving_eps: float = 1e-3  # a command component above this asks for motion


@dataclass(frozen=True)
class Alert:
    t: float
    kind: AlertKind
    message: str
    latched: bool  # the watchdog's state after this alert
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"t": self.t, "kind": self.kind, "message": self.message, "latched": self.latched, **self.data}


def roll_pitch(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float]:
    """Roll and pitch (rad) of an orientation quaternion (x, y, z, w)."""
    roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
    s = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    return roll, math.asin(s)


def rpy_from_lowstate(msg: Any) -> tuple[float, float]:
    """Roll and pitch from a Unitree lowstate: DimOS's LowStateMsg dict ({"data": {"imu_state": {"rpy": [...]}}}),
    its "data" part, or a DDS LowState object with `imu_state.rpy`."""
    if isinstance(msg, dict):
        data = msg.get("data", msg)
        rpy = data["imu_state"]["rpy"]
    else:
        rpy = msg.imu_state.rpy
    return float(rpy[0]), float(rpy[1])


class Watchdog:
    """Feed it IMU attitude, odometry and the commands on their way to the robot; call `step(now)` often (the
    module does at 20 Hz). `command()` returns what may go to the robot."""

    def __init__(self, config: WatchdogConfig | None = None,
                 on_alert: Callable[[Alert], None] | None = None) -> None:
        self.cfg = config or WatchdogConfig()
        self.on_alert: list[Callable[[Alert], None]] = [on_alert] if on_alert is not None else []
        self.latched = False
        self.reason: str | None = None
        self.latched_t: float | None = None
        self.alerts: list[Alert] = []
        self.roll = self.pitch = 0.0
        self.attitude_known = False
        self._imu_t: float | None = None
        self._odom_t: float | None = None
        self._cmd_t: float | None = None
        self._cmd: tuple[float, float, float] = ZERO
        self._vz = 0.0
        self._tilt_since: float | None = None

    # ---- inputs
    def imu(self, roll: float, pitch: float, now: float) -> list[Alert]:
        self._imu_t = now
        return self._attitude(roll, pitch, now)

    def lowstate(self, msg: Any, now: float) -> list[Alert]:
        return self.imu(*rpy_from_lowstate(msg), now)

    def odom(self, now: float, orientation: tuple[float, float, float, float] | None = None) -> list[Alert]:
        """Odometry arrived. Its orientation (x, y, z, w) is the attitude only while no IMU has been seen."""
        self._odom_t = now
        if orientation is not None and self._imu_t is None:
            return self._attitude(*roll_pitch(*orientation), now)
        return []

    def command(self, vx: float, vy: float, wz: float, now: float, vz: float = 0.0) -> tuple[float, float, float]:
        """A command on its way to the robot. Returns what to send: the command, or zero while latched. vz (a
        drone's climb rate) counts as motion too; `vz_out()` is what may go out for it."""
        self._cmd_t = now
        self._cmd = (float(vx), float(vy), float(wz))
        self._vz = float(vz)
        return ZERO if self.latched else self._cmd

    def vz_out(self) -> float:
        """The vertical command that may go to the robot: the last one, or zero while latched."""
        return 0.0 if self.latched else self._vz

    # ---- the loop
    def step(self, now: float) -> list[Alert]:
        c = self.cfg
        if self.latched:
            return []
        if self._moving() and self._cmd_t is not None and now - self._cmd_t > c.cmd_timeout_s:
            return self._trip("link_lost", now, f"no command for {now - self._cmd_t:.2f} s while moving",
                              silent_s=round(now - self._cmd_t, 3))
        if self._odom_t is not None and now - self._odom_t > c.odom_timeout_s:
            return self._trip("odom_stale", now, f"no odometry for {now - self._odom_t:.2f} s",
                              silent_s=round(now - self._odom_t, 3))
        if self._imu_t is not None and now - self._imu_t > c.imu_timeout_s:
            return self._trip("imu_stale", now, f"no IMU for {now - self._imu_t:.2f} s",
                              silent_s=round(now - self._imu_t, 3))
        return self._check_tilt(now)

    def reset(self, now: float) -> tuple[bool, str]:
        """Clear the latch if the robot is upright and its state is fresh. Returns (cleared, why)."""
        c = self.cfg
        problems = []
        if self.attitude_known and max(abs(self.roll), abs(self.pitch)) > c.upright_rad:
            problems.append(f"still tilted (roll {math.degrees(self.roll):.0f}, pitch "
                            f"{math.degrees(self.pitch):.0f} deg)")
        if self._odom_t is not None and now - self._odom_t > c.odom_timeout_s:
            problems.append("odometry stale")
        if self._imu_t is not None and now - self._imu_t > c.imu_timeout_s:
            problems.append("IMU stale")
        if problems:
            why = "; ".join(problems)
            self._alert(Alert(now, "reset_refused", f"reset refused: {why}", self.latched))
            return False, why
        was = self.reason
        self.latched, self.reason, self.latched_t = False, None, None
        self._cmd, self._vz, self._cmd_t = ZERO, 0.0, now  # a command from before the reset is not resumed
        self._tilt_since = None
        self._alert(Alert(now, "reset", f"reset (was: {was or 'not latched'})", False))
        return True, "ok"

    def status(self) -> dict[str, Any]:
        return {"latched": self.latched, "reason": self.reason, "since": self.latched_t,
                "roll_deg": round(math.degrees(self.roll), 1), "pitch_deg": round(math.degrees(self.pitch), 1)}

    # ---- internals
    def _moving(self) -> bool:
        return any(abs(v) > self.cfg.moving_eps for v in (*self._cmd, self._vz))

    def _attitude(self, roll: float, pitch: float, now: float) -> list[Alert]:
        self.roll, self.pitch = float(roll), float(pitch)
        self.attitude_known = True
        return [] if self.latched else self._check_tilt(now)

    def _check_tilt(self, now: float) -> list[Alert]:
        c = self.cfg
        if not self.attitude_known or max(abs(self.roll), abs(self.pitch)) <= c.max_tilt_rad:
            self._tilt_since = None
            return []
        if self._tilt_since is None:
            self._tilt_since = now
        if now - self._tilt_since < c.tilt_hold_s:
            return []
        return self._trip("fall", now, f"tilted: roll {math.degrees(self.roll):.0f}, pitch "
                          f"{math.degrees(self.pitch):.0f} deg", roll_deg=round(math.degrees(self.roll), 1),
                          pitch_deg=round(math.degrees(self.pitch), 1))

    def _trip(self, kind: AlertKind, now: float, message: str, **data: Any) -> list[Alert]:
        self.latched, self.reason, self.latched_t = True, kind, now
        a = Alert(now, kind, f"stopped: {message}; latched until reset", True, data)
        self._alert(a)
        return [a]

    def _alert(self, a: Alert) -> None:
        self.alerts.append(a)
        for fn in self.on_alert:
            fn(a)
