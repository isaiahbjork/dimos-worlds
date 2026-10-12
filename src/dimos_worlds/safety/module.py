"""The safety watchdog (safety.watchdog) as a DimOS module, for one robot.

    in   cmd_vel_in      Twist          the commands meant for the robot (from MovementManager, teleop, ...)
    in   imu             Imu            attitude (orientation); optional, odometry orientation is used without it
    in   odom            PoseStamped    freshness, and attitude while no IMU message has been seen
    in   safety_reset    Bool           True asks for a reset (same as the `reset` RPC)
    out  cmd_vel         Twist          what reaches the robot: cmd_vel_in, or zero while stopped
    out  safety_stopped  Bool           the latch, on every change and once a second
    out  safety_alert    String         one JSON object per alert: {"t", "kind", "message", "latched", ...}

Insert it between a robot's command source and its connection by remapping the source's output, e.g. for DimOS's
Go2 stack:

    autoconnect(..., SafetyWatchdog.blueprint()).remappings([
        (MovementManager, "cmd_vel", "cmd_vel_unsafe"),
        (SafetyWatchdog, "cmd_vel_in", "cmd_vel_unsafe"),
    ])

DimOS's Go2 connection keeps the Unitree lowstate (imu_state.rpy) to itself and DimOS's MuJoCo sim emits none, so
the `imu` input is an Imu message. A process that holds a lowstate stream can feed it with the `lowstate` RPC, or
with `Watchdog.lowstate` in-process. In-process callers can also add callbacks to `self.watchdog.on_alert`.

`guarded(blueprint, robot)` does that for a blueprint whose MovementManager drives one robot (go2-warehouse,
go2-lot-night). The shipped blueprints all insert the watchdog unless DIMOS_WORLDS_SAFETY=0 (`safety_default`).

Kinds, thresholds and the reset rules: safety.watchdog. The checks run at `rate_hz` on the wall clock.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from dimos_lcm.std_msgs import Bool
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.std_msgs.String import String
from dimos.utils.logging_config import setup_logger

from dimos_worlds.safety.watchdog import Alert, Watchdog, WatchdogConfig, roll_pitch

logger = setup_logger()


def safety_default() -> bool:
    """Blueprints insert the watchdog by default; DIMOS_WORLDS_SAFETY=0 (or false/off/no) leaves it out."""
    return os.environ.get("DIMOS_WORLDS_SAFETY", "1").strip().lower() not in ("0", "false", "off", "no")


class SafetyWatchdogConfig(ModuleConfig):
    max_tilt_deg: float = 40.0
    tilt_hold_s: float = 0.1
    upright_deg: float = 15.0
    cmd_timeout_s: float = 0.5
    odom_timeout_s: float = 0.5
    imu_timeout_s: float = 0.5
    rate_hz: float = 20.0
    robot: str = ""  # name used in log lines and alerts


def watchdog_config(c: SafetyWatchdogConfig) -> WatchdogConfig:
    import math

    return WatchdogConfig(max_tilt_rad=math.radians(c.max_tilt_deg), tilt_hold_s=c.tilt_hold_s,
                          upright_rad=math.radians(c.upright_deg), cmd_timeout_s=c.cmd_timeout_s,
                          odom_timeout_s=c.odom_timeout_s, imu_timeout_s=c.imu_timeout_s)


class SafetyWatchdog(Module):
    """Stops the robot on a fall, a lost command link or stale odometry/IMU, and keeps it stopped until reset."""

    config: SafetyWatchdogConfig
    cmd_vel_in: In[Twist]
    imu: In[Imu]
    odom: In[PoseStamped]
    safety_reset: In[Bool]
    cmd_vel: Out[Twist]
    safety_stopped: Out[Bool]
    safety_alert: Out[String]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._clock = time.monotonic
        self.watchdog = Watchdog(watchdog_config(self.config), on_alert=self._publish_alert)
        self._last_state_t = -1e9
        self._last_latched: bool | None = None

    @rpc
    def start(self) -> None:
        super().start()
        sub = self.register_disposable
        sub(Disposable(self.cmd_vel_in.subscribe(self._on_cmd)))
        sub(Disposable(self.imu.subscribe(self._on_imu)))
        sub(Disposable(self.odom.subscribe(self._on_odom)))
        sub(Disposable(self.safety_reset.subscribe(lambda m: self.reset() if m.data else None)))
        threading.Thread(target=self._watch, name="safety-watchdog", daemon=True).start()
        logger.info(f"safety watchdog up{' for ' + self.config.robot if self.config.robot else ''}")

    @rpc
    def stop(self) -> None:
        self._stop.set()
        super().stop()

    # ---- RPCs
    @rpc
    def reset(self) -> dict[str, Any]:
        with self._lock:
            ok, why = self.watchdog.reset(self._clock())
        self._tick()
        return {"cleared": ok, "why": why, **self.watchdog.status()}

    @rpc
    def status(self) -> dict[str, Any]:
        with self._lock:
            return self.watchdog.status()

    @rpc
    def lowstate(self, msg: dict[str, Any]) -> None:
        """A Unitree lowstate (DimOS LowStateMsg dict): its imu_state.rpy is the attitude."""
        with self._lock:
            self.watchdog.lowstate(msg, self._clock())
        self._tick()

    # ---- inputs
    def _on_cmd(self, msg: Twist) -> None:
        with self._lock:
            vx, vy, wz = self.watchdog.command(msg.linear.x, msg.linear.y, msg.angular.z, self._clock())
        self.cmd_vel.publish(Twist(linear=Vector3(vx, vy, 0.0), angular=Vector3(0.0, 0.0, wz)))

    def _on_imu(self, msg: Imu) -> None:
        q = msg.orientation
        with self._lock:
            self.watchdog.imu(*roll_pitch(q.x, q.y, q.z, q.w), self._clock())
        self._tick()

    def _on_odom(self, msg: PoseStamped) -> None:
        q = msg.orientation
        with self._lock:
            self.watchdog.odom(self._clock(), (q.x, q.y, q.z, q.w))

    # ---- loop
    def _watch(self) -> None:  # (not `_loop`: DimOS's Module keeps its event loop there)
        period = 1.0 / self.config.rate_hz
        while not self._stop.wait(period):
            try:
                self._tick()
            except Exception:
                logger.exception("safety watchdog tick failed")

    def _tick(self) -> None:
        now = self._clock()
        with self._lock:
            self.watchdog.step(now)
            latched = self.watchdog.latched
        if latched:
            self.cmd_vel.publish(Twist())  # hold the stop against anything downstream that keeps a last command
        if latched != self._last_latched or now - self._last_state_t >= 1.0:
            self._last_latched, self._last_state_t = latched, now
            self.safety_stopped.publish(Bool(data=latched))

    def _publish_alert(self, a: Alert) -> None:
        who = f"{self.config.robot}: " if self.config.robot else ""
        (logger.warning if a.latched or a.kind == "reset_refused" else logger.info)(f"safety: {who}{a.message}")
        row = a.to_dict() | ({"robot": self.config.robot} if self.config.robot else {})
        self.safety_alert.publish(String(json.dumps(row)))


def guarded(blueprint: Any, robot: str = "", safety: bool | None = None) -> Any:
    """`blueprint` with a SafetyWatchdog between its MovementManager's cmd_vel and the robot (cmd_vel_unsafe in
    between). safety=None: safety_default()."""
    from dimos.core.coordination.blueprints import autoconnect
    from dimos.navigation.movement_manager.movement_manager import MovementManager

    if not (safety_default() if safety is None else safety):
        return blueprint
    return autoconnect(blueprint, SafetyWatchdog.blueprint(robot=robot)).remappings([
        (MovementManager, "cmd_vel", "cmd_vel_unsafe"),
        (SafetyWatchdog, "cmd_vel_in", "cmd_vel_unsafe"),
    ])
