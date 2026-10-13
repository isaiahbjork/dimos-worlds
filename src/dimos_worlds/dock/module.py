"""`DroneDock`: the Dock 3-class station and its M4D-class quadcopter as a DimOS module (headless MuJoCo, real time).

    in   dock_command       String        "launch [alt_m]" | "land" | "open" | "close" | "goto x y z" | "hold"
    in   drone_cmd_vel      Twist         world-frame velocity: linear x/y/z (m/s), angular.z (yaw rate, rad/s);
                                          a dead-man: 0.5 s without one and the drone hovers
    in   safety_stopped     Bool          the safety watchdog's latch: True holds the drone where it is
    out  odom               PoseStamped   the drone's pose (world frame), 20 Hz
    out  dock_state         String        JSON: phase, covers, cover_deg, drone xyz, alt above pad, safety_stopped; 2 Hz
                                          and on every change
    out  dock_status        String        what happened: phases, refusals ("error: ..."), "ok <command>"
    out  dock_internal_image  Image       the dock's internal camera (fisheye by default), `feed_fps`
    out  dock_external_image  Image       the dock's external camera
    out  drone_image          Image       the drone's wide camera

The pilot (dock.pilot.DockPilot) runs launch and landing through the covers: covers open, climb, covers close once
the drone is 3 m clear; back over the pad, covers open, descend, touch down, covers close. Physics and the camera
renders share one thread (the GL context lives there); the loop paces the sim to `speed` x real time.

    dimos topic send /dock_command 'String("launch 4")'
    dimos topic echo /dock_state String
"""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

from dimos_lcm.std_msgs import Bool
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.std_msgs.String import String
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

FEED_OUTS = {"dock_internal": "dock_internal_image", "dock_external": "dock_external_image",
             "drone_wide": "drone_image"}


class DroneDockConfig(ModuleConfig):
    speed: float = 1.0  # x real time
    launch_alt_m: float = 4.0
    odom_hz: float = 20.0
    state_hz: float = 2.0
    feed_fps: float = 1.0  # frames per second per camera; 0 turns the cameras off
    feed_width: int = 1280
    feed_height: int = 720
    feed_lens: str = "fisheye"  # dock cameras: "fisheye" (sensor image) or "stream" (dewarped livestream view)
    feeds: list[str] = list(FEED_OUTS)


def parse_command(text: str) -> tuple[str, list[float]]:
    """'launch 4' -> ('launch', [4.0]). Raises ValueError on an unknown verb or a wrong argument count."""
    parts = text.replace(",", " ").split()
    if not parts:
        raise ValueError("empty command")
    verb, args = parts[0].lower(), [float(a) for a in parts[1:]]
    want = {"launch": (0, 1), "land": (0, 0), "open": (0, 0), "close": (0, 0), "hold": (0, 0), "goto": (3, 4)}
    if verb not in want:
        raise ValueError(f"unknown command {verb!r}; known: {sorted(want)}")
    lo, hi = want[verb]
    if not lo <= len(args) <= hi:
        raise ValueError(f"{verb} takes {lo}-{hi} numbers, got {len(args)}")
    return verb, args


def apply_command(pilot: Any, verb: str, args: list[float]) -> None:
    """Run a parsed command on a DockPilot (raises PilotError / RuntimeError on a refusal)."""
    if verb == "launch":
        pilot.launch(*args)
    elif verb == "land":
        pilot.land()
    elif verb == "open":
        pilot.open()
    elif verb == "close":
        pilot.close()
    elif verb == "hold":
        pilot.velocity(0.0, 0.0, 0.0, 0.0)
        if pilot.phase == "flying":
            pilot._move = None
    elif verb == "goto":
        pilot.goto(*args)


class DroneDock(Module):
    """Drone dock + quadcopter: launch/land through the covers, velocity and goto flight, dock and drone cameras."""

    config: DroneDockConfig
    dock_command: In[String]
    drone_cmd_vel: In[Twist]
    safety_stopped: In[Bool]
    odom: Out[PoseStamped]
    dock_state: Out[String]
    dock_status: Out[String]
    dock_internal_image: Out[Image]
    dock_external_image: Out[Image]
    drone_image: Out[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cmds: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {}
        self._ready = threading.Event()

    @rpc
    def start(self) -> None:
        super().start()
        sub = self.register_disposable
        sub(Disposable(self.dock_command.subscribe(lambda m: self._cmds.put(("cmd", str(m.data))))))
        sub(Disposable(self.drone_cmd_vel.subscribe(lambda m: self._cmds.put(("vel", m)))))
        sub(Disposable(self.safety_stopped.subscribe(lambda m: self._cmds.put(("stop", bool(m.data))))))
        self._start_sim()

    def _start_sim(self) -> None:
        """The sim thread (start() after wiring the inputs; tests call it directly)."""
        self._thread = threading.Thread(target=self._run, name="drone-dock", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=60.0)

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        super().stop()

    # ---- RPCs (same as the dock_command strings)
    @rpc
    def command(self, text: str) -> None:
        self._cmds.put(("cmd", text))

    @rpc
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    # ---- the sim thread
    def _status(self, text: str) -> None:
        logger.info(f"dock: {text}")
        self.dock_status.publish(String(text))

    def _handle(self, pilot: Any, kind: str, val: Any) -> None:
        from dimos_worlds.dock.pilot import PilotError

        if kind == "stop":
            pilot.stopped(val)
        elif kind == "vel":
            try:
                pilot.velocity(val.linear.x, val.linear.y, val.linear.z, val.angular.z)
            except PilotError as e:
                if not getattr(self, "_vel_refused", False):  # once per refusal streak, not at 10 Hz
                    self._status(f"error: {e}")
                self._vel_refused = True
            else:
                self._vel_refused = False
        else:
            try:
                verb, args = parse_command(val)
                apply_command(pilot, verb, args)
                self._status(f"ok {val.strip()}")
            except (ValueError, PilotError, RuntimeError) as e:
                self._status(f"error: {e}")

    def _run(self) -> None:
        from dimos_worlds.dock.pilot import DockPilot
        from dimos_worlds.dock.sim import DockSim

        c = self.config
        try:
            sim = DockSim()
            pilot = DockPilot(sim, launch_alt_m=c.launch_alt_m)
        except Exception:
            logger.exception("dock: could not build the dock model")
            self._ready.set()
            return
        feeds = [f for f in c.feeds if f in FEED_OUTS] if c.feed_fps > 0 else []
        for f in feeds:  # warm the renderers (GL context, first frame ~1 s) before odometry starts flowing
            self._publish_feed(sim, f)
        next_odom = next_state = 0.0
        next_feed = {f: 0.0 for f in feeds}
        last_key: tuple = ()
        self._ready.set()
        self._status("ready: drone parked, covers closed")
        wall0, sim0 = time.monotonic(), pilot.t
        while not self._stop.is_set():
            while True:
                try:
                    kind, val = self._cmds.get_nowait()
                except queue.Empty:
                    break
                self._handle(pilot, kind, val)
            pilot.tick()
            for e in pilot.events:
                self._status(e)
            pilot.events.clear()
            t = pilot.t
            if t >= next_odom:
                next_odom = t + 1.0 / c.odom_hz
                self._publish_odom(pilot)
            st = pilot.state()
            self._state = st
            key = (st["phase"], st["covers"], st["safety_stopped"])
            if t >= next_state or key != last_key:
                next_state = t + 1.0 / c.state_hz
                last_key = key
                self.dock_state.publish(String(json.dumps(st)))
            for f in feeds:
                if t >= next_feed[f]:
                    next_feed[f] = t + 1.0 / c.feed_fps
                    self._publish_feed(sim, f)
                    break  # one render per tick at most
            ahead = (pilot.t - sim0) / c.speed - (time.monotonic() - wall0)
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -1.0:  # fell behind (renders, a slow machine): don't sprint to catch up
                wall0, sim0 = time.monotonic(), pilot.t

    def _publish_odom(self, pilot: Any) -> None:
        x, y, z = (float(v) for v in pilot.pos)
        qx, qy, qz, qw = pilot.quat()
        self.odom.publish(PoseStamped(ts=time.time(), frame_id="world", position=Vector3(x, y, z),
                                      orientation=Quaternion(qx, qy, qz, qw)))

    def _publish_feed(self, sim: Any, feed: str) -> None:
        if getattr(self, "_feeds_off", False):
            return
        c = self.config
        try:
            rgb = sim.render(feed, c.feed_width, c.feed_height, None if feed == "drone_wide" else c.feed_lens)
        except Exception as e:  # noqa: BLE001 - no GL context (no EGL): run without cameras
            self._feeds_off = True
            self._status(f"cameras off: {e}")
            return
        getattr(self, FEED_OUTS[feed]).publish(
            Image.from_numpy(rgb, format=ImageFormat.RGB, frame_id=feed, ts=time.time()))
