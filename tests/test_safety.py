"""Safety watchdog rules (safety.watchdog) and its DimOS module, without a world."""
from __future__ import annotations

import json
import math

import pytest

from dimos_worlds.safety.watchdog import Watchdog, WatchdogConfig, roll_pitch, rpy_from_lowstate

FWD = (0.5, 0.0, 0.0)


def _fresh(w: Watchdog, t: float) -> None:
    w.imu(0.0, 0.0, t)
    w.odom(t)


def test_commands_pass_until_a_trip_then_zero_whatever_arrives() -> None:
    w = Watchdog()
    _fresh(w, 0.0)
    assert w.command(*FWD, 0.0) == FWD
    w.imu(math.radians(60), 0.0, 0.05)
    assert w.step(0.1) == []  # tilted for 0.05 s: not yet (a spike is not a fall)
    (a,) = w.step(0.16)
    assert a.kind == "fall" and a.latched and w.latched
    assert w.command(*FWD, 0.2) == (0.0, 0.0, 0.0)


def test_walking_tilt_is_not_a_fall() -> None:
    w = Watchdog()
    for i in range(100):
        t = i * 0.02
        w.odom(t)
        w.imu(math.radians(6) * math.sin(i), math.radians(6) * math.cos(i), t)
        w.command(*FWD, t)
        assert w.step(t) == []


def test_link_lost_only_while_motion_was_asked_for() -> None:
    w = Watchdog()
    _fresh(w, 0.0)
    w.command(0.0, 0.0, 0.0, 0.0)  # a stop, then silence: fine
    for t in (0.2, 0.4, 0.6, 1.0):
        _fresh(w, t)
        assert w.step(t) == []
    w.command(*FWD, 1.0)
    _fresh(w, 1.4)
    assert w.step(1.4) == []
    _fresh(w, 1.6)
    (a,) = w.step(1.6)
    assert a.kind == "link_lost" and a.data["silent_s"] == pytest.approx(0.6)


def test_stale_odom_and_imu_trip_once_seen() -> None:
    w = Watchdog()
    assert w.step(10.0) == []  # nothing seen yet: nothing to be stale
    w.odom(10.0)
    w.imu(0.0, 0.0, 10.0)
    w.imu(0.0, 0.0, 10.6)
    (a,) = w.step(10.6)
    assert a.kind == "odom_stale"
    w2 = Watchdog()
    w2.imu(0.0, 0.0, 0.0)
    w2.odom(0.6)
    assert [x.kind for x in w2.step(0.6)] == ["imu_stale"]


def test_latch_needs_a_reset_and_the_reset_needs_an_upright_robot_with_fresh_state() -> None:
    seen = []
    w = Watchdog(on_alert=seen.append)
    _fresh(w, 0.0)
    w.imu(math.radians(90), 0.0, 0.0)
    w.step(0.2)
    assert w.latched
    _fresh(w, 5.0)  # stood back up, but...
    w.imu(math.radians(30), 0.0, 5.0)  # ... still leaning
    ok, why = w.reset(5.0)
    assert not ok and "tilted" in why and w.latched
    ok, why = w.reset(9.0)  # upright now? no: and everything went stale meanwhile
    assert not ok and "odometry stale" in why and "IMU stale" in why
    _fresh(w, 9.0)
    assert w.reset(9.0) == (True, "ok") and not w.latched
    assert [a.kind for a in seen] == ["fall", "reset_refused", "reset_refused", "reset"]
    # the command from before the stop is not resumed: no link_lost for it, nothing moves until a new command
    _fresh(w, 10.0)
    assert w.step(10.0) == []
    assert w.command(*FWD, 10.0) == FWD


def test_attitude_from_odometry_until_an_imu_speaks() -> None:
    w = Watchdog(WatchdogConfig(tilt_hold_s=0.0))
    q = (math.sin(math.radians(50) / 2), 0.0, 0.0, math.cos(math.radians(50) / 2))  # 50 deg roll
    assert [a.kind for a in w.odom(0.0, q)] == ["fall"]
    w2 = Watchdog(WatchdogConfig(tilt_hold_s=0.0))
    w2.imu(0.0, 0.0, 0.0)
    assert w2.odom(0.0, q) == [] and w2.step(0.0) == []  # an IMU was seen: odometry orientation is ignored


def test_roll_pitch_and_unitree_lowstate() -> None:
    r, p = roll_pitch(0.0, math.sin(0.3 / 2), 0.0, math.cos(0.3 / 2))
    assert r == pytest.approx(0.0, abs=1e-12) and p == pytest.approx(0.3)
    msg = {"type": "msg", "topic": "rt/lf/lowstate", "data": {"imu_state": {"rpy": [0.008, -0.75, 2.98]}}}
    assert rpy_from_lowstate(msg) == (0.008, -0.75)
    w = Watchdog(WatchdogConfig(max_tilt_rad=math.radians(40), tilt_hold_s=0.0))
    w.lowstate(msg, 0.0)
    assert w.latched and w.reason == "fall"


def test_module_gates_cmd_vel_and_publishes_alerts_and_the_latch() -> None:
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from dimos.msgs.geometry_msgs.Quaternion import Quaternion
    from dimos.msgs.geometry_msgs.Twist import Twist
    from dimos.msgs.geometry_msgs.Vector3 import Vector3
    from dimos.msgs.sensor_msgs.Imu import Imu

    from dimos_worlds.safety.module import SafetyWatchdog

    m = SafetyWatchdog(robot="go2")
    now = [0.0]
    m._clock = lambda: now[0]
    cmds, alerts, stopped = [], [], []
    m.cmd_vel.subscribe(cmds.append)
    m.safety_alert.subscribe(lambda s: alerts.append(json.loads(s.data)))
    m.safety_stopped.subscribe(lambda b: stopped.append(b.data))
    try:
        m._on_odom(PoseStamped(position=Vector3(0, 0, 0), orientation=Quaternion(0, 0, 0, 1)))
        m._on_imu(Imu())
        m._on_cmd(Twist(linear=Vector3(0.4, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.2)))
        assert (cmds[-1].linear.x, cmds[-1].angular.z) == (0.4, 0.2)
        now[0] = 0.7  # the sender went quiet mid-walk; odometry and IMU still fresh
        m._on_odom(PoseStamped(position=Vector3(0, 0, 0), orientation=Quaternion(0, 0, 0, 1)))
        m._on_imu(Imu())
        assert alerts[-1]["kind"] == "link_lost" and alerts[-1]["robot"] == "go2"
        assert stopped[-1] is True and cmds[-1].linear.x == 0.0
        m._on_cmd(Twist(linear=Vector3(0.4, 0.0, 0.0)))
        assert cmds[-1].linear.x == 0.0  # latched
        assert m.reset()["cleared"] and stopped[-1] is False
        m._on_cmd(Twist(linear=Vector3(0.4, 0.0, 0.0)))
        assert cmds[-1].linear.x == 0.4
    finally:
        m.stop()


def test_module_slots_between_a_go2_stack_and_its_connection() -> None:
    """The wiring the README shows: MovementManager's cmd_vel goes to the watchdog, the watchdog's to the robot."""
    from dimos.core.coordination.blueprints import autoconnect
    from dimos.navigation.movement_manager.movement_manager import MovementManager

    from dimos_worlds.safety.module import SafetyWatchdog

    bp = autoconnect(MovementManager.blueprint(), SafetyWatchdog.blueprint(robot="go2")).remappings([
        (MovementManager, "cmd_vel", "cmd_vel_unsafe"),
        (SafetyWatchdog, "cmd_vel_in", "cmd_vel_unsafe"),
    ])
    assert {a.module for a in bp.blueprints} == {MovementManager, SafetyWatchdog}
