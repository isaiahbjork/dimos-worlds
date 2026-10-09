"""Right-of-way rules (coord.traffic), without a world or planners: fast, pure."""
from __future__ import annotations

import math

import numpy as np
import pytest

from dimos_worlds.coord.traffic import (Agent, Arrived, Goal, SettleConfig, Stop, Traffic, TrafficConfig, Velocity,
                                        settle_command)

RES = 0.1


def _corridor() -> np.ndarray:
    """22 x 6 m: open bays at both ends (x < 2, x > 20), a 2.5 m aisle between racks in the middle (y 1.75-4.25)."""
    h, w = int(6 / RES), int(22 / RES)
    g = np.zeros((h, w), dtype=np.int8)
    ys = (np.arange(h) + 0.5) * RES
    xs = (np.arange(w) + 0.5) * RES
    gx, gy = np.meshgrid(xs, ys)
    g[(gx > 2) & (gx < 20) & ((gy < 1.75) | (gy > 4.25))] = 100
    g[(gx < 0.1) | (gx > 21.9) | (gy < 0.1) | (gy > 5.9)] = 100
    return g


def _line(x0: float, x1: float, y: float = 3.0) -> np.ndarray:
    n = int(abs(x1 - x0) / 0.1) + 1
    return np.stack([np.linspace(x0, x1, n), np.full(n, y)], axis=1)


def _traffic(cfg: TrafficConfig | None = None, settle: bool = False) -> Traffic:
    t = Traffic([Agent("a", 0.3, priority=2, settle=SettleConfig() if settle else None),
                 Agent("b", 0.35, priority=1)], cfg)
    t.set_map(_corridor(), 0.0, 0.0, RES)
    return t


def _of(actions: list, kind: type, robot: str) -> list:
    return [a for a in actions if isinstance(a, kind) and a.robot == robot]


def test_head_on_lower_priority_yields_to_a_pocket_off_the_winners_path() -> None:
    t = _traffic()
    t.update_pose("a", 3.0, 3.0, 0.0, 0.0)
    t.update_pose("b", 17.0, 3.0, math.pi, 0.0)
    assert _of(t.request_goal("a", 21.0, 3.0, 0.0, 0.0), Goal, "a")
    assert _of(t.request_goal("b", 1.0, 3.0, math.pi, 0.0), Goal, "b")
    t.on_path("a", _line(3.0, 21.0), 0.1)
    t.on_path("b", _line(17.0, 1.0), 0.1)
    out = t.step(0.2)
    assert t.status()["grants"] == ["a>b:yield"]
    (pocket,) = _of(out, Goal, "b")
    assert not _of(out, Goal, "a") and not _of(out, Stop, "a")  # the winner just keeps going
    # the pocket is off a's whole path by both radii + margins, and in the open bay behind b (the aisle is too narrow)
    assert abs(pocket.y - 3.0) >= 0.3 + 0.35 + t.cfg.pocket_margin_m - 1e-9
    assert pocket.x > 20.0
    # b sits in its pocket while a passes; once a is past, b gets its own goal back (nobody resends by hand)
    t.on_goal_reached("b", True, 5.0)
    t.update_pose("b", pocket.x, pocket.y, math.pi, 5.0)
    for k, x in enumerate(np.arange(3.0, 21.01, 0.5)):
        t.update_pose("a", float(x), 3.0, 0.0, 6.0 + k)
        assert not _of(t.step(6.0 + k), Goal, "b")  # a has not passed b's way back yet
    t.on_goal_reached("a", True, 50.0)  # a parks at its goal, clear of b's way back
    out = t.step(50.1)
    (g,) = _of(out, Goal, "b")
    assert (g.x, g.y) == (1.0, 3.0) and g.reason == "released from pocket"


def test_crossing_robot_already_on_the_path_goes_first_other_holds_then_walks_on() -> None:
    t = _traffic()
    # a walks north across the open bay at x=21; b walks east along y=3 and is already standing on a's line
    t.update_pose("a", 21.0, 1.5, math.pi / 2, 0.0)
    t.update_pose("b", 20.9, 3.0, 0.0, 0.0)
    t.request_goal("a", 21.0, 5.5, math.pi / 2, 0.0)
    t.request_goal("b", 21.6, 3.0, 0.0, 0.0)
    t.on_path("a", np.stack([np.full(41, 21.0), np.linspace(1.5, 5.5, 41)], axis=1), 0.0)
    t.on_path("b", _line(20.9, 21.6), 0.0)
    out = t.step(0.1)
    assert t.status()["grants"] == ["b>a:hold"]  # b is on a's path, a is not on b's: b goes although a ranks higher
    assert _of(out, Stop, "a")  # a is within stop_margin of the crossing already
    assert t.status()["a"]["mode"] == "hold"
    t.update_pose("b", 21.6, 3.0, 0.0, 1.0)
    t.on_goal_reached("b", True, 3.0)  # b parks at its goal, 0.6 m off a's line: still too close for a to pass
    out = t.step(3.1)
    assert t.status()["grants"] == ["a>b:yield"]  # parked on a's path: b moves aside, a walks on
    assert _of(out, Goal, "b") and _of(out, Goal, "a")


def test_neither_on_the_others_path_first_to_arrive_goes() -> None:
    t = _traffic()
    t.update_pose("a", 21.0, 0.5, math.pi / 2, 0.0)  # 2.5 m from the crossing at (21, 3)
    t.update_pose("b", 2.5, 3.0, 0.0, 0.0)  # 18.5 m from it
    t.request_goal("a", 21.0, 5.5, math.pi / 2, 0.0)
    t.request_goal("b", 21.5, 3.0, 0.0, 0.0)
    t.on_path("a", np.stack([np.full(51, 21.0), np.linspace(0.5, 5.5, 51)], axis=1), 0.0)
    t.on_path("b", _line(2.5, 21.5), 0.0)
    t.step(0.1)
    assert t.status()["grants"] == []  # b reaches the crossing beyond the horizon: no conflict yet
    t.update_pose("b", 15.0, 3.0, 0.0, 1.0)
    t.step(1.1)
    assert t.status()["grants"] == ["a>b:hold"]


def test_follower_keeps_its_distance() -> None:
    t = _traffic()
    t.update_pose("a", 8.0, 3.0, 0.0, 0.0)  # leader
    t.update_pose("b", 5.0, 3.0, 0.0, 0.0)
    t.request_goal("a", 21.0, 3.0, 0.0, 0.0)
    t.request_goal("b", 21.0, 1.0, 0.0, 0.0)
    t.on_path("a", _line(8.0, 21.0), 0.0)
    t.on_path("b", np.concatenate([_line(5.0, 21.0), [[21.0, 2.0], [21.0, 1.0]]]), 0.0)
    t.step(0.1)
    assert t.status()["grants"] == ["a>b:hold"]
    assert t.status()["b"]["mode"] == "go"  # 3 m behind: walks
    t.update_pose("b", 6.8, 3.0, 0.0, 1.0)
    assert _of(t.step(1.1), Stop, "b")  # too close behind: stops
    t.update_pose("a", 11.0, 3.0, 0.0, 2.0)
    assert _of(t.step(2.1), Goal, "b")  # leader moved on: walks on


def test_planner_giving_up_is_retried_without_anyone_resending() -> None:
    t = _traffic(TrafficConfig(retry_s=2.0))
    t.update_pose("a", 3.0, 3.0, 0.0, 0.0)
    t.update_pose("b", 1.0, 1.0, 0.0, 0.0)
    t.request_goal("a", 10.0, 3.0, 0.0, 0.0)
    assert t.on_goal_reached("a", False, 0.5) == []  # within stale_s of our own command: the planner's answer to it
    t.on_goal_reached("a", False, 5.0)
    assert not _of(t.step(6.0), Goal, "a")
    (g,) = _of(t.step(7.1), Goal, "a")
    assert g.reason == "retry" and (g.x, g.y) == (10.0, 3.0)


def test_deadlock_swaps_the_right_of_way() -> None:
    t = _traffic(TrafficConfig(deadlock_s=10.0))
    t.update_pose("a", 3.0, 3.0, 0.0, 0.0)
    t.update_pose("b", 17.0, 3.0, math.pi, 0.0)
    t.request_goal("a", 21.0, 3.0, 0.0, 0.0)
    t.request_goal("b", 1.0, 3.0, math.pi, 0.0)
    t.on_path("a", _line(3.0, 21.0), 0.0)
    t.on_path("b", _line(17.0, 1.0), 0.0)
    t.step(0.1)
    assert t.status()["grants"] == ["a>b:yield"]
    for k in range(120):  # a never moves (its planner is stuck): after deadlock_s the roles swap
        t.step(0.2 + 0.1 * k)
    assert t.status()["grants"] == ["b>a:yield"]
    assert any("deadlock" in text for _, text in t.events)


def test_arrival_settles_then_reports() -> None:
    t = _traffic(settle=True)
    t.update_pose("a", 5.0, 3.0, 0.0, 0.0)
    t.update_pose("b", 19.0, 5.0, 0.0, 0.0)
    t.request_goal("a", 5.3, 3.1, 0.5, 0.0)
    t.on_goal_reached("a", True, 2.0)  # the planner stops 0.3 m short and 29 degrees off
    assert t.status()["a"]["mode"] == "settle"
    (v,) = _of(t.step(2.1), Velocity, "a")
    assert v.vx > 0 and v.wz > 0
    t.update_pose("a", 5.25, 3.08, 0.47, 3.0)  # inside the tolerances
    t.step(3.0)
    out = t.step(3.7)
    assert _of(out, Arrived, "a") and _of(out, Velocity, "a")[-1] == Velocity("a", 0.0, 0.0, 0.0)
    assert t.status()["a"]["mode"] == "idle"


@pytest.mark.parametrize("translate", [True, False])
def test_settle_command_closes_position_and_heading(translate: bool) -> None:
    sc = SettleConfig(translate=translate)
    x, y, yaw = 0.0, 0.0, 0.0
    goal = (0.3, -0.2, 1.0)
    for _ in range(400):  # an ideal body following the command at 20 ms steps
        cmd = settle_command(sc, (x, y, yaw), goal)
        if cmd is None:
            break
        vx, vy, wz = cmd
        x += (math.cos(yaw) * vx - math.sin(yaw) * vy) * 0.02
        y += (math.sin(yaw) * vx + math.cos(yaw) * vy) * 0.02
        yaw += wz * 0.02
    assert cmd is None
    assert abs(yaw - 1.0) <= sc.yaw_tol
    if translate:
        assert math.hypot(x - 0.3, y + 0.2) <= sc.pos_tol
    else:
        assert (x, y) == (0.0, 0.0)


def test_module_class_has_one_port_set_per_robot() -> None:
    from typing import get_type_hints

    from dimos_worlds.coord.module import traffic_coordinator

    cls = traffic_coordinator(("r1", "r2", "r3"), "ThreeRobotTraffic")
    hints = get_type_hints(cls)
    for rid in ("r1", "r2", "r3"):
        for port in ("goal_request", "clicked_point", "odom", "path", "goal_reached", "nav_goal", "cmd_vel",
                     "arrived"):
            assert f"{rid}_{port}" in hints
    assert "global_costmap" in hints and cls.ROBOT_IDS == ("r1", "r2", "r3")
    assert cls.__module__ == __name__
