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
    assert t.status()["grants"] == ["a>b:yield"]  # parked on a's path: b moves aside ...
    assert _of(out, Goal, "b") and not _of(out, Goal, "a")  # ... a, 0.6 m from b's body, waits (close quarters)
    t.update_pose("b", 21.6, 4.5, 0.0, 4.0)  # b in its pocket
    out = [a for k in range(41, 70) for a in t.step(0.1 * k)]
    assert _of(out, Goal, "a")  # a walks on once b has been clear guard_release_s


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
    t.update_pose("b", 6.8, 3.0, 0.0, 2.0)
    assert [a for k in range(21, 40) for a in t.step(0.1 * k) if isinstance(a, Goal) and a.robot == "b"]  # walks on


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


def _open_floor() -> np.ndarray:
    g = np.zeros((int(10 / RES), int(10 / RES)), dtype=np.int8)
    g[[0, -1], :] = 100
    g[:, [0, -1]] = 100
    return g


def test_ring_of_four_each_bound_for_the_next_ones_spot_breaks_the_hold_cycle() -> None:
    """Every robot's goal is where the next one stands: pairwise, each holds for the next (a cycle nobody leaves).
    After cycle_s the lowest priority robot steps aside for the one holding for it; the ring unwinds."""
    corners = [(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]
    ids = ["a", "b", "c", "d"]
    t = Traffic([Agent(rid, 0.3, priority=4 - i) for i, rid in enumerate(ids)])
    t.set_map(_open_floor(), 0.0, 0.0, RES)
    for i, rid in enumerate(ids):
        t.update_pose(rid, *corners[i], 0.0, 0.0)
    for i, rid in enumerate(ids):
        t.request_goal(rid, *corners[(i + 1) % 4], 0.0, 0.0)
    t.step(0.1)
    assert sorted(t.status()["grants"]) == ["a>d:hold", "b>a:hold", "c>b:hold", "d>c:hold"]
    out: list = []
    for k in range(2, 80):  # nobody moves (each is blocked by the robot at its goal)
        out += t.step(0.1 * k)
    assert not any("hold cycle" in e for te, e in t.events if te < 5.0)
    # a waits for b, b for c, c for d, d for a: d (lowest priority) steps aside for c, which is waiting on it
    assert any("hold cycle a > b > c > d > a: d steps aside for c" in e for _, e in t.events)
    assert "c>d:yield" in t.status()["grants"] and t.status()["d"]["mode"] == "yield"
    (pocket,) = _of(out, Goal, "d")
    # the pocket is off every other robot's way: a's, b's and c's straight runs to their next corners
    for i in range(3):
        (x0, y0), (x1, y1) = corners[i], corners[(i + 1) % 4]
        seg = np.stack([np.linspace(x0, x1, 21), np.linspace(y0, y1, 21)], axis=1)
        assert np.min(np.linalg.norm(seg - [pocket.x, pocket.y], axis=1)) >= 0.3 + 0.3 + t.cfg.pocket_margin_m - 1e-9


def test_settle_turns_in_place_first_when_far_off_heading() -> None:
    """Walking sideways or backwards while turning hard, the G1 runs off forwards: turn first, then translate."""
    sc = SettleConfig()
    vx, vy, wz = settle_command(sc, (0.0, 0.0, 0.0), (0.3, 0.0, math.pi))
    assert vx == vy == 0.0 and abs(wz) > 0.0
    vx, vy, wz = settle_command(sc, (0.0, 0.0, math.pi - 0.2), (0.3, 0.0, math.pi))
    assert math.hypot(vx, vy) > 0.0


def test_close_quarters_winner_stops_short_of_a_body_that_did_not_move_aside() -> None:
    """The right of way says a goes; b, parked on a's path, gets a pocket but has not moved: a stops short of it and
    walks on only once b is clear."""
    t = _traffic()
    t.update_pose("a", 20.5, 1.0, math.pi / 2, 0.0)
    t.update_pose("b", 20.5, 2.4, 0.0, 0.0)
    t.request_goal("a", 20.5, 5.0, math.pi / 2, 0.0)
    t.on_path("a", np.stack([np.full(41, 20.5), np.linspace(1.0, 5.0, 41)], axis=1), 0.0)
    for k in range(1, 6):  # a walks north at 0.3 m/s toward b
        t.update_pose("a", 20.5, 1.0 + 0.03 * k, math.pi / 2, 0.1 * k)
        out = t.step(0.1 * k)
    assert t.status()["grants"] == ["a>b:yield"]
    assert t.status()["a"]["mode"] == "hold" and any("b too close ahead" in e for _, e in t.events)
    t.update_pose("b", 21.4, 4.6, 0.0, 1.0)  # b in a pocket
    out = [x for k in range(10, 40) for x in t.step(0.1 * k)]
    assert _of(out, Goal, "a")


def test_a_robot_circling_its_goal_settles_from_where_it_is() -> None:
    t = _traffic(settle=True)
    t.update_pose("a", 10.0, 3.0, 0.0, 0.0)
    t.update_pose("b", 1.0, 1.0, 0.0, 0.0)
    t.request_goal("a", 10.5, 3.0, 0.0, 0.0)
    out = []
    for k in range(1, 260):  # 0.5-0.9 m from the goal, round and round, for 26 s
        a = 0.25 * k
        t.update_pose("a", 10.5 + 0.7 * math.cos(a), 3.0 + 0.7 * math.sin(a), a + math.pi / 2, 0.1 * k)
        out += t.step(0.1 * k)
    assert any("settles from here" in e for _, e in t.events)
    assert _of(out, Stop, "a") and _of(out, Velocity, "a")
