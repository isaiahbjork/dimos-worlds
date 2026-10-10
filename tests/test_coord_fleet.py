"""Coordination against DimOS's own planners in the shared warehouse (coord.inproc).

Stepped on sim time: the same result on any machine, however loaded. DIMOS_WORLDS_REALTIME=1 also runs the same
checks with the live timing (planner and coordinator threads on the wall clock: minutes, and load-dependent).
"""
from __future__ import annotations

import math
import os

import pytest

from dimos_worlds.coord.inproc import InProcessFleet
from dimos_worlds.coord.scenarios import SCENARIOS, passed, run
from dimos_worlds.fleet.robots import g1, go2

MODES = [False, pytest.param(True, marks=pytest.mark.skipif(
    os.environ.get("DIMOS_WORLDS_REALTIME") != "1", reason="real-time variant: set DIMOS_WORLDS_REALTIME=1"))]
IDS = ["stepped", "realtime"]


@pytest.mark.parametrize("realtime", MODES, ids=IDS)
def test_head_on_in_an_aisle_both_arrive_without_a_resend(realtime: bool) -> None:
    sc = SCENARIOS["head-on"]
    out = run(sc, realtime=realtime)
    assert passed(sc, out), (out.arrived, out.final, out.min_gap_m, out.falls, out.events)
    assert any("yields to" in e for e in out.events)  # resolved by a yield, not by luck


def _g1_settle(realtime: bool):
    fleet = InProcessFleet((go2(3.0, 18.6, 0.0), g1(6.0, 11.0, 0.0)), realtime=realtime)
    try:
        fleet.start()
        fleet.goal("g1", 9.0, 11.6, math.pi / 2)  # 3 m east, then a quarter turn
        out = fleet.run_until_arrived(120.0)
        state = fleet.world.state_hash()
    finally:
        fleet.close()
    return out, state


@pytest.mark.parametrize("realtime", MODES, ids=IDS)
def test_g1_settles_to_goal_heading_and_position(realtime: bool) -> None:
    """The planner alone leaves the G1 up to 0.5 m and tens of degrees off; settled, within 0.2 m and 10 degrees."""
    out, _ = _g1_settle(realtime)
    assert "g1" in out.arrived
    dp, dyaw = out.errors("g1")
    assert dp <= 0.2 and math.degrees(dyaw) <= 10.0, (dp, math.degrees(dyaw), out.events)
    assert out.falls["g1"] == 0, out.events


def test_stepped_runs_repeat_exactly() -> None:
    """Same scenario twice: same events at the same sim times, same final state, bit for bit."""
    a, ha = _g1_settle(False)
    b, hb = _g1_settle(False)
    assert a.events == b.events
    assert a.arrived == b.arrived and a.final == b.final
    assert ha == hb


def test_live_blueprint_gives_each_planner_its_robot_speed() -> None:
    from dimos_worlds.fleet.blueprint import warehouse_fleet_blueprint
    from dimos_worlds.fleet.planner import FleetPlanner
    from dimos_worlds.fleet.robots import warehouse_fleet

    planners = {a.kwargs["frame_id_prefix"]: a.kwargs for a in warehouse_fleet_blueprint.blueprints
                if a.module is FleetPlanner}
    assert {rid: kw["nav_speed"] for rid, kw in planners.items()} == {s.id: s.nav_speed for s in warehouse_fleet()}


@pytest.mark.parametrize(("run_speed", "nav_speed", "expect"), [(1.0, 0.4, 0.22), (1.0, 1.0, 0.55), (0.5, 0.4, 0.11)])
def test_fleet_planner_scales_the_run_config_built_in_the_worker(run_speed: float, nav_speed: float,
                                                                   expect: float) -> None:
    # The run's config (CLI overrides applied) reaches the module as `g`; only nerf_speed is changed, per robot.
    from dimos.core.global_config import GlobalConfig

    from dimos_worlds.fleet.planner import FleetPlanner

    g = GlobalConfig(simulation="mujoco", nerf_speed=run_speed, robot_model="cli-override")
    p = FleetPlanner(robot_width=0.7, robot_rotation_diameter=0.8, nav_speed=nav_speed, g=g)
    try:
        gc = p._planner._global_config
        assert (gc.simulation, gc.robot_model, gc.robot_width, gc.robot_rotation_diameter) == (
            "mujoco", "cli-override", 0.7, 0.8)
        assert p._planner._local_planner._controller._speed == pytest.approx(expect)
    finally:
        p.stop()
