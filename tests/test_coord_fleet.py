"""Coordination against DimOS's own planners in the shared warehouse (coord.inproc, real time: minutes)."""
from __future__ import annotations

import math

from dimos_worlds.coord.inproc import InProcessFleet
from dimos_worlds.coord.scenarios import SCENARIOS, passed, run
from dimos_worlds.fleet.robots import g1, go2


def test_head_on_in_an_aisle_both_arrive_without_a_resend() -> None:
    sc = SCENARIOS["head-on"]
    out = run(sc)
    assert passed(sc, out), (out.arrived, out.final, out.min_gap_m, out.events)
    assert any("yields to" in e for e in out.events)  # resolved by a yield, not by luck


def test_g1_settles_to_goal_heading_and_position() -> None:
    """The planner alone leaves the G1 up to 0.5 m and tens of degrees off; settled, within 0.2 m and 10 degrees."""
    fleet = InProcessFleet((go2(3.0, 18.6, 0.0), g1(6.0, 11.0, 0.0)))
    try:
        fleet.start()
        fleet.goal("g1", 9.0, 11.6, math.pi / 2)  # 3 m east, then a quarter turn
        out = fleet.run_until_arrived(120.0)
    finally:
        fleet.close()
    assert "g1" in out.arrived
    dp, dyaw = out.errors("g1")
    assert dp <= 0.2 and math.degrees(dyaw) <= 10.0, (dp, math.degrees(dyaw), out.events)
    assert out.falls["g1"] == 0
