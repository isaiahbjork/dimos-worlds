"""Scripted traffic scenarios in the warehouse, two to four robots, run against DimOS's planners (coord.inproc).

    python -m dimos_worlds.coord.scenarios head-on --runs 5
    python -m dimos_worlds.coord.scenarios crossing --runs 5
    python -m dimos_worlds.coord.scenarios head-on --no-coord      # the same without coordination
    python -m dimos_worlds.coord.scenarios head-on --realtime      # planner threads on the wall clock, as live

Runs are stepped on sim time by default (coord.inproc): the same scenario gives the same result on any machine, and
the timeout is in sim seconds. --realtime runs the live timing instead; its outcome depends on machine load.

Each run: fresh world, both goals sent once at the start, nothing resent by hand. A run passes when both robots
report arrival (after settling) within the timeout and stand within 0.5 m and 15 degrees of their goal poses, nobody
fell, and the body centres never came closer than `min_gap` (0.7 m; the footprint radii add up to 0.65 m).

head-on   aisle A-B (2.5 m between racks, 19 m long): the Go2 at its west end, the G1 at its east end, each
          heading through the aisle past the other to the side aisle beyond the other's end.
crossing  the Go2 walks east along the cross aisle to the dock side while the G1 walks south down the east aisle
          (1.6 m wide) across its path; both are about 5 m from where the paths cross.

Three and four robots (a second and third Go2 body: go2b, go2c; priority in list order):

three-way the junction of the cross aisle and the east aisle: a Go2 from the west bound for the dock, the G1 from
          the north bound south, a second Go2 from the dock bound west down the cross aisle, head-on with the first.
convoy    aisle C-D (2.5 m, 19 m long), all three heading east into the dock: a Go2 in front, the G1 behind it and a
          second Go2 behind the G1, about 2 m apart. Both followers walk behind slower or stopped robots.
ring      four robots on the corners of a 2.4 m square on the open dock floor, each bound for the next one's corner
          (counter-clockwise). Every robot's goal is held by the next one, so pairwise right of way makes a cycle of
          holds; the coordinator breaks it (coord.traffic, hold cycle).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
import sys

from dimos_worlds.fleet.robots import RobotSpec, g1, go2

AISLE_AB_Y = 15.35
AISLE_CD_Y = 7.35
RING = ((24.8, 6.8), (27.2, 6.8), (27.2, 9.2), (24.8, 9.2))  # counter-clockwise


@dataclass(frozen=True)
class Scenario:
    name: str
    specs: tuple[RobotSpec, ...]
    goals: dict[str, tuple[float, float, float]]
    timeout_s: float
    min_gap_m: float = 0.7  # body centres; the footprint radii add up to 0.65


SCENARIOS = {
    "head-on": Scenario(
        "head-on",
        (go2(3.0, AISLE_AB_Y, 0.0), g1(19.8, AISLE_AB_Y, math.pi)),
        {"go2": (21.7, AISLE_AB_Y, 0.0), "g1": (1.1, AISLE_AB_Y, math.pi)},
        timeout_s=240.0),
    "crossing": Scenario(
        "crossing",
        (go2(17.0, 11.4, 0.0), g1(21.7, 16.0, -math.pi / 2)),
        {"go2": (24.5, 9.0, 0.0), "g1": (21.7, 5.0, -math.pi / 2)},
        timeout_s=200.0),
    "three-way": Scenario(
        "three-way",
        (go2(16.0, 11.4, 0.0), g1(21.7, 16.0, -math.pi / 2), go2(27.5, 10.5, math.pi, robot_id="go2b")),
        {"go2": (25.0, 8.0, 0.0), "g1": (21.7, 5.0, -math.pi / 2), "go2b": (14.0, 11.4, math.pi)},
        timeout_s=240.0),
    "convoy": Scenario(
        "convoy",
        (go2(6.6, AISLE_CD_Y, 0.0), g1(4.5, AISLE_CD_Y, 0.0), go2(2.4, AISLE_CD_Y, 0.0, robot_id="go2b")),
        {"go2": (27.0, 6.5, 0.0), "g1": (24.0, 8.0, 0.0), "go2b": (27.0, 9.5, 0.0)},
        timeout_s=300.0),
    "ring": Scenario(
        "ring",
        tuple(f(*RING[i], math.pi / 2 * i, **kw) for i, (f, kw) in enumerate(
            [(go2, {}), (g1, {}), (go2, {"robot_id": "go2b"}), (go2, {"robot_id": "go2c"})])),
        {rid: (*RING[(i + 1) % 4], math.pi / 2 * i) for i, rid in enumerate(("go2", "g1", "go2b", "go2c"))},
        timeout_s=240.0),
}


def run(sc: Scenario, *, coordinate: bool = True, verbose: bool = False, realtime: bool = False):  # -> Outcome
    from dimos_worlds.coord.inproc import InProcessFleet

    fleet = InProcessFleet(sc.specs, coordinate=coordinate, realtime=realtime)
    try:
        fleet.start()
        for rid, (x, y, yaw) in sc.goals.items():
            fleet.goal(rid, x, y, yaw)
        out = fleet.run_until_arrived(sc.timeout_s)
    finally:
        fleet.close()
    if verbose:
        for e in out.events:
            print("   ", e)
    return out


def passed(sc: Scenario, out, pos_tol_m: float = 0.5, yaw_tol_deg: float = 15.0) -> bool:
    """All arrived and stand at the goal pose (DimOS's planner can report arrival without moving), no falls,
    never closer than min_gap."""
    return (out.all_arrived
            and all(out.errors(rid)[0] <= pos_tol_m and math.degrees(out.errors(rid)[1]) <= yaw_tol_deg
                    for rid in sc.goals)
            and not any(out.falls.values()) and out.min_gap_m >= sc.min_gap_m)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=sorted(SCENARIOS))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-coord", action="store_true", help="planners only, no coordinator")
    ap.add_argument("--realtime", action="store_true", help="wall-clock planner threads (live timing)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the coordinator's events")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    sc = SCENARIOS[a.scenario]
    ok = 0
    for i in range(a.runs):
        out = run(sc, coordinate=not a.no_coord, verbose=a.verbose, realtime=a.realtime)
        good = passed(sc, out)
        ok += good
        parts = []
        for rid in sc.goals:
            if rid in out.arrived:
                dp, dyaw = out.errors(rid)
                parts.append(f"{rid} arrived {out.arrived[rid]:.0f} s ({dp:.2f} m, {math.degrees(dyaw):.0f} deg)")
            else:
                x, y, _ = out.final[rid]
                parts.append(f"{rid} NOT arrived (at {x:.1f}, {y:.1f})")
        print(f"{sc.name} run {i + 1}: {'PASS' if good else 'FAIL'}  " + "; ".join(parts)
              + f"; min gap {out.min_gap_m:.2f} m; falls {out.falls}", flush=True)
    print(f"{sc.name}: {ok}/{a.runs} passed ({'coordinated' if not a.no_coord else 'no coordination'})")
    return 0 if ok == a.runs else 1


if __name__ == "__main__":
    sys.exit(main())
