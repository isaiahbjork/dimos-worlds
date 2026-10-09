"""Scripted two-robot traffic scenarios in the warehouse, run against DimOS's planners (coord.inproc).

    python -m dimos_worlds.coord.scenarios head-on --runs 5
    python -m dimos_worlds.coord.scenarios crossing --runs 5
    python -m dimos_worlds.coord.scenarios head-on --no-coord      # the same without coordination

Each run: fresh world, both goals sent once at the start, nothing resent by hand. A run passes when both robots
report arrival (after settling) within the timeout and stand within 0.5 m of their goals, nobody fell, and the bodies
never came closer than `min_gap`.

head-on   aisle A-B (2.5 m between racks, 19 m long): the Go2 at its west end, the G1 at its east end, each
          heading through the aisle past the other to the side aisle beyond the other's end.
crossing  the Go2 walks east along the cross aisle to the dock side while the G1 walks south down the east aisle
          (1.6 m wide) across its path.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
import sys

from dimos_worlds.fleet.robots import RobotSpec, g1, go2

AISLE_AB_Y = 15.35


@dataclass(frozen=True)
class Scenario:
    name: str
    specs: tuple[RobotSpec, ...]
    goals: dict[str, tuple[float, float, float]]
    timeout_s: float
    min_gap_m: float = 0.55  # body centres; the bodies' radii are 0.3 + 0.35


SCENARIOS = {
    "head-on": Scenario(
        "head-on",
        (go2(3.0, AISLE_AB_Y, 0.0), g1(19.8, AISLE_AB_Y, math.pi)),
        {"go2": (21.7, AISLE_AB_Y, 0.0), "g1": (1.1, AISLE_AB_Y, math.pi)},
        timeout_s=240.0),
    "crossing": Scenario(
        "crossing",
        (go2(14.0, 11.4, 0.0), g1(21.7, 17.5, -math.pi / 2)),
        {"go2": (24.5, 9.0, 0.0), "g1": (21.7, 5.0, -math.pi / 2)},
        timeout_s=200.0),
}


def run(sc: Scenario, *, coordinate: bool = True, verbose: bool = False):  # -> inproc.Outcome
    from dimos_worlds.coord.inproc import InProcessFleet

    fleet = InProcessFleet(sc.specs, coordinate=coordinate)
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


def passed(sc: Scenario, out, pos_tol_m: float = 0.5) -> bool:
    """Both arrived (and are where they were sent: DimOS's planner can report arrival without moving), no falls,
    never closer than min_gap."""
    return (out.all_arrived and all(out.errors(rid)[0] <= pos_tol_m for rid in sc.goals)
            and not any(out.falls.values()) and out.min_gap_m >= sc.min_gap_m)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario", choices=sorted(SCENARIOS))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--no-coord", action="store_true", help="planners only, no coordinator")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the coordinator's events")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    sc = SCENARIOS[a.scenario]
    ok = 0
    for i in range(a.runs):
        out = run(sc, coordinate=not a.no_coord, verbose=a.verbose)
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
