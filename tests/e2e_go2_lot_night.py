"""End-to-end: build go2-lot-night headless, walk the Go2 to named places, trigger an event.

    DIMOS_WORLDS_E2E=1 python tests/e2e_go2_lot_night.py [place ...]

Starts DimOS workers and the MuJoCo child (real time), so it takes minutes. Used by test_lot_e2e.py.
Prints one line per place: place, arrived, seconds, distance left.
"""

from __future__ import annotations

import sys
import time

from dimos.core.global_config import global_config


def main(places: list[str], timeout_s: float = 150.0) -> int:
    global_config.update(viewer="none", zenoh_scout_addr="224.0.0.224:17392")
    from dimos.core.coordination.module_coordinator import ModuleCoordinator

    from dimos_worlds.lot.blueprint import go2_lot_night
    from dimos_worlds.lot.modules import LotEvents, LotPlaces

    coord = ModuleCoordinator.build(go2_lot_night)
    failures = 0
    try:
        nav = coord.get_instance(LotPlaces)
        ev = coord.get_instance(LotEvents)
        print("event:", ev.trigger("gate_open:gate-2"), ev.trigger("vehicle_moved:row-c:14:99"), flush=True)
        time.sleep(5)  # odom + costmap flowing
        for place in places:
            t0 = time.time()
            print(nav.go_to_place(place), flush=True)
            st: dict = {}
            while time.time() - t0 < timeout_s:
                time.sleep(2)
                st = nav.status()
                d = st.get("distance_m")
                if d is not None and d < 1.0:
                    break
            d = st.get("distance_m")
            ok = d is not None and d < 1.0
            failures += 0 if ok else 1
            dist = "?" if d is None else f"{d:.2f}"
            print(f"RESULT {place} arrived={ok} t={time.time() - t0:.0f}s dist={dist} planner_reached={st.get('arrived')}", flush=True)
    finally:
        coord.stop()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["row-d"]))
