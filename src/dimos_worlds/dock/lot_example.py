"""Optional example: the Dock 3-class station on the night lot's office roof, with its M4D-class quadcopter.

    python -m dimos_worlds.dock.lot_example            # attach, compile, open, fly a round over row D, land, close
    python -m dimos_worlds.dock.lot_example --out lot_dock.jpg   # plus a still of the open dock from the roof

The lot itself is unchanged (`lot/scene/lot.xml`); this builds a separate MuJoCo model from it with
`fixture.attach_dock` and drives it with `DockSim`. It is not part of any blueprint: no robot commands go through
it, so it needs no safety watchdog. The dock stands on the flat roof of the office (`lot.layout.OFFICE`, roof at
height + parapet = 4.45 m), its long axis east-west, clear of the parapet edge.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

from dimos_worlds.dock.fixture import attach_dock
from dimos_worlds.dock.sim import DockSim

PREFIX = "dock_"
YAW_DEG = 0.0


def roof_pose() -> tuple[float, float, float]:
    """Dock origin on the office roof: 2 m west of the roof centre, on the roof plane."""
    from dimos_worlds.lot.layout import OFFICE

    (cx, cy), h = OFFICE["center"], OFFICE["height"] + OFFICE["parapet"]
    return (cx - 2.0, cy, h)


def lot_with_dock(pos: tuple[float, float, float] | None = None, yaw_deg: float = YAW_DEG) -> mujoco.MjModel:
    """The lot (as in lot.xml, person included) plus the station and quadcopter, compiled."""
    from dimos_worlds.lot.places import LOT_XML

    spec = mujoco.MjSpec.from_file(str(LOT_XML))
    attach_dock(spec, pos=pos or roof_pose(), yaw_deg=yaw_deg, prefix=PREFIX)
    return spec.compile()


def round_trip(sim: DockSim, over: tuple[float, float] = (-2.0, -17.0), alt: float = 12.0) -> dict:
    """Open, take off, fly to `over` at `alt` (lot frame), come back, land, close. Returns timings and offsets."""
    sim.open()
    t_open = sim.run_until_idle()
    sim.takeoff(3.0)
    sim.fly_to(over[0], over[1], alt)
    reached = [float(v) for v in sim.d.xpos[sim.drone]]
    sim.fly_to(float(sim.pad_pos[0]), float(sim.pad_pos[1]), float(sim.pad_pos[2]) + 3.0)
    sim.land()
    sim.close()
    t_close = sim.run_until_idle()
    st = sim.state()
    return {"open_s": round(t_open, 2), "close_s": round(t_close, 2), "reached": [round(v, 2) for v in reached],
            "state": st.state, "drone": st.drone, "max_penetration_mm": round(sim.max_pen * 1000, 2),
            "off_pad_contacts": sorted(sim.bad_contacts)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, help="write a still of the open dock (JPEG)")
    a = ap.parse_args()
    model = lot_with_dock()
    sim = DockSim.from_model(model, prefix=PREFIX)
    print(round_trip(sim))
    if a.out:
        from PIL import Image

        sim.set_cover_deg(90.0)
        Image.fromarray(sim.render("dock_external", 1280, 720, lens="stream")).save(a.out, quality=85)
        print(a.out)


if __name__ == "__main__":
    main()
