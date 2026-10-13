"""MuJoCo preview sheet of the station (no Blender): overview + dock feeds at closed / half / open.

    python -m dimos_worlds.dock.preview --out preview.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .sim import DockSim


def overview(sim: DockSim, w=640, h=400, cam="overview", az=None, el=None, dist=None, look=None):
    r = sim._renderer(w, h)
    if az is None:
        r.update_scene(sim.d, camera=sim.m.camera(cam).id)
    else:
        c = mujoco.MjvCamera()
        c.lookat[:] = look or (0, 0, 0.35)
        c.azimuth, c.elevation, c.distance = az, el, dist
        r.update_scene(sim.d, camera=c)
    return r.render()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sim = DockSim()
    tiles = []
    for deg in (0, 45, 90):
        sim.set_cover_deg(deg)
        tiles.append([
            (f"front {deg}", overview(sim, az=90, el=-12, dist=2.6)),
            (f"3/4 {deg}", overview(sim, az=125, el=-25, dist=2.4)),
            (f"top {deg}", overview(sim, az=90, el=-89, dist=2.6)),
            (f"internal {deg}", np.asarray(Image.fromarray(sim.render("dock_internal", 640, 360)).resize((640, 400)))),
            (f"external {deg}", np.asarray(Image.fromarray(sim.render("dock_external", 640, 360)).resize((640, 400)))),
            (f"drone_wide {deg}", np.asarray(Image.fromarray(sim.render("drone_wide", 640)).resize((640, 400)))),
        ])
    cols = len(tiles[0])
    sheet = Image.new("RGB", (640 * cols, 400 * len(tiles)), "white")
    dr = ImageDraw.Draw(sheet)
    for i, row in enumerate(tiles):
        for j, (label, img) in enumerate(row):
            sheet.paste(Image.fromarray(img), (640 * j, 400 * i))
            dr.text((640 * j + 6, 400 * i + 6), label, fill=(255, 0, 0))
    sheet.save(a.out, quality=85)
    print(a.out)


if __name__ == "__main__":
    main()
