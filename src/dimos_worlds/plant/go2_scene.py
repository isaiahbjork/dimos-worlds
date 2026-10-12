"""The Demo plant as a world for DimOS's Go2 MuJoCo sim, and the same plant as the planner's prior map.

* `scene_xml()`: scene/plant.xml, written for DimOS's Go2 loader as is (gen_mjcf.py: radians, no <option>/<default>,
  friction on every geom, no joints).
* `occupancy()`: 0.1 m grid of everything a dog could hit between 0.02 and 1.0 m (walls, plinths, pumps, pipe
  supports, low pipes, tank bund, stairs, rack columns); lintels, overhead pipes and roofs do not count. The stairs
  (pipe-rack stair tower, bund stile) are stay-out zones and solid in the grid, so the planner never routes over them.
* `places()`: the dock first, then every station and stay-out zone, poses in the plant frame.

Frame: metres, origin at the SW corner of the slab, x east, y north. The sim's MuJoCo world frame is this frame.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from dimos_worlds.lot.go2_scene import _footprints, _hull, cell, clearance, occupancy_grid

HERE = Path(__file__).resolve().parent
PLANT_XML = HERE / "scene" / "plant.xml"
LAYOUT_JSON = HERE / "layout.json"

GRID_RES = 0.1
GRID_BOUNDS = (-3.0, -3.0, 59.0, 39.0)  # the slab plus a margin
OBSTACLE_BAND = (0.02, 1.0)  # a standing Go2 with a payload is ~0.7 m tall: lintels and overhead pipes do not block

__all__ = ["HOME", "cell", "clearance", "layout", "occupancy", "occupancy_grid", "places", "scene_xml"]


@lru_cache(maxsize=1)
def layout() -> dict:
    return json.loads(LAYOUT_JSON.read_text())


def _home() -> dict:
    lay = layout()
    dock = next(p for p in lay["places"] if p["id"] == lay["robot"]["dock"])
    return {"id": dock["id"], "name": dock["name"], "kind": "dock",
            "pose": {"x": dock["pose"]["x"], "y": dock["pose"]["y"]}}


HOME = _home()


def places() -> list[dict]:
    """Dock first, then stations and stay-out zones (centre as pose for a stay-out)."""
    out = [HOME]
    for p in layout()["places"]:
        if p["id"] == HOME["id"]:
            continue
        pose = p.get("pose") or {"x": p["center"][0], "y": p["center"][1]}
        out.append({"id": p["id"], "name": p["name"], "kind": p["kind"], "pose": {"x": pose["x"], "y": pose["y"]}})
    return out


@lru_cache(maxsize=1)
def scene_xml() -> str:
    return PLANT_XML.read_text()


@lru_cache(maxsize=2)
def occupancy(band: tuple[float, float] = OBSTACLE_BAND) -> dict:
    """{"grid": int8 (H, W), "origin_x", "origin_y", "resolution"}; row 0 is y = origin_y."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(scene_xml())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    x0, y0, x1, y1 = GRID_BOUNDS
    w = int(round((x1 - x0) / GRID_RES))
    h = int(round((y1 - y0) / GRID_RES))
    grid = np.zeros((h, w), dtype=np.int8)
    xs = x0 + (np.arange(w) + 0.5) * GRID_RES
    ys = y0 + (np.arange(h) + 0.5) * GRID_RES
    for fp in _footprints(model, data, band):
        d = GRID_RES * 0.71  # a thin geom must still block a cell: dilate by half a cell diagonal
        grow = _hull(np.concatenate([fp + [dx, dy] for dx in (-d, d) for dy in (-d, d)]))
        if len(grow) < 3:
            continue
        bx0, by0 = grow.min(axis=0)
        bx1, by1 = grow.max(axis=0)
        ix0 = max(0, int((bx0 - x0) / GRID_RES))
        ix1 = min(w, int((bx1 - x0) / GRID_RES) + 1)
        iy0 = max(0, int((by0 - y0) / GRID_RES))
        iy1 = min(h, int((by1 - y0) / GRID_RES) + 1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        gx, gy = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1])
        inside = np.ones(gx.shape, dtype=bool)
        n = len(grow)
        for i in range(n):
            ax, ay = grow[i]
            bx, by = grow[(i + 1) % n]
            inside &= (bx - ax) * (gy - ay) - (by - ay) * (gx - ax) >= 0
        grid[iy0:iy1, ix0:ix1][inside] = 100
    for p in layout()["places"]:  # stay-out zones (stairs) are walls for the planner, whatever their height
        if p["kind"] != "stay_out" or "poly" not in p:
            continue
        pxs = [q[0] for q in p["poly"]]
        pys = [q[1] for q in p["poly"]]
        ix0, ix1 = int((min(pxs) - x0) / GRID_RES), int((max(pxs) - x0) / GRID_RES) + 1
        iy0, iy1 = int((min(pys) - y0) / GRID_RES), int((max(pys) - y0) / GRID_RES) + 1
        grid[max(0, iy0):min(h, iy1), max(0, ix0):min(w, ix1)] = 100
    return {"grid": grid, "origin_x": x0, "origin_y": y0, "resolution": GRID_RES}


if __name__ == "__main__":
    prior = occupancy()
    print("grid", prior["grid"].shape, "occupied %.1f%%" % (100 * (prior["grid"] == 100).mean()))
    for p in places():
        x, y = p["pose"]["x"], p["pose"]["y"]
        print(f"{p['id']:18s} ({x:6.2f},{y:6.2f}) cell={cell(prior, x, y):4d} clearance={clearance(prior, x, y):.2f} m")
