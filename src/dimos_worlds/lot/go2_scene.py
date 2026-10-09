"""The lot as a world for DimOS's Go2 MuJoCo sim, and the same lot as the planner's prior map.

* `scene_xml()`: the scene half for `dimos.simulation.mujoco.model.load_model(controller, robot, scene_xml)`.
  DimOS prepends `<include file="unitree_go1.xml"/>` (its Go2 sim runs the Go1 body and ONNX policy) and adds
  its own `person` mocap mesh. lot.xml is already written for this (radians, no <option>/<default>, no joints;
  see gen_lot.py), so the only change is dropping the lot's capsule `person`: DimOS's mesh takes its place, and
  `events.person_at` / DimOS's `/person_pose` topic move that one.
* `occupancy()`: a 2D grid (0 free, 100 occupied) of everything a walking robot could hit between 0.02 m and
  3 m, from the same model, so planner and physics agree. Wheel stops (0.13 m) count as obstacles: the planner
  routes around them even though the Go2 can step over one.
* `occupancy_grid()`: the same grid as a DimOS `OccupancyGrid` message.

Coordinates are the lot frame (metres, x east, y north, origin at the lot centre). The sim's MuJoCo world frame
is this frame, so odom == lot frame.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from dimos_worlds.lot.places import HOME, LOT_XML, Place, load_places

GRID_RES = 0.1
GRID_BOUNDS = (-46.0, -31.0, 46.0, 31.0)  # x0, y0, x1, y1: the lot plus the street apron
OBSTACLE_BAND = (0.02, 3.0)  # z range that blocks a walking robot


def places() -> dict[str, Place]:
    """Named places plus `home` (the robot's start point)."""
    return load_places(with_home=True)


@lru_cache(maxsize=1)
def scene_xml() -> str:
    """lot.xml without its capsule person, ready for DimOS's Go2 model loader."""
    root = ET.parse(LOT_XML).getroot()
    compiler = root.find("compiler")
    if compiler is None or compiler.get("angle") != "radian":
        raise ValueError("lot.xml must be generated with angle=radian (run gen_lot)")
    world = root.find("worldbody")
    assert world is not None
    for body in list(world.findall("body")):
        if body.get("name") == "person":
            world.remove(body)
    return ET.tostring(root, encoding="unicode")


def _lot_model_without_person():  # type: ignore[no-untyped-def]
    import mujoco

    return mujoco.MjModel.from_xml_string(scene_xml())


# ---------------------------------------------------------------- occupancy


def _footprints(model: Any, data: Any, band: tuple[float, float] = OBSTACLE_BAND) -> list[np.ndarray]:
    """Convex xy footprints (N x 2) of every colliding geom that reaches into `band` (z range)."""
    import mujoco

    lo, hi = band
    out: list[np.ndarray] = []
    T = mujoco.mjtGeom
    ring = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)])
    for g in range(model.ngeom):
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        gtype = int(model.geom_type[g])
        if gtype == int(T.mjGEOM_PLANE):
            continue
        pos = data.geom_xpos[g]
        mat = data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        if gtype == int(T.mjGEOM_BOX):
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * size
            pts = pos + corners @ mat.T
        elif gtype in (int(T.mjGEOM_CYLINDER), int(T.mjGEOM_CAPSULE)):
            r, half = size[0], size[1]
            axis, u, v = mat[:, 2], mat[:, 0], mat[:, 1]  # the end discs lie in the geom's local x-y plane
            disc = np.array([r * (c * u + s_ * v) for c, s_ in ring])
            ends = [pos - axis * half, pos + axis * half]
            pts = np.concatenate([e + disc for e in ends])
            if gtype == int(T.mjGEOM_CAPSULE):  # hemispherical caps
                pts = np.concatenate([pts, *[(e + axis * sgn * r)[None, :] for e, sgn in zip(ends, (-1, 1))]])
        elif gtype in (int(T.mjGEOM_SPHERE), int(T.mjGEOM_ELLIPSOID)):
            r = float(np.max(size[:3])) if gtype == int(T.mjGEOM_ELLIPSOID) else float(size[0])
            pts = np.column_stack([pos[0] + r * ring[:, 0], pos[1] + r * ring[:, 1], np.full(len(ring), pos[2])])
            pts = np.concatenate([pts + [0, 0, r], pts - [0, 0, r]])
        else:
            rb = float(model.geom_rbound[g])
            pts = np.array(
                [[pos[0] + dx, pos[1] + dy, pos[2] + dz] for dx in (-rb, rb) for dy in (-rb, rb) for dz in (-rb, rb)]
            )
        if pts[:, 2].max() < lo or pts[:, 2].min() > hi:
            continue
        out.append(pts[:, :2])
    return out


def _hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set(map(tuple, np.round(points, 4))))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):  # type: ignore[no-untyped-def]
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def occupancy_from(model: Any, data: Any, band: tuple[float, float] = OBSTACLE_BAND) -> dict:
    """Occupancy of a live model/data (e.g. after events moved a gate or a vehicle)."""
    x0, y0, x1, y1 = GRID_BOUNDS
    w = int(round((x1 - x0) / GRID_RES))
    h = int(round((y1 - y0) / GRID_RES))
    grid = np.zeros((h, w), dtype=np.int8)
    xs = x0 + (np.arange(w) + 0.5) * GRID_RES
    ys = y0 + (np.arange(h) + 0.5) * GRID_RES
    for fp in _footprints(model, data, band):
        # A thin geom (fence fabric is 5 cm) must still block a cell: dilate by half a cell diagonal first.
        d = GRID_RES * 0.71
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
    return {"grid": grid, "origin_x": x0, "origin_y": y0, "resolution": GRID_RES}


@lru_cache(maxsize=2)
def occupancy(band: tuple[float, float] = OBSTACLE_BAND) -> dict:
    """Baseline occupancy: {"grid": int8 (H, W), "origin_x", "origin_y", "resolution"}; row 0 is y = origin_y."""
    import mujoco

    model = _lot_model_without_person()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return occupancy_from(model, data, band)


def occupancy_grid(prior: dict | None = None, frame_id: str = "world"):  # type: ignore[no-untyped-def]
    """The prior as a DimOS nav_msgs OccupancyGrid."""
    from dimos.msgs.geometry_msgs.Pose import Pose
    from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid

    prior = prior or occupancy()
    origin = Pose(position=[prior["origin_x"], prior["origin_y"], 0.0], orientation=[0.0, 0.0, 0.0, 1.0])
    return OccupancyGrid(grid=prior["grid"], resolution=prior["resolution"], origin=origin, frame_id=frame_id)


def cell(prior: dict, x: float, y: float) -> int:
    """Grid value at (x, y): 0 free, 100 occupied, -1 outside the grid."""
    ix = int((x - prior["origin_x"]) / prior["resolution"])
    iy = int((y - prior["origin_y"]) / prior["resolution"])
    g = prior["grid"]
    if not (0 <= ix < g.shape[1] and 0 <= iy < g.shape[0]):
        return -1
    return int(g[iy, ix])


def clearance(prior: dict, x: float, y: float) -> float:
    """Distance in metres from (x, y) to the nearest occupied cell (capped at 5 m)."""
    g = prior["grid"]
    res = prior["resolution"]
    r = int(5.0 / res)
    ix = int((x - prior["origin_x"]) / res)
    iy = int((y - prior["origin_y"]) / res)
    win = g[max(0, iy - r) : iy + r + 1, max(0, ix - r) : ix + r + 1]
    occ = np.argwhere(win == 100)
    if len(occ) == 0:
        return 5.0
    oy, ox = max(0, iy - r), max(0, ix - r)
    d = np.hypot(occ[:, 1] + ox - ix, occ[:, 0] + oy - iy) * res
    return float(d.min())


def reachable(prior: dict, start: tuple[float, float], robot_radius: float = 0.3) -> np.ndarray:
    """Boolean mask of cells reachable from `start` by 4-connected moves through cells with at least
    `robot_radius` clearance (the occupied set inflated by the radius)."""
    from collections import deque

    g = prior["grid"] == 100
    res = prior["resolution"]
    h, w = g.shape
    r = max(1, int(math.ceil(robot_radius / res)))
    inflated = g.copy()
    ys, xs = np.nonzero(g)
    offsets = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r]
    for dy, dx in offsets:
        yy = np.clip(ys + dy, 0, h - 1)
        xx = np.clip(xs + dx, 0, w - 1)
        inflated[yy, xx] = True
    seen = np.zeros_like(g)
    sx = int((start[0] - prior["origin_x"]) / res)
    sy = int((start[1] - prior["origin_y"]) / res)
    if inflated[sy, sx]:
        return seen
    q = deque([(sy, sx)])
    seen[sy, sx] = True
    while q:
        y, x = q.popleft()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and not inflated[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen


if __name__ == "__main__":
    prior = occupancy()
    print("grid", prior["grid"].shape, "occupied %.1f%%" % (100 * (prior["grid"] == 100).mean()))
    for p in places().values():
        print(f"{p.id:18s} ({p.x:6.1f},{p.y:6.1f}) cell={cell(prior, p.x, p.y):4d} clearance={clearance(prior, p.x, p.y):.2f} m")
    print("home", HOME.xy)
