"""Occupancy priors from a MuJoCo scene, and robot-aware masks on top of them.

A planner fed only its own lidar costmap cannot plan to the far side of a building it has not seen yet, and it cannot
see a second robot that its sensors have not hit. So each robot gets its own grid:

    prior (the scene's static geometry at body height)  +  stay-out polygons (places.json)
      +  every OTHER robot's footprint, from live odometry

The robot itself is never painted, so its own start cell stays plannable. Unknown cells (outside the scene's floor)
are sealed as occupied so a plan cannot leave through a gap into nothing.

Pure numpy + mujoco (no DimOS imports), so tests can check the grids without a running blueprint.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from dimos_worlds.fleet.places import paint_stays

BODY_Z = (0.05, 1.4)  # geometry in this height band blocks a walking robot


@dataclass
class Prior:
    grid: np.ndarray  # int8, row = y, col = x; 0 free, 100 occupied, -1 unknown
    origin_x: float
    origin_y: float
    resolution: float

    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        ix = int(math.floor((x - self.origin_x) / self.resolution))
        iy = int(math.floor((y - self.origin_y) / self.resolution))
        h, w = self.grid.shape
        return (ix, iy) if 0 <= ix < w and 0 <= iy < h else None

    def value(self, x: float, y: float) -> int:
        c = self.cell(x, y)
        return -1 if c is None else int(self.grid[c[1], c[0]])


def _geom_aabb(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> tuple[np.ndarray, np.ndarray] | None:
    t = model.geom_type[g]
    size = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        half = size[:3]
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER or t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half = np.array([size[0], size[0], size[1] + (size[0] if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.0)])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = np.array([size[0]] * 3)
    else:
        return None  # planes (the floor), meshes, height fields: not obstacles here
    rot = data.geom_xmat[g].reshape(3, 3)
    ext = np.abs(rot) @ half
    c = data.geom_xpos[g]
    return c - ext, c + ext


def prior_from_model(model: mujoco.MjModel, *, resolution: float = 0.1, margin_m: float = 1.0,
                     body_z: tuple[float, float] = BODY_Z) -> Prior:
    """Rasterise a static scene: floor plane extent is free, geoms crossing the body band are occupied."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    floor = None
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    for g in range(model.ngeom):
        if model.body_mocapid[model.geom_bodyid[g]] >= 0:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            sx, sy = model.geom_size[g][:2]
            cx, cy = data.geom_xpos[g][:2]
            if sx > 0 and sy > 0:
                floor = (cx - sx, cy - sy, cx + sx, cy + sy)
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue  # visual only
        box = _geom_aabb(model, data, g)
        if box is None:
            continue
        lo, hi = box
        if hi[2] < body_z[0] or lo[2] > body_z[1]:
            continue
        boxes.append(box)
    if floor is None:
        raise ValueError("scene has no floor plane")
    x0, y0, x1, y1 = floor
    ox = math.floor((x0 - margin_m) / resolution) * resolution
    oy = math.floor((y0 - margin_m) / resolution) * resolution
    w = int(math.ceil((x1 + margin_m - ox) / resolution))
    h = int(math.ceil((y1 + margin_m - oy) / resolution))
    grid = np.full((h, w), -1, dtype=np.int8)
    xs = ox + (np.arange(w) + 0.5) * resolution
    ys = oy + (np.arange(h) + 0.5) * resolution
    in_x = (xs >= x0) & (xs <= x1)
    in_y = (ys >= y0) & (ys <= y1)
    grid[np.ix_(in_y, in_x)] = 0
    for lo, hi in boxes:
        i0 = max(0, int(math.floor((lo[0] - ox) / resolution)))
        i1 = min(w - 1, int(math.floor((hi[0] - ox) / resolution)))
        j0 = max(0, int(math.floor((lo[1] - oy) / resolution)))
        j1 = min(h - 1, int(math.floor((hi[1] - oy) / resolution)))
        if i0 <= i1 and j0 <= j1:
            grid[j0:j1 + 1, i0:i1 + 1] = 100
    return Prior(grid=grid, origin_x=float(ox), origin_y=float(oy), resolution=float(resolution))


def seal_unknown(grid: np.ndarray) -> np.ndarray:
    """Unknown (-1) cells become occupied unless they touch a free cell (door thresholds stay passable)."""
    out = np.array(grid, copy=True)
    unknown = out == -1
    if not bool(unknown.any()):
        return out
    padded = np.pad(out == 0, 1, mode="constant", constant_values=False)
    touches_free = (padded[:-2, :-2] | padded[:-2, 1:-1] | padded[:-2, 2:] | padded[1:-1, :-2] | padded[1:-1, 2:]
                    | padded[2:, :-2] | padded[2:, 1:-1] | padded[2:, 2:])
    out[unknown & ~touches_free] = 100
    out[unknown & touches_free] = 0
    return out


def paint_disc(grid: np.ndarray, origin_x: float, origin_y: float, resolution: float,
               x: float, y: float, radius: float) -> None:
    """Mark a disc occupied, in place."""
    h, w = grid.shape
    cx = (x - origin_x) / resolution
    cy = (y - origin_y) / resolution
    cells = int(radius / resolution) + 1
    for iy in range(max(0, int(cy) - cells), min(h, int(cy) + cells + 1)):
        for ix in range(max(0, int(cx) - cells), min(w, int(cx) + cells + 1)):
            dx = (ix + 0.5 - cx) * resolution
            dy = (iy + 0.5 - cy) * resolution
            if dx * dx + dy * dy <= radius * radius:
                grid[iy, ix] = 100


def robot_masks(prior: Prior, places: list[dict], poses: dict[str, tuple[float, float]],
                radii: dict[str, float], *, pad_m: float = 0.25,
                near_m: float | None = None) -> dict[str, np.ndarray]:
    """One grid per robot id in `radii`: prior + stay-outs + sealed unknown + every other robot's footprint
    (its radius + pad_m) at its pose. A robot without a pose yet is simply not painted.

    near_m: paint another robot only when it is within this distance of the robot the grid is for. A robot far
    away is no obstacle yet, and painting it can block the goal itself (DimOS's planner then gives up)."""
    base = seal_unknown(paint_stays(prior.grid, prior.origin_x, prior.origin_y, prior.resolution, places))
    out: dict[str, np.ndarray] = {}
    for rid in radii:
        grid = np.array(base, copy=True)
        for other, at in poses.items():
            if other == rid or other not in radii:
                continue
            me = poses.get(rid)
            if near_m is not None and me is not None and math.hypot(at[0] - me[0], at[1] - me[1]) > near_m:
                continue
            paint_disc(grid, prior.origin_x, prior.origin_y, prior.resolution, at[0], at[1], radii[other] + pad_m)
        out[rid] = grid
    return out
