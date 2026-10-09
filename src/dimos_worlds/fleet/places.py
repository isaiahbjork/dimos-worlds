"""Named places: points, areas and stay-outs, in world x/y.

File format (the scene package's places.json):

    {"places": [{"id": "...", "name": "Shelf A", "kind": "point", "x": 2.1, "y": 16.0},
                {"id": "...", "name": "Dock", "kind": "area", "points": [{"x":..,"y":..}, ...], "goal": {"x":..,"y":..}},
                {"id": "...", "name": "Arm cell", "kind": "stay", "points": [...]}]}

A stay-out is painted occupied into every robot's costmap (paint_stays), so a planner never routes through it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KINDS = ("point", "area", "stay")


def load_places(path: Path | str) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    places = data.get("places") if isinstance(data, dict) else None
    if not isinstance(places, list):
        return []
    return [p for p in places if isinstance(p, dict) and p.get("kind") in KINDS]


def find_place(places: list[dict], name_or_id: str) -> dict | None:
    key = name_or_id.strip().lower()
    for p in places:
        if str(p.get("id", "")).lower() == key or str(p.get("name", "")).lower() == key:
            return p
    return None


def goal_xy(place: dict) -> tuple[float, float] | None:
    if place.get("kind") == "point":
        return float(place["x"]), float(place["y"])
    if place.get("kind") == "area":
        goal = place.get("goal") or {}
        return float(goal["x"]), float(goal["y"])
    return None


def points_in_poly(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd test. px and py are the same shape; poly is (n, 2) in x/y."""
    x = poly[:, 0]
    y = poly[:, 1]
    inside = np.zeros(px.shape, dtype=bool)
    for i in range(len(poly)):
        x1, y1 = x[i], y[i]
        x2, y2 = x[(i + 1) % len(poly)], y[(i + 1) % len(poly)]
        if y1 == y2:
            continue
        cross = ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1)
        inside ^= cross
    return inside


def paint_stays(grid: np.ndarray, origin_x: float, origin_y: float, resolution: float,
                places: list[dict]) -> np.ndarray:
    """Copy of grid (row = y, col = x) with stay-out cells set to 100. The test point is the cell centre."""
    stays = [np.array([[p["x"], p["y"]] for p in place.get("points") or []], dtype=float)
             for place in places if place.get("kind") == "stay" and len(place.get("points") or []) >= 3]
    if not stays or grid.size == 0:
        return grid
    out = np.array(grid, dtype=np.int8, copy=True)
    height, width = out.shape
    xs = origin_x + (np.arange(width) + 0.5) * resolution
    ys = origin_y + (np.arange(height) + 0.5) * resolution
    xx, yy = np.meshgrid(xs, ys)
    mask = np.zeros(xx.shape, dtype=bool)
    for poly in stays:
        mask |= points_in_poly(xx.ravel(), yy.ravel(), poly).reshape(xx.shape)
    out[mask] = 100
    return out
