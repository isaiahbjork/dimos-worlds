"""Occupancy prior from the scene, stay-outs, and robot-aware masks."""
from __future__ import annotations

import mujoco
import numpy as np

from dimos_worlds.fleet import costmap
from dimos_worlds.fleet.places import load_places
from dimos_worlds.warehouse import cell, scene


def _prior() -> costmap.Prior:
    return costmap.prior_from_model(mujoco.MjModel.from_xml_path(str(scene.SCENE_XML)))


def test_prior_marks_racks_walls_and_free_aisles() -> None:
    p = _prior()
    x0, y0, x1, y1 = cell.LAYOUT["shelves"]["A"]["rect"]
    assert p.value((x0 + x1) / 2, (y0 + y1) / 2) == 100  # rack
    assert p.value(15.0, 11.4) == 0  # cross aisle
    assert p.value(-0.1, 10.0) == 100  # west wall
    assert p.value(cell.ARM_BASE[0], cell.ARM_BASE[1]) == 100  # pedestal
    for home in (cell.LAYOUT["humanoid"]["home"], cell.LAYOUT["quadruped"]["home"]):
        assert p.value(home[0], home[1]) == 0


def test_masks_paint_the_other_robot_and_the_stay_out_not_self() -> None:
    p = _prior()
    places = load_places(scene.PLACES_JSON)
    poses = {"go2": (10.0, 11.4), "g1": (14.0, 11.4)}
    masks = costmap.robot_masks(p, places, poses, {"go2": 0.3, "g1": 0.35})

    def at(grid: np.ndarray, x: float, y: float) -> int:
        c = p.cell(x, y)
        assert c is not None
        return int(grid[c[1], c[0]])

    assert at(masks["go2"], 14.0, 11.4) == 100 and at(masks["go2"], 10.0, 11.4) == 0
    assert at(masks["g1"], 10.0, 11.4) == 100 and at(masks["g1"], 14.0, 11.4) == 0
    x0, y0, x1, y1 = cell.LAYOUT["markings"]["arm_cell_keepout"]
    stay_xy = (x0 + 0.1, (y0 + y1) / 2)  # inside the stay-out, off the pedestal
    assert p.value(*stay_xy) == 0
    assert at(masks["go2"], *stay_xy) == 100
    assert not (masks["go2"] == -1).any()  # unknown sealed
    # near_m: a robot 4 m away is not painted with near_m=3, so it cannot block the other's goal
    far = costmap.robot_masks(p, places, poses, {"go2": 0.3, "g1": 0.35}, near_m=3.0)
    assert at(far["go2"], 14.0, 11.4) == 0 and at(far["g1"], 10.0, 11.4) == 0
    near = costmap.robot_masks(p, places, poses, {"go2": 0.3, "g1": 0.35}, near_m=5.0)
    assert at(near["go2"], 14.0, 11.4) == 100
