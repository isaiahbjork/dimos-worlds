"""Warehouse scene package: format, composition with DimOS's robots, floor plan routes."""
from __future__ import annotations

import json
import math

import mujoco
import pytest

from dimos_worlds.warehouse import cell, gen_layout, scene


def test_layout_json_is_generated_and_has_no_agv() -> None:
    on_disk = json.loads(cell.LAYOUT_PATH.read_text())
    assert on_disk == json.loads(json.dumps(gen_layout.build()))
    assert "agv" not in on_disk
    assert "AGV" not in cell.LAYOUT_PATH.read_text()


def test_cooked_package_matches_the_builder() -> None:
    assert scene.SCENE_XML.read_text() == scene.scene_xml()
    assert json.loads(scene.PLACES_JSON.read_text()) == {"places": scene.places()}


def test_dimos_loads_the_package() -> None:
    from dimos.simulation.scenes.catalog import resolve_scene_package

    pkg = resolve_scene_package(str(scene.PACKAGE_DIR))
    assert pkg is not None
    assert pkg.mujoco_scene_path == scene.SCENE_XML
    assert pkg.stats["name"] == "warehouse"
    raw = json.loads(scene.META_JSON.read_text())
    assert raw["artifact_frames"]["mujoco"] == "dimos_world"


def test_scene_is_static_with_a_named_floor() -> None:
    m = mujoco.MjModel.from_xml_path(str(scene.SCENE_XML))
    assert m.nq == 0 and m.nv == 0  # robots composed in keep qpos[7:] = their joints
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    for cam in cell.LAYOUT["cameras"]:
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0


@pytest.mark.parametrize("robot", ["unitree_go1", "unitree_g1"])
def test_composes_with_dimos_robot_models(robot: str) -> None:
    from dimos.simulation.mujoco.model import get_assets, get_model_xml

    m = mujoco.MjModel.from_xml_string(get_model_xml(robot, scene.SCENE_XML.read_text()), assets=get_assets())
    assert m.nq == {"unitree_go1": 19, "unitree_g1": 36}[robot]


def test_route_goes_around_the_racks() -> None:
    start = tuple(cell.LAYOUT["humanoid"]["home"])
    loc = cell.BY_ID["shelf-a"]
    goal = loc.stand_point(*loc.center)
    legs = cell.route(start, goal)
    assert len(legs) >= 2 and legs[-1] == goal
    # no leg crosses a rack footprint
    for (x0, y0, _), (x1, y1, _) in zip([start, *legs[:-1]], legs, strict=True):
        for k in range(51):
            t = k / 50
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            for rx0, ry0, rx1, ry1 in (sh["rect"] for sh in cell.LAYOUT["shelves"].values()):
                assert not (rx0 < x < rx1 and ry0 < y < ry1), (x, y)
    assert math.hypot(legs[-1][0] - goal[0], legs[-1][1] - goal[1]) < 1e-9


def test_prior_map_is_free_in_the_aisles_and_blocked_at_racks_and_outside() -> None:
    from dimos_worlds.warehouse.modules import warehouse_prior

    grid, p = warehouse_prior()

    def at(x: float, y: float) -> int:
        return int(grid[round((y - p.origin_y) / p.resolution), round((x - p.origin_x) / p.resolution)])

    home = cell.LAYOUT["quadruped"]["home"]
    assert at(home[0], home[1]) == 0
    assert at(12.0, 11.4) == 0 and at(15.0, 7.5) == 0  # cross aisle, rack aisle
    assert at(10.0, 5.0) == 100 and at(10.0, 5.5) == 100  # rack frame, inside a rack
    assert at(-2.0, 5.0) == 100  # outside the building: sealed
