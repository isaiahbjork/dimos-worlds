"""Headless tests for the lot scene package (no window, no DimOS workers)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from dimos_worlds.lot import camera_model as CM
from dimos_worlds.lot import events, gen_lot, go2_scene, layout
from dimos_worlds.lot.places import HOME, LOT_XML, SCENE_META, load_model, load_places


@pytest.fixture()
def lot():
    model = load_model()
    data = mujoco.MjData(model)
    events.reset(model, data)
    return model, data


# ---- scene package -------------------------------------------------------------------------------


def test_lot_xml_is_up_to_date():
    assert LOT_XML.read_text() == gen_lot.build(), "scene/lot.xml is stale: run python -m dimos_worlds.lot.gen_lot"


def test_scene_loads_in_mujoco(lot):
    model, data = lot
    assert model.nq == 0, "the lot must have no joints (robots rely on their own qpos layout)"
    assert model.ncam == len(layout.CAMERAS)
    assert model.nmocap == 2 + 23 + 1  # gates, row vehicles, person
    for _ in range(200):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.mocap_pos).all()


def test_scene_meta_loads_with_dimos():
    from dimos.simulation.scene_assets.spec import load_scene_package
    from dimos.simulation.scenes.catalog import resolve_scene_package

    pkg = resolve_scene_package(str(SCENE_META.parent))
    assert pkg is not None
    assert pkg.mujoco_scene_path == LOT_XML.resolve()
    assert load_scene_package(SCENE_META).alignment.y_up is False
    raw = json.loads(SCENE_META.read_text())
    assert raw["artifact_frames"]["mujoco"] == "dimos_world"


def test_scene_attaches_a_robot_with_mjspec():
    """The robot-agnostic path DimOS's MujocoSimModule uses: scene spec + attached robot spec."""
    scene = mujoco.MjSpec.from_file(str(LOT_XML))
    robot = mujoco.MjSpec()
    body = robot.worldbody.add_body(name="base", pos=[0, 0, 0.3])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.3, 0.15, 0.1])
    frame = scene.worldbody.add_frame(pos=[HOME.x, HOME.y, 0.0])
    scene.attach(robot, prefix="r-", frame=frame)
    model = scene.compile()
    assert model.nq == 7


def test_go2_scene_merges_with_dimos_go1_file():
    """go2_scene.scene_xml() through DimOS's own get_model_xml + the Go1 asset files."""
    pytest.importorskip("mujoco_playground")
    from dimos.simulation.mujoco.model import get_assets, get_model_xml

    try:
        assets = get_assets()
    except Exception as exc:  # data download unavailable (offline CI)
        pytest.skip(f"DimOS sim data unavailable: {exc}")
    model = mujoco.MjModel.from_xml_string(get_model_xml("unitree_go1", go2_scene.scene_xml()), assets=assets)
    assert model.nq == 19  # Go1 free joint + 12 joints, nothing from the lot
    assert model.keyframe("home") is not None
    names = [model.body(i).name for i in range(model.nbody)]
    assert names.count("person") == 1  # DimOS's mesh replaces the lot's capsule person


# ---- places --------------------------------------------------------------------------------------


def test_every_place_inside_the_lot_and_reachable():
    prior = go2_scene.occupancy()
    reach = go2_scene.reachable(prior, HOME.goal_xy, robot_radius=0.3)
    res = prior["resolution"]
    for p in load_places(with_home=True).values():
        for x, y in (p.xy, p.goal_xy):
            assert -layout.HX < x < layout.HX and -layout.HY < y < layout.HY, p.id
        x, y = p.goal_xy
        assert go2_scene.cell(prior, x, y) == 0, f"{p.id} goal is inside an obstacle"
        assert go2_scene.clearance(prior, x, y) >= 0.5, f"{p.id} goal has < 0.5 m clearance"
        iy, ix = int((y - prior["origin_y"]) / res), int((x - prior["origin_x"]) / res)
        assert reach[iy, ix], f"{p.id} not reachable from home with a 0.3 m robot radius"


def test_place_cameras_exist():
    for p in load_places().values():
        assert p.camera in layout.CAMERAS
        assert p.id in layout.PERSON_POSES


# ---- events --------------------------------------------------------------------------------------


def test_person_at_moves_the_person(lot):
    model, data = lot
    assert events.person_place(model, data) is None
    events.person_at(model, data, "north-fence")
    assert events.person_place(model, data) == "north-fence"
    assert events.render_state(model, data)["person"]["pose"] == "walk"
    events.apply(model, data, "person_clear")
    assert events.person_place(model, data) is None


def test_gate_open_moves_the_gate_and_frees_the_opening(lot):
    model, data = lot
    closed = go2_scene.occupancy_from(model, data)
    events.apply(model, data, "gate_open:front-gate")
    events.apply(model, data, "gate_open:gate-2")
    assert events.render_state(model, data)["gates"] == pytest.approx({"front-gate": -6.2, "gate-2": -85.0})
    opened = go2_scene.occupancy_from(model, data)
    # the middle of the front-gate opening (x -7, on the fence line) is blocked when shut, free when open
    assert go2_scene.cell(closed, -7.0, -24.85) == 100
    assert go2_scene.cell(opened, -7.0, -24.85) == 0
    events.apply(model, data, "gate_close:gate-2")
    assert "gate-2" not in events.render_state(model, data)["gates"]


def test_vehicle_moved(lot):
    model, data = lot
    assert events.vehicle_at_spot(model, data, 16) is None
    bid = events.vehicle_at_spot(model, data, 14)
    events.apply(model, data, "vehicle_moved:row-c:14:16")
    assert events.vehicle_at_spot(model, data, 16) == bid
    assert events.vehicle_at_spot(model, data, 14) is None
    assert "veh-14" in events.render_state(model, data)["vehicles"]
    with pytest.raises(ValueError, match="occupied"):
        events.apply(model, data, "vehicle_moved:row-c:13:16")
    with pytest.raises(ValueError, match="no vehicle"):
        events.apply(model, data, "vehicle_moved:row-c:14:15")


@pytest.mark.parametrize(
    "spec", ["person_at:nowhere", "gate_open:back-gate", "vehicle_moved:row-z:1:2", "vehicle_moved:row-a:1:7", "dance"]
)
def test_bad_event_specs_raise(lot, spec):
    model, data = lot
    with pytest.raises(ValueError):
        events.apply(model, data, spec)


def test_reset_restores_baseline(lot):
    model, data = lot
    base = data.mocap_pos.copy()
    for spec in ("person_at:row-a", "gate_open:gate-2", "vehicle_moved:row-c:14:16"):
        events.apply(model, data, spec)
    assert not np.allclose(base, data.mocap_pos)
    events.apply(model, data, "reset")
    assert np.allclose(base, data.mocap_pos)


# ---- camera model --------------------------------------------------------------------------------


def _source(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((CM.SRC_H, CM.SRC_W, 3)) * 0.2).astype(np.float32)


def test_camera_model_is_deterministic_for_a_seed():
    src = _source()
    a = CM.isp(src, 4.0, "night", ("cam5", "t0"))
    b = CM.isp(src, 4.0, "night", ("cam5", "t0"))
    c = CM.isp(src, 4.0, "night", ("cam5", "t1"))
    assert a.shape == (CM.OUT_H, CM.OUT_W, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert CM.to_jpeg(a, 80) == CM.to_jpeg(b, 80)


def test_camera_model_lens_fov():
    assert CM.effective_hfov_deg(2.8) == pytest.approx(100.0, abs=3.0)
    assert CM.effective_hfov_deg(4.0) == pytest.approx(80.0, abs=3.0)


def _gl_available() -> bool:
    if os.environ.get("DIMOS_WORLDS_NO_GL") == "1":
        return False
    try:
        r = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), 8, 8)
        r.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gl_available(), reason="no OpenGL context (set MUJOCO_GL=egl or osmesa)")
def test_cctv_render_deterministic_and_sees_events():
    from dimos_worlds.lot.cctv import CctvRenderer

    r = CctvRenderer()
    try:
        a = r.frame("cam6", (1,))
        assert np.array_equal(a, r.frame("cam6", (1,)))
        events.person_at(r.model, r.data, "north-fence")
        b = r.frame("cam6", (1,))
        changed = (np.abs(b.astype(int) - a.astype(int)).max(axis=2) > 30).sum()
        assert changed > 50, changed
    finally:
        r.close()


# ---- blueprint -----------------------------------------------------------------------------------


def test_blueprints_discoverable_via_entry_points():
    from dimos.core.coordination.blueprints import Blueprint
    from dimos.robot.external_blueprints import list_external_blueprint_names, resolve_external_blueprint_by_name

    names = list_external_blueprint_names()
    assert "dimos-worlds.go2-lot-night" in names
    assert "dimos-worlds.lot-cctv" in names
    assert isinstance(resolve_external_blueprint_by_name("dimos-worlds.go2-lot-night"), Blueprint)
    assert isinstance(resolve_external_blueprint_by_name("dimos-worlds.lot-cctv"), Blueprint)
