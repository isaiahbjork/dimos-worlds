"""Demo plant: scene matches its generator, gauge truth, prior map, places, gauge-loop geometry."""
from __future__ import annotations

import json
import math

import pytest

from dimos_worlds.plant import gen_mjcf, go2_scene, plant_state


def test_scene_matches_its_generator() -> None:
    assert gen_mjcf.build(gen_mjcf.load()) == go2_scene.PLANT_XML.read_text()


def test_needle_angle_round_trips() -> None:
    spec = plant_state.instruments()["PG-102"]
    for v in (0.0, 2.5, 5.6, 8.9, 10.0):
        assert plant_state.value_at(spec, plant_state.needle_angle(spec, v)) == pytest.approx(v, abs=1e-9)
    assert plant_state.needle_angle(spec, 0.0) == 225.0
    assert plant_state.needle_angle(spec, 10.0) == -45.0


def test_events_set_truth() -> None:
    t = plant_state.truth(["gauge_high:PG-102:8.9", "gauge_low:TG-201", "valve_wrong:HV-202"])
    assert t["gauges"]["PG-102"]["value"] == 8.9 and not t["gauges"]["PG-102"]["in_normal"]
    assert t["gauges"]["TG-201"]["value"] < t["gauges"]["TG-201"]["normal"][0]
    assert t["gauges"]["PG-101"]["in_normal"]
    assert not t["valves"]["HV-202"]["as_expected"]
    with pytest.raises(plant_state.EventError):
        plant_state.truth(["gauge_high:M-103"])  # a motor is not a gauge


def test_state_file_round_trip(tmp_path) -> None:
    p = tmp_path / "state.json"
    plant_state.save_events(["leak:DT-101"], p)
    assert plant_state.load_events(p) == ["leak:DT-101"]
    with pytest.raises(plant_state.EventError):
        plant_state.save_events(["nonsense"], p)


def test_prior_map_keeps_every_station_reachable() -> None:
    mujoco = pytest.importorskip("mujoco")  # noqa: F841
    prior = go2_scene.occupancy()
    assert 0.02 < (prior["grid"] == 100).mean() < 0.5
    for p in go2_scene.places():
        if p["kind"] == "stay_out":
            assert go2_scene.cell(prior, p["pose"]["x"], p["pose"]["y"]) == 100
            continue
        assert go2_scene.cell(prior, p["pose"]["x"], p["pose"]["y"]) == 0, p["id"]
        assert go2_scene.clearance(prior, p["pose"]["x"], p["pose"]["y"]) >= 0.3, p["id"]
    for ins in go2_scene.layout()["instruments"]:
        if "stand" in ins:
            x, y, _ = ins["stand"]
            assert go2_scene.cell(prior, x, y) == 0, ins["tag"]


def test_every_gauge_faces_a_walkable_stand() -> None:
    for g in (i for i in go2_scene.layout()["instruments"] if i["kind"] == "gauge"):
        gx, gy, gz = g["pos"]
        sx, sy, _ = g["stand"]
        fx, fy = g["facing"]
        out = (sx - gx) * fx + (sy - gy) * fy  # the stand is in front of the face
        assert 0.5 < out < 2.0, g["tag"]
        assert 0.3 < gz < 2.5
        assert math.hypot(fx, fy) == pytest.approx(1.0)
