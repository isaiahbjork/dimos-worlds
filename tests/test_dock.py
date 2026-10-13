"""Dock 3-class station + M4D-class quadcopter: the MJCF loads and matches its generator, envelopes match the
published dimensions, the covers run through the actuator linkage in the spec's travel time, the quadcopter fits
under the closed covers, launch and landing, and the lot-roof example."""
from __future__ import annotations

import math
import os
import re

import mujoco
import numpy as np
import pytest

from dimos_worlds.dock import check, gen_mjcf
from dimos_worlds.dock import geometry as G
from dimos_worlds.dock.sim import DEFAULT_XML, FEEDS, DockSim

ENVELOPE_TOL_M = 0.03  # the open length is 24 mm short of DJI's 1760 mm: DJI's own drawings disagree there (README)
TIMING_TOL_S = 0.1
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")


@pytest.fixture(scope="module")
def spec() -> dict:
    return G.load_spec()


@pytest.fixture()
def sim() -> DockSim:
    return DockSim()


def _same_xml(a: str, b: str, tol: float = 1e-5) -> bool:
    """Equal up to the assets-hash comment and float noise (mesh inertia comes out of manifold3d)."""
    strip = [ln for ln in a.splitlines() if "Assets hash" not in ln], [ln for ln in b.splitlines() if "Assets hash" not in ln]
    if len(strip[0]) != len(strip[1]):
        return False
    for la, lb in zip(*strip):
        if _NUM.sub("#", la) != _NUM.sub("#", lb):
            return False
        na, nb = [float(x) for x in _NUM.findall(la)], [float(x) for x in _NUM.findall(lb)]
        if any(abs(x - y) > tol * max(1.0, abs(x)) for x, y in zip(na, nb)):
            return False
    return True


@pytest.mark.parametrize("scene", [True, False])
def test_mjcf_matches_its_generator(spec: dict, scene: bool) -> None:
    manifest = gen_mjcf.ensure_assets()
    xml = DEFAULT_XML if scene else DEFAULT_XML.with_name("dock_fixture.xml")
    assert _same_xml(gen_mjcf.build_xml(spec, manifest, scene=scene), xml.read_text())


def test_mjcf_loads_with_every_part(sim: DockSim) -> None:
    m = sim.m
    for body in ("dock", "cover_r", "cover_l", "act_cyl_r", "act_rod_r", "act_cyl_l", "act_rod_l", "drone"):
        assert m.body(body).id >= 0
    for joint in ("cover_r_hinge", "cover_l_hinge", "act_r_swing", "act_r_slide", "act_l_swing", "act_l_slide"):
        assert m.joint(joint).id >= 0
    assert {m.camera(i).name for i in range(m.ncam)} >= set(FEEDS)
    assert m.neq == 2  # one connect per actuator: the closed chain
    st = sim.state()
    assert (st.state, st.drone) == ("closed", "parked")


def test_fixture_loads_without_ground() -> None:
    m = mujoco.MjModel.from_xml_path(str(DEFAULT_XML.with_name("dock_fixture.xml")))
    assert not [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]


def test_envelopes_match_published_dimensions(sim: DockSim, spec: dict) -> None:
    closed = check.envelope(sim)
    assert closed == pytest.approx(spec["envelope"]["closed_lwh"], abs=ENVELOPE_TOL_M)
    sim.set_cover_deg(spec["cover"]["open_deg"])
    assert check.envelope(sim) == pytest.approx(spec["envelope"]["open_lwh"], abs=ENVELOPE_TOL_M)


def test_covers_open_and_close_through_the_linkage_in_spec_time(sim: DockSim, spec: dict) -> None:
    m, d = sim.m, sim.d
    travel, open_deg = spec["cover"]["travel_s"], spec["cover"]["open_deg"]
    eye = {s: m.site(f"act_rod_{s}_eye").id for s in "rl"}
    pin = {s: m.site(f"cover_{s}_pin").id for s in "rl"}
    worst_gap = 0.0
    strokes: list[float] = []

    def watch(seconds: float) -> None:
        nonlocal worst_gap
        for _ in range(int(seconds / m.opt.timestep / 10)):
            sim.step(10)
            for s in "rl":
                worst_gap = max(worst_gap, float(np.linalg.norm(d.site_xpos[eye[s]] - d.site_xpos[pin[s]])))
            strokes.append(float(d.qpos[m.jnt_qposadr[sim.j_act["r"][1]]]))

    sim.open()
    watch(travel / 2)
    assert sim.state().state == "opening"
    assert 0.3 * open_deg < sim.state().cover_deg < 0.7 * open_deg
    t_open = (travel / 2) + sim.run_until_idle()
    assert t_open == pytest.approx(travel, abs=TIMING_TOL_S)
    assert sim.state().state == "open"
    assert sim.state().cover_deg == pytest.approx(open_deg, abs=1.0)
    # the rod's eye stays on the cover pin (equality connect), and the rod moved through its stroke
    assert worst_gap < 0.003
    assert max(strokes) - min(strokes) > 0.02
    # physics agrees with the analytic linkage at the open end
    for s in "rl":
        swing, stroke = sim.linkage(open_deg, s)
        js, jl = sim.j_act[s]
        assert d.qpos[m.jnt_qposadr[js]] == pytest.approx(swing, abs=math.radians(1.0))
        assert d.qpos[m.jnt_qposadr[jl]] == pytest.approx(stroke, abs=0.003)
    sim.close()
    assert sim.run_until_idle() == pytest.approx(travel, abs=TIMING_TOL_S)
    assert sim.state().state == "closed"


def test_reversal_mid_travel(sim: DockSim, spec: dict) -> None:
    sim.open()
    sim.run(spec["cover"]["travel_s"] / 2)
    half = sim.state().cover_deg
    sim.close()
    t = sim.run_until_idle()
    assert sim.state().state == "closed"
    assert t == pytest.approx(half / spec["cover"]["open_deg"] * spec["cover"]["travel_s"], abs=TIMING_TOL_S)


def test_drone_fits_under_closed_covers(sim: DockSim, spec: dict) -> None:
    vis = check.visual_clearance(sim, spec)
    assert vis["cover_gap_mm"] > 0
    # only the belly pads may sink into the pad floor (soft contact), and by less than the contact tolerance
    for part, v in vis["overlap"].items():
        assert part in ("tray", "marker_orient") and v["depth_mm"] <= check.PEN_TOL_M * 1000, (part, v)
    sim.open()
    sim.run_until_idle()
    sim.close()
    sim.run_until_idle()
    assert not sim.bad_contacts  # closing over the parked drone touched nothing but its feet on the pad


def test_close_refused_while_drone_in_the_sweep(sim: DockSim) -> None:
    with pytest.raises(RuntimeError):
        sim.takeoff(2.0)  # covers closed: nowhere to go
    sim.open()
    sim.run_until_idle()
    sim.takeoff(2.0)
    with pytest.raises(RuntimeError):
        sim.close()


def test_check_report_passes() -> None:
    out = check.run_checks()
    assert out["fail"] == [], out["fail"]
    assert out["close_refused_in_sweep"]
    assert out["drone_contacts_off_pad"] == []


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
def test_feeds_render(sim: DockSim) -> None:
    for feed in FEEDS:
        img = sim.render(feed, 160, 90)
        assert img.shape[2] == 3 and img.std() > 1.0


def test_lot_roof_example() -> None:
    from dimos_worlds.dock.lot_example import PREFIX, lot_with_dock, roof_pose, round_trip

    sim = DockSim.from_model(lot_with_dock(), prefix=PREFIX)
    x, y, z = roof_pose()
    assert sim.pad_pos[:2] == pytest.approx((x, y), abs=1e-6) and sim.pad_pos[2] > z
    out = round_trip(sim)
    assert (out["state"], out["drone"]) == ("closed", "parked")
    assert out["open_s"] == pytest.approx(G.load_spec()["cover"]["travel_s"], abs=TIMING_TOL_S)
    assert out["off_pad_contacts"] == [] and out["max_penetration_mm"] <= check.PEN_TOL_M * 1000
