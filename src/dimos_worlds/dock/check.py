"""Verify the generated MJCF: loads, covers open/close at the spec timing, envelopes match DJI's published
dimensions, the drone launches from and lands on the pad with no interpenetration, cameras exist.

    python -m dimos_worlds.dock.check      # exits non-zero on any failure; prints a JSON summary
"""
from __future__ import annotations

import json
import sys

import mujoco
import numpy as np

from . import geometry as G
from .sim import FEEDS, DockSim

PEN_TOL_M = 0.003  # MuJoCo soft contacts: up to a few mm of solver penetration is normal resting contact


def envelope(sim: DockSim) -> tuple[float, float, float]:
    """World AABB of every visible dock mesh (not the drone, not the ground) at the current pose: the same
    triangles the renders show, so this is the envelope DJI's dimensions describe."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    m, d = sim.m, sim.d
    for g in range(m.ngeom):
        if g in sim.drone_geoms or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] == 3:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        w = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    return tuple(float(v) for v in (hi - lo))


def visual_clearance(sim: DockSim, spec: dict, search: float = 0.03) -> dict:
    """The drone's visual solids against the dock's (closed covers, body, pad) at the current pose, all in the dock
    frame: overlap volume per dock part (cm3) and the smallest gap to the covers (mm). Collision geoms are boxes and
    capsules; this checks the meshes the renders show."""
    from manifold3d import Manifold, OpType

    from . import shapes

    m, d = sim.m, sim.d
    dock = m.body(f"{sim.p}dock").id
    Rd, pd = d.xmat[dock].reshape(3, 3), d.xpos[dock]
    Rq, pq = d.xmat[sim.drone].reshape(3, 3), d.xpos[sim.drone]
    T = np.hstack([Rd.T @ Rq, (Rd.T @ (pq - pd))[:, None]])  # drone frame -> dock frame
    drone = Manifold.batch_boolean([mm.transform(T) for _, _, mm in shapes.drone(spec)], OpType.Add)
    dock_parts = [(n, mm) for n, _, mm in shapes.base_and_body(spec) + shapes.pad(spec) if isinstance(mm, Manifold)]
    dock_parts += list(shapes.cover_solids(spec).items())
    overlap = {}
    for n, mm in dock_parts:
        x = drone ^ mm
        if x.volume() > 1e-9:
            bb = x.bounding_box()
            overlap[n] = {"cm3": round(x.volume() * 1e6, 3), "depth_mm": round((bb[5] - bb[2]) * 1000, 2)}
    gap = min(drone.min_gap(mm, search) for mm in shapes.cover_solids(spec).values())
    return {"overlap": overlap, "cover_gap_mm": round(gap * 1000, 1)}


def run_checks() -> dict:
    """Every check on a fresh DockSim. Returns the report; report["fail"] lists what failed (empty = pass)."""
    spec = G.load_spec()
    out: dict = {"fail": []}
    sim = DockSim()
    out["mujoco"] = mujoco.__version__
    out["cameras"] = [sim.m.camera(i).name for i in range(sim.m.ncam)]
    for f in FEEDS:
        if f not in out["cameras"]:
            out["fail"].append(f"camera {f} missing")
    out["closed_lwh_m"] = [round(v, 3) for v in envelope(sim)]

    # cover timing
    sim.open()
    t_open = sim.run_until_idle()
    st = sim.state()
    out["open_s"] = round(t_open, 2)
    out["open_deg"] = round(st.cover_deg, 2)
    out["open_lwh_m"] = [round(v, 3) for v in envelope(sim)]
    if abs(t_open - spec["cover"]["travel_s"]) > 0.1:
        out["fail"].append(f"open took {t_open:.2f} s, spec {spec['cover']['travel_s']}")
    if st.state != "open" or abs(st.cover_deg - spec["cover"]["open_deg"]) > 1.0:
        out["fail"].append(f"not open: {st}")
    for key, pub in (("closed_lwh_m", spec["envelope"]["closed_lwh"]), ("open_lwh_m", spec["envelope"]["open_lwh"])):
        err = [abs(a - b) for a, b in zip(out[key], pub)]
        out[key.replace("lwh_m", "err_mm")] = [round(e * 1000) for e in err]
        if max(err) > 0.03:
            out["fail"].append(f"{key} {out[key]} vs DJI {pub}")

    # launch, hover, land
    sim.takeoff(2.0)
    out["airborne_z"] = round(float(sim.d.xpos[sim.drone][2]), 3)
    try:
        sim.close()
        out["fail"].append("close() allowed with the drone 2 m above the open dock")
    except RuntimeError:
        out["close_refused_in_sweep"] = True
    sim.fly_to(4.0, 2.0, 6.0)
    sim.close()
    t_close_air = sim.run_until_idle()
    sim.open()
    sim.run_until_idle()
    sim.fly_to(0.0, 0.0, sim.pad_pos[2] + 3.0)
    sim.land()
    st = sim.state()
    out["landed_offset_mm"] = round(float(np.linalg.norm(np.array(st.drone_pos) - sim.pad_pos)) * 1000, 1)
    if st.drone != "parked":
        out["fail"].append(f"drone not parked after landing: {st}")
    sim.close()
    t_close = sim.run_until_idle()
    out["close_s"] = round(t_close, 2)
    out["close_while_away_s"] = round(t_close_air, 2)
    if abs(t_close - spec["cover"]["travel_s"]) > 0.1:
        out["fail"].append(f"close took {t_close:.2f} s")
    if sim.state().state != "closed":
        out["fail"].append(f"not closed: {sim.state()}")
    vis = visual_clearance(sim, spec)
    out["visual_overlap"] = vis["overlap"]
    out["visual_cover_gap_mm"] = vis["cover_gap_mm"]
    for n, v in vis["overlap"].items():
        # resting on the pad floor sinks the belly pads by the soft-contact depth; anything else is a clash
        if n not in ("tray", "marker_orient") or v["depth_mm"] > PEN_TOL_M * 1000:
            out["fail"].append(f"drone mesh overlaps {n}: {v} (closed, parked)")
    if vis["cover_gap_mm"] <= 0:
        out["fail"].append("drone mesh touches the closed covers")
    out["max_penetration_mm"] = round(sim.max_pen * 1000, 2)
    out["drone_contacts_off_pad"] = sorted(sim.bad_contacts)
    if sim.max_pen > PEN_TOL_M:
        out["fail"].append(f"penetration {sim.max_pen * 1000:.1f} mm > {PEN_TOL_M * 1000} mm")
    if sim.bad_contacts:
        out["fail"].append(f"drone touched something other than the pad: {sorted(sim.bad_contacts)}")
    out["sim_time_s"] = round(float(sim.d.time), 1)
    return out


def main() -> int:
    out = run_checks()
    print(json.dumps(out, indent=1))
    return 1 if out["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
