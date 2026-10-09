"""Pose the Menagerie robots by forward kinematics and dump world-space visual meshes to assets/cache/robots.npz.

    python render/robots_fk.py [--iiwa-tip X Y Z] [--out PATH]   # plain python3 with `mujoco` + numpy

Robots are posed at their own origin (iiwa base at 0, G1 standing on z = 0); build_scene.py places them.
The iiwa's pose: by default its suction tool hovers over the pick table (the layout's reach target); --iiwa-tip puts
the tool tip at a site-frame point instead (pointing down, solved by damped least squares ), so
render.py --state can show the arm where a recorded run had it.
Per robot: <robot>/<k>/v (N,3 float32 metres), <robot>/<k>/f (M,3 int32), <robot>/<k>/rgba (4,), one entry per visual
geom. Licences: both models are BSD-3-Clause (see assets/manifest.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parents[1]
CACHE = HERE / "assets" / "cache"
LAY = json.loads((HERE / "layout.json").read_text())
ARM = LAY["arm_cell"]
TOOL_Z = 0.045 + ARM["tool_m"]  # flange to tool tip along link7's z
READY_Q = np.array([0.0, 0.3, 0.0, -1.6, 0.0, 1.2, 0.0])  # the warehouse arm's ready pose


def iiwa_ik(xml: Path, tip_site: np.ndarray) -> dict[str, float]:
    """Joint angles putting the tool tip at tip_site (site frame), tool pointing straight down."""
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    base = np.array([*ARM["pedestal_center"], ARM["pedestal_height"]], dtype=float)
    tip = tip_site - base
    link7 = m.body("link7").id
    jids = [m.joint(f"joint{i}").id for i in range(1, 8)]
    qadr = np.array([m.jnt_qposadr[j] for j in jids])
    dadr = np.array([m.jnt_dofadr[j] for j in jids])
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    d.qpos[qadr] = READY_Q
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    down = np.array([0.0, 0.0, -1.0])
    err = 1.0
    for _ in range(600):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rot = d.xmat[link7].reshape(3, 3)
        pos = d.xpos[link7] + rot @ np.array([0.0, 0.0, TOOL_Z])
        ep = tip - pos
        eo = np.cross(rot[:, 2], down)  # tool z onto world -z
        err = float(np.linalg.norm(ep))
        if err < 0.004 and np.linalg.norm(eo) < 0.03:
            break
        mujoco.mj_jac(m, d, jp, jr, pos, link7)
        jac = np.vstack([jp[:, dadr], jr[:, dadr]])
        e = np.concatenate([ep, 0.5 * eo])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.02 ** 2 * np.eye(6), e)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + np.clip(dq, -0.2, 0.2), lo, hi)
    print(f"IIWA IK tip {tip_site.round(3).tolist()} error {err * 100:.1f} cm", flush=True)
    return {f"joint{i + 1}": float(d.qpos[a]) for i, a in enumerate(qadr)}


def dump(name: str, xml: Path, key: str | None, joints: dict | None, out: dict) -> None:
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key) if key else -1
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(m, d, kid)
    for j, q in (joints or {}).items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        d.qpos[m.jnt_qposadr[jid]] = q
    mujoco.mj_forward(m, d)
    geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2]
    if not geoms:
        geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] == 0]
    tris = 0
    for k, g in enumerate(geoms):
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        v = m.mesh_vert[va:va + vn].astype(np.float64)
        f = m.mesh_face[fa:fa + fn].astype(np.int32)
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        out[f"{name}/{k}/v"] = w.astype(np.float32)
        out[f"{name}/{k}/f"] = f
        out[f"{name}/{k}/rgba"] = np.asarray(rgba, dtype=np.float32)
        tris += len(f)
    zs = np.concatenate([out[f"{name}/{k}/v"][:, 2] for k in range(len(geoms))])
    print(f"ROBOT {name}: {len(geoms)} geoms, {tris} tris, z {zs.min():.3f}..{zs.max():.3f} m")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iiwa-tip", type=float, nargs=3, default=None)
    ap.add_argument("--out", default=str(CACHE / "robots.npz"))
    a = ap.parse_args()
    out: dict = {}
    g1 = CACHE / "menagerie_unitree_g1" / "unitree_g1"
    iw = CACHE / "menagerie_kuka_iiwa_14" / "kuka_iiwa_14"
    xml = g1 / "g1.xml" if (g1 / "g1.xml").exists() else g1 / "g1_with_hands.xml"
    dump("g1", xml, "stand", None, out)
    tip = np.array(a.iiwa_tip if a.iiwa_tip else ARM["reach_targets"]["pick_table"], dtype=float)
    if not a.iiwa_tip:
        tip = tip + np.array([0.0, 0.0, ARM["clearance_m"]])  # hovering over the table
    dump("iiwa", iw / "iiwa14.xml", None, iiwa_ik(iw / "iiwa14.xml", tip), out)
    np.savez_compressed(a.out, **out)
    print("ROBOTS_OK", a.out)


if __name__ == "__main__":
    sys.exit(main())
