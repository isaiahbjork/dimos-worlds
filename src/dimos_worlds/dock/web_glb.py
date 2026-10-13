"""Web GLB of the Dock 3-class station + M4D-class quadcopter, from the same meshes and linkage as the sim.

    python -m dimos_worlds.dock.web_glb --out dock_station.glb [--tol-mm 0.4] [--fps 10]

No Blender. shapes.py builds the meshes (manifold3d; meshes over 4000 triangles simplified so no surface moves more
than --tol-mm), MuJoCo poses the bodies from dock.xml, and this module writes glTF 2.0 binary itself.

  * Nodes: a root "dock_station" turns MuJoCo's Z-up frame into glTF's Y-up (a dock-frame point (x, y, z) lands at
    (x, z, -y)); under it one node per body: dock, cover_r, cover_l, act_cyl_r, act_rod_r, act_cyl_l, act_rod_l,
    drone. One mesh per body, one primitive per material, crease-angle normals.
  * Rest pose: covers closed, the quadcopter parked on the pad.
  * Animation "covers_open": covers and actuators from closed (t 0) to open (t = spec cover.travel_s); play it
    backwards to close, or set its time from a cover angle (t = angle / open_deg * travel_s). The drone node has no
    channels, so a viewer can fly it.
  * extras on the root: open_deg, travel_s, close_clear_alt_m, drone_park (dock frame, m).
Same input, same bytes (no timestamps).
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

from . import geometry as G

BODIES = ("dock", "cover_r", "cover_l", "act_cyl_r", "act_rod_r", "act_cyl_l", "act_rod_l", "drone")
ANIMATED = ("cover_r", "cover_l", "act_cyl_r", "act_rod_r", "act_cyl_l", "act_rod_l")
Z_UP_TO_Y_UP = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))  # glTF quaternion (x, y, z, w): -90 deg about X
FLOAT, U16, U32 = 5126, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


def crease_normals(v: np.ndarray, f: np.ndarray, angle_deg: float = 35.0):
    """Per-corner normals: area-weighted over the faces round a vertex whose normals are within angle_deg of the
    corner's own face, so curved surfaces stay smooth and edges stay sharp. Returns (verts, normals, faces) with
    vertices split where needed."""
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length = 2 x area
    ln = np.linalg.norm(fn, axis=1)
    unit = fn / np.maximum(ln, 1e-30)[:, None]
    corner_v = f.ravel()
    corner_f = np.repeat(np.arange(len(f)), 3)
    order = np.argsort(corner_v, kind="stable")
    cv, cf = corner_v[order], corner_f[order]
    starts = np.flatnonzero(np.r_[True, cv[1:] != cv[:-1]])
    counts = np.diff(np.r_[starts, len(cv)])
    group = np.repeat(np.arange(len(starts)), counts)  # vertex group of each sorted corner
    size = counts[group]
    a = np.repeat(np.arange(len(cv)), size)  # every corner paired with every corner of its vertex
    first = np.repeat(starts[group], size)
    b = first + (np.arange(len(a)) - np.repeat(np.cumsum(size) - size, size))
    ok = np.einsum("ij,ij->i", unit[cf[a]], unit[cf[b]]) >= math.cos(math.radians(angle_deg))
    acc = np.zeros((len(cv), 3))
    np.add.at(acc, a[ok], fn[cf[b[ok]]])
    nrm = np.linalg.norm(acc, axis=1)
    bad = nrm < 1e-20
    acc[bad] = unit[cf[bad]]
    acc[bad & (np.linalg.norm(acc, axis=1) < 0.5)] = (0.0, 0.0, 1.0)
    acc /= np.linalg.norm(acc, axis=1)[:, None]
    n_corner = np.empty_like(acc)
    n_corner[order] = acc
    key = np.column_stack([corner_v.astype(np.float64), np.round(n_corner * 1000.0)])  # split on 0.001 normal steps
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    idx = np.zeros(len(uniq), dtype=np.int64)
    idx[inv] = np.arange(len(inv))  # any corner of each new vertex
    return v[corner_v[idx]], n_corner[idx], inv.reshape(-1, 3)


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


class _Glb:
    def __init__(self) -> None:
        self.bin = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, arr: np.ndarray, kind: str, target: int | None = None, minmax: bool = False) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        comp = {np.dtype(np.float32): FLOAT, np.dtype(np.uint16): U16, np.dtype(np.uint32): U32}[arr.dtype]
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": arr.nbytes}
        if target:
            view["target"] = target
        self.bin += arr.tobytes()
        self.views.append(view)
        acc = {"bufferView": len(self.views) - 1, "componentType": comp, "count": int(arr.shape[0]), "type": kind}
        if minmax:
            flat = arr.reshape(arr.shape[0], -1)
            acc["min"] = [float(x) for x in flat.min(axis=0)]
            acc["max"] = [float(x) for x in flat.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def write(self, doc: dict, path: Path) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        doc["buffers"] = [{"byteLength": len(self.bin)}]
        doc["bufferViews"] = self.views
        doc["accessors"] = self.accessors
        js = json.dumps(doc, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            fh.write(struct.pack("<4sII", b"glTF", 2, total))
            fh.write(struct.pack("<I4s", len(js), b"JSON") + js)
            fh.write(struct.pack("<I4s", len(self.bin), b"BIN\x00") + bytes(self.bin))
        return total


def _quat_xyzw(q_wxyz) -> list[float]:
    w, x, y, z = (float(c) for c in q_wxyz)
    return [x, y, z, w]


def body_poses(spec: dict, degs: list[float]) -> dict[str, list[tuple[list[float], list[float]]]]:
    """World translation and rotation (glTF x, y, z, w) of every body at each cover angle, from MuJoCo."""
    from .sim import DockSim

    sim = DockSim()
    out: dict[str, list] = {b: [] for b in BODIES}
    ids = {b: sim.m.body(b).id for b in BODIES}
    for deg in degs:
        sim.set_cover_deg(deg)
        for b, i in ids.items():
            out[b].append(([float(c) for c in sim.d.xpos[i]], _quat_xyzw(sim.d.xquat[i])))
    return out


def build(out: Path, tol_mm: float = 0.4, fps: float = 10.0, crease_deg: float = 35.0) -> dict:
    from . import shapes
    from .gen_mjcf import mesh_groups

    spec = G.load_spec()
    parts = shapes.build_parts(spec, simplify_tol=tol_mm / 1000.0)
    groups = mesh_groups(parts)  # (parent, material) -> merged (verts, faces) in the parent body's frame
    cv = spec["cover"]
    travel, open_deg = float(cv["travel_s"]), float(cv["open_deg"])
    n_keys = max(2, int(round(travel * fps)) + 1)
    times = np.linspace(0.0, travel, n_keys)
    poses = body_poses(spec, [open_deg * t / travel for t in times])

    glb = _Glb()
    mats = sorted({m for _, m in groups})
    mat_index = {m: i for i, m in enumerate(mats)}
    materials = []
    for m in mats:
        (r, g, b, a), rough, metal = G.MATERIALS[m]
        mat = {"name": m, "pbrMetallicRoughness": {
            "baseColorFactor": [round(srgb_to_linear(r), 5), round(srgb_to_linear(g), 5), round(srgb_to_linear(b), 5), a],
            "metallicFactor": metal, "roughnessFactor": rough}}
        if a < 1.0:
            mat["alphaMode"] = "BLEND"
        materials.append(mat)

    meshes, nodes, tris = [], [], 0
    for body in BODIES:
        prims = []
        for (parent, m), (v, f) in sorted(groups.items()):
            if parent != body:
                continue
            vv, nn, ff = crease_normals(np.asarray(v, np.float64), np.asarray(f, np.int64), crease_deg)
            pos = glb.add(vv.astype(np.float32), "VEC3", ARRAY_BUFFER, minmax=True)
            nrm = glb.add(nn.astype(np.float32), "VEC3", ARRAY_BUFFER)
            idx_t = np.uint16 if len(vv) < 65536 else np.uint32
            ind = glb.add(ff.astype(idx_t).ravel(), "SCALAR", ELEMENT_ARRAY_BUFFER)
            prims.append({"attributes": {"POSITION": pos, "NORMAL": nrm}, "indices": ind, "material": mat_index[m]})
            tris += len(ff)
        t0, q0 = poses[body][0]
        node = {"name": body, "translation": t0, "rotation": q0}
        if prims:
            meshes.append({"name": body, "primitives": prims})
            node["mesh"] = len(meshes) - 1
        nodes.append(node)

    park = poses["drone"][0][0]
    root = {"name": "dock_station", "rotation": list(Z_UP_TO_Y_UP), "children": list(range(len(nodes))),
            "extras": {"open_deg": open_deg, "travel_s": travel,
                       "close_clear_alt_m": float(spec["ops"]["close_clear_alt_m"]),
                       "drone_park": [round(c, 4) for c in park],
                       "frame": "dock frame under this node: x toward the gauge cover, -y front, z up; metres"}}
    nodes.append(root)

    t_acc = glb.add(times.astype(np.float32), "SCALAR", minmax=True)
    samplers, channels = [], []
    for body in ANIMATED:
        ni = BODIES.index(body)
        tr = np.array([p[0] for p in poses[body]], np.float32)
        rq = np.array([p[1] for p in poses[body]], np.float64)
        for k in range(1, len(rq)):  # keep each key on the same hemisphere as the last, for a short slerp
            if np.dot(rq[k], rq[k - 1]) < 0:
                rq[k] = -rq[k]
        for path, arr, kind in (("translation", tr, "VEC3"), ("rotation", rq.astype(np.float32), "VEC4")):
            samplers.append({"input": t_acc, "output": glb.add(arr, kind), "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": ni, "path": path}})

    doc = {"asset": {"version": "2.0", "generator": "dimos_worlds.dock.web_glb (manifold3d + MuJoCo, no Blender)"},
           "scene": 0, "scenes": [{"nodes": [len(nodes) - 1]}],
           "nodes": nodes, "meshes": meshes, "materials": materials,
           "animations": [{"name": "covers_open", "samplers": samplers, "channels": channels}]}
    size = glb.write(doc, out)
    return {"out": str(out), "bytes": size, "triangles": tris, "primitives": sum(len(m["primitives"]) for m in meshes),
            "keys": n_keys}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tol-mm", type=float, default=0.4, help="simplify heavy meshes to this surface tolerance")
    ap.add_argument("--fps", type=float, default=10.0, help="animation keys per second")
    a = ap.parse_args()
    print(json.dumps(build(a.out, a.tol_mm, a.fps)))


if __name__ == "__main__":
    main()
