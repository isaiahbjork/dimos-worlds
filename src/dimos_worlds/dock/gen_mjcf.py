"""Write the MuJoCo models and mesh assets of the Dock 3-class station + M4D-class quadcopter from spec.json.

    python -m dimos_worlds.dock.gen_mjcf     # -> assets/ (meshes, untracked), dock.xml (scene) and dock_fixture.xml

dock_fixture.xml has only the dock and the drone (no ground, no light), for attaching into any env with
fixture.attach_dock(). Both are generated: edit the spec, shapes.py or geometry.py, never the XML. The meshes in
assets/ are rebuilt automatically (ensure_assets) whenever the spec or the shape code changes.

Model:
  * body dock (static): visual meshes (shapes.py) + simple collision solids (geometry.collision, group 3).
  * bodies cover_r / cover_l: hinges on the fitted pivots, 0..90 deg, driven by intvelocity actuators whose ctrl
    range is exactly open_angle / travel_s, so full ctrl opens (or closes) in the measured travel time.
  * linear actuators: act_cyl_* swings on its body pin, act_rod_* slides in it, and an equality <connect> ties the
    rod's eye to the cover pin, so the visible linkage follows the cover exactly (closed kinematic chain).
  * cameras dock_internal / dock_external ride cover_r; drone_wide rides the drone (DJI 84 deg diagonal, 4:3).
  * body drone: free joint, DJI mass 1.85 kg with an inertia tensor from a part split (shapes.drone_inertia).
Collision filter: the dock and covers never collide with each other (contype 1, conaffinity 0); the drone
(contype 0, conaffinity 1) collides with both and with the ground. Visual meshes collide with nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from . import geometry as G

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
SOURCES = ("spec.json", "shapes.py", "geometry.py", "gen_mjcf.py")


def _f(v, n=5) -> str:
    return " ".join(f"{x:.{n}f}".rstrip("0").rstrip(".") if abs(x) > 1e-12 else "0" for x in v)


# ---------------------------------------------------------------- assets

def source_hash() -> str:
    h = hashlib.sha256()
    for name in SOURCES:
        h.update((HERE / name).read_bytes())
    return h.hexdigest()[:16]


def _write_obj(path: Path, v, f) -> None:
    lines = [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in v]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


def mesh_groups(parts: list[dict]) -> dict[tuple[str, str], tuple]:
    """Merge parts per (parent, material): one MuJoCo mesh + geom each."""
    import numpy as np

    groups: dict[tuple[str, str], list] = {}
    for p in parts:
        groups.setdefault((p["parent"], p["mat"]), []).append(p)
    out = {}
    for key, ps in sorted(groups.items()):
        vs, fs, off = [], [], 0
        for p in ps:
            vs.append(p["verts"])
            fs.append(p["faces"] + off)
            off += len(p["verts"])
        out[key] = (np.concatenate(vs), np.concatenate(fs))
    return out


def write_assets(spec: dict | None = None, out: Path = ASSETS) -> dict:
    """Build every mesh (shapes.py) and write assets/*.obj, assets/parts.npz and assets/manifest.json."""
    import numpy as np

    from . import shapes

    spec = spec or G.load_spec()
    out.mkdir(parents=True, exist_ok=True)
    parts = shapes.build_parts(spec)
    names = []
    for (parent, mat), (v, f) in mesh_groups(parts).items():
        name = f"{parent}__{mat}"
        _write_obj(out / f"{name}.obj", v, f)
        names.append(name)
    for name, piece in shapes.pad_collision(spec).items():  # convex pieces (MuJoCo collides a mesh through its hull)
        v, f = shapes._mesh(piece)
        _write_obj(out / f"{name}.obj", v, f)
    arrays = {"names": np.array([p["name"] for p in parts]), "parents": np.array([p["parent"] for p in parts]),
              "mats": np.array([p["mat"] for p in parts])}
    for i, p in enumerate(parts):
        arrays[f"v{i}"] = p["verts"].astype(np.float32)
        arrays[f"f{i}"] = p["faces"].astype(np.int32)
    with tempfile.NamedTemporaryFile(dir=out, suffix=".npz", delete=False) as tmp:
        np.savez_compressed(tmp, **arrays)
    os.replace(tmp.name, out / "parts.npz")
    inertia = shapes.drone_inertia(spec)
    manifest = {"hash": source_hash(), "meshes": names, "drone_inertia": {
        "mass": inertia["mass"], "com": inertia["com"], "inertia": inertia["inertia"].tolist()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def ensure_assets(out: Path = ASSETS) -> dict:
    """Rebuild the meshes if they are missing or older than the spec / shape code."""
    try:
        man = json.loads((out / "manifest.json").read_text())
        if man.get("hash") == source_hash() and all((out / f"{n}.obj").is_file() for n in man["meshes"]):
            return man
    except (OSError, ValueError, KeyError):
        pass
    return write_assets(out=out)


# ---------------------------------------------------------------- MJCF

def _material(name: str) -> str:
    rgba, rough, metal = G.MATERIALS[name]
    return (f'<material name="{name}" rgba="{_f(rgba, 3)}" specular="{0.25 + 0.5 * metal:.2f}" '
            f'shininess="{max(0.05, 1 - rough):.2f}" reflectance="{0.15 * metal:.2f}"/>')


def _visual(parent: str, mats: list[str]) -> list[str]:
    return [f'<geom name="{parent}__{m}" type="mesh" mesh="{parent}__{m}" material="{m}" contype="0" conaffinity="0" '
            f'group="1"/>' for m in mats]


def _collision(p: dict, drone: bool) -> str:
    col = ('contype="0" conaffinity="1" condim="3" friction="0.9 0.01 0.001" solref="0.004 1" solimp="0.95 0.99 0.001"'
           if drone else 'contype="1" conaffinity="0"')
    common = f'name="{p["name"]}" {col} group="3" rgba="0.9 0.3 0.2 0.35"'
    quat = "" if p.get("quat", G.Q0) == G.Q0 else f' quat="{_f(p["quat"], 6)}"'
    k = p["kind"]
    if k == "box":
        return f'<geom {common} type="box" size="{_f(p["size"])}" pos="{_f(p["pos"])}"{quat}/>'
    if k == "cyl":
        return f'<geom {common} type="cylinder" size="{_f(p["size"])}" pos="{_f(p["pos"])}"{quat}/>'
    if k == "capsule":
        return f'<geom {common} type="capsule" size="{p["size"][0]}" fromto="{_f(p["from"])} {_f(p["to"])}"/>'
    if k == "hull":
        return f'<geom {common} type="mesh" mesh="{p["mesh"]}"/>'
    raise ValueError(k)


def _cam(c: dict, fovy: float) -> str:
    x, y = G.camera_axes(c["look"], c["up"])
    return (f'<camera name="{c["name"]}" pos="{_f(c["pos"])}" xyaxes="{_f(x, 6)} {_f(y, 6)}" fovy="{fovy:.2f}"/>\n'
            f'<site name="{c["name"]}_site" pos="{_f(c["pos"])}" size="0.006" rgba="1 0.2 0.2 0.6" group="3"/>')


def fisheye_vfov_deg(c: dict) -> float:
    """Vertical coverage of an equidistant fisheye (DJI's FOV is diagonal). MuJoCo is pinhole: sim.py renders
    wider and remaps to the true fisheye; this value only labels the camera."""
    return c["dfov_deg"] * c["height"] / math.hypot(c["width"], c["height"])


def build_xml(spec: dict, manifest: dict, *, scene: bool) -> str:
    cv = spec["cover"]
    speed = math.radians(cv["open_deg"]) / cv["travel_s"]
    open_r = math.radians(cv["open_deg"])
    groups: dict[str, list[str]] = {}
    for name in manifest["meshes"]:
        parent, mat = name.split("__")
        groups.setdefault(parent, []).append(mat)
    cols: dict[str, list[dict]] = {}
    for p in G.collision(spec):
        cols.setdefault(p["parent"], []).append(p)
    cams = {c["name"]: c for c in G.cameras(spec)}
    ind = "\n" + " " * 6

    def cover_body(side: str) -> str:
        pv = G.pivot(spec, side)
        axis = "0 1 0" if side == "r" else "0 -1 0"
        sx = 1 if side == "r" else -1
        body = _visual(f"cover_{side}", groups[f"cover_{side}"]) + [_collision(p, False) for p in cols[f"cover_{side}"]]
        if side == "r":
            body += [_cam(cams[n], fisheye_vfov_deg(cams[n])) for n in ("dock_internal", "dock_external")]
        _, pin = G.actuator_pins(spec, side)
        body.append(f'<site name="cover_{side}_pin" pos="{_f(G.closed_to_local(spec, pin, side))}" size="0.005" group="3"/>')
        inner = (ind + "  ").join("\n".join(body).split("\n"))
        return f'''
      <body name="cover_{side}" pos="{_f(pv)}">
        <inertial pos="{_f((-sx * 0.12, 0, 0.30))}" mass="{cv['mass_kg']}" diaginertia="0.28 0.22 0.16"/>
        <joint name="cover_{side}_hinge" type="hinge" axis="{axis}" range="0 {cv['open_deg']}" limited="true"
               damping="40" armature="0.2"/>
        {inner}
      </body>'''

    def actuator_bodies(side: str) -> str:
        bp, cp = G.actuator_pins(spec, side)
        d = G._sub(cp, bp)
        L0 = math.sqrt(sum(c * c for c in d))
        return f'''
      <body name="act_cyl_{side}" pos="{_f(bp)}" zaxis="{_f(d, 6)}">
        <inertial pos="0 0 0.06" mass="0.45" diaginertia="0.0012 0.0012 0.0001"/>
        <joint name="act_{side}_swing" type="hinge" axis="0 1 0" damping="0.05"/>
        {_visual(f"act_cyl_{side}", groups[f"act_cyl_{side}"])[0]}
        <body name="act_rod_{side}" pos="0 0 {L0:.6f}">
          <inertial pos="0 0 -0.07" mass="0.15" diaginertia="0.0004 0.0004 0.00001"/>
          <joint name="act_{side}_slide" type="slide" axis="0 0 1" range="-0.01 0.11" limited="true" damping="0.5"/>
          {_visual(f"act_rod_{side}", groups[f"act_rod_{side}"])[0]}
          <site name="act_rod_{side}_eye" pos="0 0 0" size="0.005" group="3"/>
        </body>
      </body>'''

    dock_geoms = ind.join(_visual("dock", groups["dock"]) + [_collision(p, False) for p in cols["dock"]])
    drone_geoms = (ind).join(_visual("drone", groups["drone"]) + [_collision(p, True) for p in cols["drone"]])
    di = manifest["drone_inertia"]
    inertia = di["inertia"]
    full = _f((inertia[0][0], inertia[1][1], inertia[2][2], inertia[0][1], inertia[0][2], inertia[1][2]), 7)
    wide = cams["drone_wide"]
    wide_fovy = G.dfov_to_fovy(wide["dfov_deg"], wide["width"], wide["height"])
    floor_z = spec["pad"]["floor_z"]
    mesh_assets = [f'<mesh name="{n}" file="{n}.obj" inertia="shell"/>' for n in manifest["meshes"]]
    mesh_assets += [f'<mesh name="{c["mesh"]}" file="{c["mesh"]}.obj"/>' for c in G.collision(spec) if c["kind"] == "hull"]
    mats = sorted({n.split("__")[1] for n in manifest["meshes"]})
    scene_assets = ('''
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.62 0.74 0.88" rgb2="0.95 0.96 0.98" width="256" height="256"/>
    <texture name="grass" type="2d" builtin="checker" rgb1="0.36 0.45 0.24" rgb2="0.33 0.42 0.22" width="512" height="512"/>
    <material name="grass" texture="grass" texrepeat="40 40" reflectance="0"/>''' if scene else "")
    scene_world = ('''
    <light name="sun" pos="2 -3 6" dir="-0.3 0.4 -1" directional="true" diffuse="0.8 0.8 0.78" specular="0.1 0.1 0.1" castshadow="true"/>
    <light name="fill" pos="-3 3 4" dir="0.4 -0.4 -1" directional="true" diffuse="0.35 0.36 0.4" castshadow="false"/>
    <geom name="ground" type="plane" size="30 30 0.1" material="grass" contype="1" conaffinity="0"/>
    <camera name="overview" pos="1.9 -2.3 1.5" xyaxes="0.77 0.64 0 -0.27 0.33 0.9" fovy="45"/>''' if scene else "")
    a_ = "\n    "
    return f'''<!-- GENERATED by gen_mjcf.py from spec.json + shapes.py + geometry.py. Do not edit. -->
<!-- Dock 3-class station + M4D-class quadcopter: generic models built to DJI's published dimensions (sources in spec.json). -->
<!-- Meshes: assets/ (generated, untracked; gen_mjcf.ensure_assets rebuilds them). Assets hash {manifest["hash"]} -->
<mujoco model="dock_station">
  <compiler angle="degree" inertiafromgeom="false" meshdir="assets"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.3 0.3 0.3"/>
  </visual>
  <asset>{scene_assets}
    {a_.join(_material(m) for m in mats)}
    {a_.join(mesh_assets)}
  </asset>
  <worldbody>{scene_world}
    <body name="dock" pos="0 0 0">
      {dock_geoms}{cover_body("r")}{cover_body("l")}{actuator_bodies("r")}{actuator_bodies("l")}
    </body>
    <body name="drone" pos="0 0 {floor_z + 0.0005:.4f}">
      <freejoint name="drone_free"/>
      <inertial pos="{_f(di['com'])}" mass="{di['mass']:.4f}" fullinertia="{full}"/>
      {drone_geoms}
      <camera name="drone_wide" pos="{_f(wide['pos'])}" xyaxes="0 -1 0 0 0 1" fovy="{wide_fovy:.2f}"/>
      <site name="drone_wide_site" pos="{_f(wide['pos'])}" size="0.005" group="3"/>
      <site name="drone_com" pos="{_f(di['com'])}" size="0.005" group="3"/>
    </body>
  </worldbody>
  <equality>
    <connect name="act_r_link" body1="act_rod_r" body2="cover_r" anchor="0 0 0" solref="0.004 1"/>
    <connect name="act_l_link" body1="act_rod_l" body2="cover_l" anchor="0 0 0" solref="0.004 1"/>
  </equality>
  <actuator>
    <intvelocity name="cover_r_act" joint="cover_r_hinge" kp="4000" kv="40" ctrlrange="{-speed:.6f} {speed:.6f}"
                 actrange="0 {open_r:.6f}" forcerange="-400 400"/>
    <intvelocity name="cover_l_act" joint="cover_l_hinge" kp="4000" kv="40" ctrlrange="{-speed:.6f} {speed:.6f}"
                 actrange="0 {open_r:.6f}" forcerange="-400 400"/>
  </actuator>
  <sensor>
    <jointpos name="cover_r_angle" joint="cover_r_hinge"/>
    <jointpos name="cover_l_angle" joint="cover_l_hinge"/>
    <jointpos name="act_r_stroke" joint="act_r_slide"/>
    <jointpos name="act_l_stroke" joint="act_l_slide"/>
    <framepos name="drone_pos" objtype="body" objname="drone"/>
  </sensor>
</mujoco>
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=HERE)
    a = ap.parse_args()
    spec = G.load_spec()
    manifest = write_assets(spec)
    (a.out_dir / "dock.xml").write_text(build_xml(spec, manifest, scene=True))
    (a.out_dir / "dock_fixture.xml").write_text(build_xml(spec, manifest, scene=False))
    print(f"wrote {len(manifest['meshes'])} meshes to {ASSETS}, {a.out_dir / 'dock.xml'} and dock_fixture.xml")


if __name__ == "__main__":
    main()
