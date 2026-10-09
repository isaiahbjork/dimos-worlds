"""Blender side of render_heroes.py: open a built scene, place the Menagerie robots, render one still with Cycles.

    blender -b SCENE.blend --disable-autoexec -P docs/media/hero_blender.py -- --scene lot|warehouse \
        --robots robots.npz --place go2:X,Y,YAW_DEG [--place g1:X,Y,YAW_DEG] --cam X,Y,Z --look X,Y,Z \
        [--hfov 55] [--light night|day] [--samples 256] [--exposure 0] [--size 1600x900] --out OUT.png

Robots already in the scene (the warehouse build has g1_* and go2_* meshes) are moved; missing ones are built from
the npz written by warehouse/render/robots_fk.py. Output is display-referred (AgX), PNG; render_heroes.py makes
the JPEG.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

PKG = Path(__file__).resolve().parents[2] / "src" / "dimos_worlds"


def args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1:]
    out: dict = {"place": []}
    i = 0
    while i < len(argv):
        k, v = argv[i].lstrip("-"), argv[i + 1]
        if k == "place":
            out["place"].append(v)
        else:
            out[k] = v
        i += 2
    return out


def vec(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(","))


def robot_objects(name: str) -> list:
    return [o for o in bpy.data.objects if o.name.startswith(f"{name}_") and o.type == "MESH"]


def build_robot(data, name: str) -> list:
    """One mesh per material colour, like warehouse/render/build_scene.py robots()."""
    ks = sorted({int(k.split("/")[1]) for k in data.files if k.startswith(name + "/")})
    if not ks:
        raise SystemExit(f"{name} not in the robots npz (fetch the warehouse assets, rerun robots_fk.py)")
    groups: dict = {}
    for k in ks:
        groups.setdefault(tuple(round(float(x), 3) for x in data[f"{name}/{k}/rgba"]), []).append(k)
    objs = []
    for rgba, klist in groups.items():
        vs, fs, off = [], [], 0
        for k in klist:
            v, f = data[f"{name}/{k}/v"], data[f"{name}/{k}/f"]
            vs.append(v)
            fs.append(f + off)
            off += len(v)
        me = bpy.data.meshes.new(f"{name}_{len(objs)}")
        me.from_pydata(np.concatenate(vs).tolist(), [], np.concatenate(fs).tolist())
        me.update()
        for p in me.polygons:
            p.use_smooth = True
        mat = bpy.data.materials.new(f"hero_{name}_{len(objs)}")
        mat.use_nodes = True
        b = mat.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*[c ** 2.2 for c in rgba[:3]], 1.0)  # Menagerie rgba is sRGB-ish
        b.inputs["Roughness"].default_value = 0.4
        b.inputs["Metallic"].default_value = 0.1
        me.materials.append(mat)
        o = bpy.data.objects.new(f"{name}_{len(objs)}", me)
        bpy.context.scene.collection.objects.link(o)
        objs.append(o)
    return objs


def main() -> None:
    a = args()
    scene = bpy.context.scene
    light = a.get("light", "night")
    if a["scene"] == "lot":
        sys.path[:0] = [str(PKG / "lot" / "render"), str(PKG / "lot")]
        import render_frames as RF  # noqa: PLC0415

        RF.set_engine(scene, "cycles", int(a.get("samples", 256)))
        RF.set_light_mode(scene, light)
        RF.remember_baseline()
        RF.apply_state({})
    else:
        sys.path[:0] = [str(PKG / "warehouse" / "render")]
        import modes  # noqa: PLC0415

        modes.set_engine(scene, int(a.get("samples", 256)))
        modes.set_mode(scene, light, web=False)
    data = np.load(a["robots"]) if "robots" in a else None
    for spec in a["place"]:
        name, pose = spec.split(":")
        x, y, yaw = vec(pose)
        objs = robot_objects(name)
        if not objs:
            objs = build_robot(data, name)
        z0 = -min(min(v.co.z for v in o.data.vertices) for o in objs)  # meshes are in the robot's own frame
        for o in objs:
            o.location = (x, y, max(z0, 0.0))  # feet on the floor (the Go2's home pose sinks them ~2 cm)
            o.rotation_euler = (0.0, 0.0, math.radians(yaw))
            o.hide_render = False
    cam_data = bpy.data.cameras.new("hero")
    cam = bpy.data.objects.new("hero", cam_data)
    scene.collection.objects.link(cam)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle = math.radians(float(a.get("hfov", 55)))
    cam_data.clip_start, cam_data.clip_end = 0.05, 800.0
    cam.location = vec(a["cam"])
    cam.rotation_euler = (Vector(vec(a["look"])) - Vector(vec(a["cam"]))).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam
    w, h = (int(v) for v in a.get("size", "1600x900").split("x"))
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = w, h, 100
    for vt in ("AgX", "Filmic"):
        try:
            scene.view_settings.view_transform = vt
            break
        except TypeError:
            continue
    scene.view_settings.look = "None"
    scene.view_settings.exposure = float(a.get("exposure", 0.0))
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = a["out"]
    bpy.ops.render.render(write_still=True)
    print("HERO_OK", a["out"], flush=True)


if __name__ == "__main__":
    main()
