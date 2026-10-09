"""Render jobs from lot.blend: apply each job's state, render each requested camera, save linear float16 .npy.

    blender -b assets/cache/lot.blend --disable-autoexec -P render/render_frames.py -- --jobs jobs.json --out DIR

jobs.json:
  {"engine": "eevee"|"cycles", "light": "night"|"day", "samples": 32,
   "jobs": [{"id": "j0",
             "state": {"vehicles": {"veh-14": [x, y, yaw_deg]}, "gates": {"front-gate": -6.2},
                       "person": {"x": 1, "y": 2, "yaw": 90, "pose": "walk"} | null},
             "views": [{"id": "cam1", "camera": "cam1"},
                       {"id": "free", "pos": [x, y, z], "dir": [dx, dy, dz], "hfov_deg": 82, "w": 1280, "h": 720}]}]}

Writes DIR/<job id>__<view id>.npy (H x W x 3 float16, scene-linear). The camera model in camera_model.py does
the rest outside Blender. State is absolute: every job starts from the baseline (vehicles at their built
positions, gates closed, person parked).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import camera_model as CM  # noqa: E402
import layout as L  # noqa: E402
import person as P  # noqa: E402

DAY_HDRI_STRENGTH = 1.0
NIGHT_HDRI_STRENGTH = 0.02  # moonless sky with suburban skyglow: dim, as at a real lot


def set_engine(scene, engine: str, samples: int) -> None:
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        scene.cycles.device = "CPU"
        for backend in ("METAL", "OPTIX", "CUDA", "HIP", "ONEAPI"):  # Mac first, then a Linux GPU box
            try:
                prefs.compute_device_type = backend
                prefs.get_devices()
            except TypeError:
                continue
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                print(f"CYCLES_DEVICE {backend} {[d.name for d in gpus]}", flush=True)
                break
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 4
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 2
        scene.cycles.transparent_max_bounces = 16
        scene.cycles.light_sampling_threshold = 0.01
    else:
        for ident in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
            try:
                scene.render.engine = ident
                break
            except TypeError:
                continue
        ee = scene.eevee
        ee.taa_render_samples = samples
        for attr, val in (("use_shadows", True), ("use_raytracing", True), ("shadow_ray_count", 1),
                          ("shadow_step_count", 6), ("use_gtao", True), ("light_threshold", 0.001),
                          ("shadow_pool_size", "1024")):
            if hasattr(ee, attr):
                setattr(ee, attr, val)
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.exr_codec = "NONE"
    scene.render.use_compositing = False
    scene.render.use_sequencer = False


def set_light_mode(scene, mode: str) -> None:
    w = scene.world
    env = w.node_tree.nodes["env"]
    bg = w.node_tree.nodes["Background"]
    if mode == "day":
        env.image = bpy.data.images[w["day_hdri"]]
        bg.inputs["Strength"].default_value = DAY_HDRI_STRENGTH
    else:
        env.image = bpy.data.images[w["night_hdri"]]
        bg.inputs["Strength"].default_value = NIGHT_HDRI_STRENGTH
    on = mode == "night"
    for o in bpy.data.collections["lights"].objects:
        o.hide_render = not on
    for m in bpy.data.materials:
        if "night_emission" not in m:
            continue
        b = m.node_tree.nodes.get("Principled BSDF")
        if m.get("emission_via_link"):
            mr = [n for n in m.node_tree.nodes if n.bl_idname == "ShaderNodeMapRange"][0]
            scale = 1.0 if on else 0.05
            mr.inputs["To Min"].default_value = 0.03 * scale
            mr.inputs["To Max"].default_value = 0.12 * scale
        else:
            b.inputs["Emission Strength"].default_value = m["night_emission"] * (1.0 if on else 0.02)


BASE: dict[str, tuple] = {}


def remember_baseline() -> None:
    for o in bpy.data.objects:
        if o.name.startswith("veh-") or o.name in L.GATES or o.name == "person":
            BASE[o.name] = (o.location.copy(), o.rotation_euler.copy())


def apply_state(state: dict) -> None:
    for name, (loc, rot) in BASE.items():
        o = bpy.data.objects[name]
        o.location = loc
        o.rotation_euler = rot
    for name, (x, y, yaw) in (state.get("vehicles") or {}).items():
        o = bpy.data.objects[name]
        o.location = (x, y, 0.0)
        o.rotation_euler = (0, 0, math.radians(yaw))
    for gid, value in (state.get("gates") or {}).items():
        o = bpy.data.objects[gid]
        g = L.GATES[gid]
        ox, oy = g["origin"]
        if g["type"] == "slide":
            o.location = (ox + g["axis"][0] * value, oy + g["axis"][1] * value, 0.0)
        else:
            o.rotation_euler = (0, 0, math.radians(value))
    arm = bpy.data.objects["person"]
    p = state.get("person")
    if p:
        arm.location = (p["x"], p["y"], 0.0)
        arm.rotation_euler = (0, 0, math.radians(p["yaw"] + arm.get("yaw_offset", 0.0)))
        P.apply_pose(arm, p["pose"])
    else:
        arm.location = (*L.PERSON_PARK, 0.0)
        P.apply_pose(arm, "stand")
    bpy.context.view_layer.update()


def free_camera(view: dict) -> bpy.types.Object:
    cd = bpy.data.cameras.get("free_view") or bpy.data.cameras.new("free_view")
    o = bpy.data.objects.get("free_view") or bpy.data.objects.new("free_view", cd)
    if o.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(o)
    cd.sensor_fit = "HORIZONTAL"
    cd.angle = math.radians(view["hfov_deg"])
    cd.clip_start = 0.1
    cd.clip_end = 800
    o.location = view["pos"]
    o.rotation_euler = Vector(view["dir"]).to_track_quat("-Z", "Y").to_euler()
    return o


def render_view(scene, view: dict, out: Path) -> None:
    if "camera" in view:
        scene.camera = bpy.data.objects[view["camera"]]
        scene.render.resolution_x, scene.render.resolution_y = CM.SRC_W, CM.SRC_H
    else:
        scene.camera = free_camera(view)
        scene.render.resolution_x, scene.render.resolution_y = view["w"], view["h"]
    scene.render.resolution_percentage = 100
    exr = out.with_suffix(".exr")
    scene.render.filepath = str(exr)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(exr))
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    arr = px.reshape(h, w, 4)[::-1, :, :3]  # Blender stores bottom row first
    np.save(out.with_suffix(".npy"), arr.astype(np.float16))
    bpy.data.images.remove(img)
    exr.unlink()


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:]
    spec = json.loads(Path(argv[argv.index("--jobs") + 1]).read_text())
    out_dir = Path(argv[argv.index("--out") + 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    set_engine(scene, spec.get("engine", "eevee"), int(spec.get("samples", 32)))
    set_light_mode(scene, spec.get("light", "night"))
    remember_baseline()
    for job in spec["jobs"]:
        apply_state(job.get("state") or {})
        for view in job["views"]:
            t = time.time()
            render_view(scene, view, out_dir / f"{job['id']}__{view['id']}")
            print(f"RENDERED {job['id']} {view['id']} {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
