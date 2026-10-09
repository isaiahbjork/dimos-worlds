"""Day / night switching for warehouse.blend, shared by render_frames.py and export_web.py. Runs inside Blender.

Objects carry a custom property `grp`:
  floor | shell | interior | roof   always present (roof is never exported to the web GLB)
  render_only                       detailed props (Poly Haven cartons and totes, full robot meshes): hidden for web bakes
  web_only                          cheap proxies of the same props, decimated robots: hidden for renders
  door_day / door_night             roll-up door curtain, open (day shift) or closed (night)
Lights carry w_day / w_night (watts); emissive materials carry em_day / em_night (Emission Strength).
"""
from __future__ import annotations

import bpy

DAY_HDRI_STRENGTH = 1.0
NIGHT_HDRI_STRENGTH = 0.015


def set_mode(scene, mode: str, web: bool = False) -> None:
    for o in bpy.data.objects:
        g = o.get("grp")
        if g == "door_day":
            o.hide_render = mode != "day"
        elif g == "door_night":
            o.hide_render = mode != "night"
        elif g == "render_only":
            o.hide_render = web
        elif g == "web_only":
            o.hide_render = not web
        if o.type == "LIGHT":
            w = float(o.get(f"w_{mode}", 0.0))
            o.data.energy = w
            o.hide_render = w <= 0.0
    for m in bpy.data.materials:
        if f"em_{mode}" in m and m.node_tree:
            b = m.node_tree.nodes.get("Principled BSDF")
            if b:
                b.inputs["Emission Strength"].default_value = float(m[f"em_{mode}"])
    w = scene.world
    env = w.node_tree.nodes["env"]
    bg = w.node_tree.nodes["Background"]
    env.image = bpy.data.images[w["day_hdri" if mode == "day" else "night_hdri"]]
    bg.inputs["Strength"].default_value = DAY_HDRI_STRENGTH if mode == "day" else NIGHT_HDRI_STRENGTH
    bpy.context.view_layer.update()


def set_engine(scene, samples: int) -> None:
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    scene.cycles.device = "CPU"
    for backend in ("OPTIX", "CUDA", "METAL", "HIP"):
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
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 3
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.light_sampling_threshold = 0.005
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
