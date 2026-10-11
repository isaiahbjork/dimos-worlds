"""Day / night and plant state for plant.blend. Runs inside Blender (build_scene, pov_render).

set_mode(scene, "day"|"night"): lights by w_day / w_night, emissive materials by em_day / em_night, world HDRI.
apply_state(events): poses needle:<tag> from plant_state.truth (deterministic needle angle), valve pointers, stems,
the ball-valve lever, and shows puddle:<tray> for leaks. Same truth as every other reader of plant_state.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import bpy

ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENV))

import plant_state  # noqa: E402

PLATE_OPEN_DEG = 135.0  # = gauges.PLATE_OPEN_DEG / PLATE_SHUT_DEG (gauges needs Pillow, Blender has none)
PLATE_SHUT_DEG = 45.0
DAY_HDRI_STRENGTH = 1.0
NIGHT_HDRI_STRENGTH = 0.02


def set_mode(scene, mode: str) -> None:
    for o in bpy.data.objects:
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
    scene["mode"] = mode
    bpy.context.view_layer.update()


def set_engine(scene, samples: int) -> None:
    """Cycles on the GPU when there is one (OptiX, then CUDA), else CPU."""
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    scene.cycles.device = "CPU"
    # DIMOS_WORLDS_CYCLES_DEVICE=CUDA skips OptiX (a box whose driver cannot compile Blender's OptiX kernels)
    order = [os.environ["DIMOS_WORLDS_CYCLES_DEVICE"]] if os.environ.get("DIMOS_WORLDS_CYCLES_DEVICE") else ["OPTIX", "CUDA", "METAL", "HIP"]
    for backend in order:
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
    c = scene.cycles
    c.samples = samples
    c.use_denoising = True
    c.max_bounces = 6
    c.diffuse_bounces = 4
    c.glossy_bounces = 3
    c.transparent_max_bounces = 8
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0


def apply_state(events: list[str]) -> dict:
    t = plant_state.truth(events)
    obj = bpy.data.objects
    for tag, g in t["gauges"].items():
        n = obj.get(f"needle:{tag}")
        if n:
            n.rotation_euler = (0.0, 0.0, math.radians(g["angle_deg"]))
    ins = plant_state.instruments()
    for tag, v in t["valves"].items():
        is_open = v["state"] == "open"
        p = obj.get(f"pointer:{tag}")
        if p:
            p.rotation_euler = (0.0, 0.0, math.radians(PLATE_OPEN_DEG if is_open else PLATE_SHUT_DEG))
        s = obj.get(f"stem:{tag}")
        if s:
            s.scale = (1.0, 1.0, 1.0 if is_open else 0.12)  # rising stem: out when open
        lv = obj.get(f"lever:{tag}")
        if lv:
            # quarter-turn ball valve, line along y: lever along the pipe (toward the walkway) = open, across = closed
            fx, fy = ins[tag]["facing"]
            along = math.atan2(fy, fx)
            lv.rotation_euler = (0.0, 0.0, along if is_open else along + math.pi / 2)
    for tag, tr in t["trays"].items():
        pd = obj.get(f"puddle:{tag}")
        if pd:
            pd.hide_render = not tr["leak"]
            pd.hide_viewport = not tr["leak"]
    bpy.context.view_layer.update()
    return t
