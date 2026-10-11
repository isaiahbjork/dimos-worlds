"""Warm Blender worker for the plant (pov_service.py): robot POV, zoom and fixed-camera frames from one plant.blend.

    blender -b assets/cache/plant.blend --disable-autoexec -P render/pov_render.py -- [--samples 16]

Owned by pov_service.py (the HTTP side). Protocol: one JSON per line on stdin, one status line on stdout.
  in : {"id": "r1", "out": "/tmp/x.npy", "light": "day"|"night", "events": ["gauge_high:PG-102:8.9", ...],
        "view": {"pos": [x,y,z], "dir": [dx,dy,dz], "hfov_deg": 12, "w": 960, "h": 720} | {"camera": "cam3"},
        "lamp": 0..10}
  out: POV_DONE r1 <secs>    (the .npy is H x W x 3 float16, scene-linear)
       POV_FAIL r1 <reason>
Plant state (needles, valve pointers, puddles) is posed by scene_state.apply_state from the same events.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "lot"))  # dimos_worlds/lot/camera_model.py

import camera_model as CM  # noqa: E402
import scene_state as S  # noqa: E402

LAMP_W_AT_10 = 25.0  # head lamp, UNVERIFIED (same figure as the lot)


def free_camera(view: dict):
    cd = bpy.data.cameras.get("free_view") or bpy.data.cameras.new("free_view")
    o = bpy.data.objects.get("free_view") or bpy.data.objects.new("free_view", cd)
    if o.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(o)
    cd.sensor_fit = "HORIZONTAL"
    cd.angle = math.radians(view["hfov_deg"])
    cd.clip_start = 0.03
    cd.clip_end = 400
    o.location = view["pos"]
    o.rotation_euler = Vector(view["dir"]).to_track_quat("-Z", "Y").to_euler()
    return o


def head_lamp(level: int, pos, direction) -> None:
    o = bpy.data.objects.get("pov_lamp")
    if o is None:
        ld = bpy.data.lights.new("pov_lamp", "SPOT")
        ld.spot_size = math.radians(50)
        ld.spot_blend = 0.4
        ld.color = (1.0, 0.95, 0.88)
        o = bpy.data.objects.new("pov_lamp", ld)
        bpy.context.scene.collection.objects.link(o)
    o.hide_render = level <= 0
    o.data.energy = LAMP_W_AT_10 * max(0, min(level, 10)) / 10.0
    o.location = Vector(pos) + Vector(direction).normalized() * 0.05
    o.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()


def render_view(scene, view: dict, out: Path) -> None:
    if "camera" in view:
        scene.camera = bpy.data.objects[view["camera"]]
        scene.render.resolution_x, scene.render.resolution_y = CM.SRC_W, CM.SRC_H
    else:
        scene.camera = free_camera(view)
        scene.render.resolution_x, scene.render.resolution_y = int(view["w"]), int(view["h"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "16"
    exr = out.with_suffix(".exr")
    scene.render.filepath = str(exr)
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(str(exr))
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    arr = px.reshape(h, w, 4)[::-1, :, :3]
    np.save(out.with_suffix(".npy"), arr.astype(np.float16))
    bpy.data.images.remove(img)
    exr.unlink()


_mode: list[str] = []
_events: list[tuple] = []


def render_one(req: dict) -> float:
    scene = bpy.context.scene
    light = req.get("light", "day")
    if not _mode or _mode[0] != light:
        S.set_mode(scene, light)
        _mode[:] = [light]
    ev = tuple(req.get("events") or ())
    if not _events or _events[0] != ev:
        S.apply_state(list(ev))
        _events[:] = [ev]
    view = req["view"]
    if "pos" in view:
        head_lamp(int(req.get("lamp") or 0), view["pos"], view["dir"])
    elif bpy.data.objects.get("pov_lamp") is not None:
        bpy.data.objects["pov_lamp"].hide_render = True
    t = time.time()
    render_view(scene, view, Path(req["out"]).with_suffix(""))
    return time.time() - t


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    samples = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 16
    scene = bpy.context.scene
    S.set_engine(scene, samples)
    c = scene.cycles
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.04
    for den in (("OPENIMAGEDENOISE",) if os.environ.get("DIMOS_WORLDS_CYCLES_DEVICE") else ("OPTIX", "OPENIMAGEDENOISE")):
        try:
            c.denoiser = den
            break
        except TypeError:
            continue
    c.max_bounces = 4
    scene.render.use_persistent_data = True
    print(f"POV_READY cycles {samples}", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = "?"
        try:
            req = json.loads(line)
            rid = str(req["id"])
            print(f"POV_DONE {rid} {render_one(req):.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"POV_FAIL {rid} {type(e).__name__}: {str(e)[:200]}".replace("\n", " "), flush=True)


if __name__ == "__main__":
    main()
