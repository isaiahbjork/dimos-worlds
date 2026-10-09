"""Render the fixed cameras of warehouse.blend (Cycles, GPU) to linear float16 .npy, one per (light, camera).

    blender -b assets/cache/warehouse.blend --disable-autoexec -P render/render_frames.py -- \
        --out DIR [--lights day,night] [--cams cam1,cam2] [--samples 96]

The camera model (lens, noise, ISP, JPEG) runs outside Blender: render.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
for p in (Path(__file__).resolve().parents[2] / "lot",):
    if (p / "camera_model.py").exists():
        sys.path.insert(0, str(p))

import camera_model as CM  # noqa: E402
import modes  # noqa: E402


def arg(name, default=None):
    argv = sys.argv[sys.argv.index("--") + 1:]
    return argv[argv.index(name) + 1] if name in argv else default


def main() -> None:
    out = Path(arg("--out"))
    out.mkdir(parents=True, exist_ok=True)
    lights = arg("--lights", "day,night").split(",")
    cams = arg("--cams", "cam1,cam2,cam3,cam4").split(",")
    scene = bpy.context.scene
    modes.set_engine(scene, int(arg("--samples", "96")))
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "16"
    scene.render.use_compositing = False
    scene.render.resolution_x, scene.render.resolution_y = CM.SRC_W, CM.SRC_H
    scene.render.resolution_percentage = 100
    for light in lights:
        modes.set_mode(scene, light, web=False)
        for cam in cams:
            t = time.time()
            scene.camera = bpy.data.objects[cam]
            exr = out / f"{light}__{cam}.exr"
            scene.render.filepath = str(exr)
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(str(exr))
            w, h = img.size
            px = np.empty(w * h * 4, dtype=np.float32)
            img.pixels.foreach_get(px)
            arr = px.reshape(h, w, 4)[::-1, :, :3]
            np.save(out / f"{light}__{cam}.npy", arr.astype(np.float16))
            bpy.data.images.remove(img)
            exr.unlink()
            print(f"RENDERED {light} {cam} {time.time() - t:.1f}s mean {float(arr.mean()):.4f}", flush=True)


if __name__ == "__main__":
    main()
