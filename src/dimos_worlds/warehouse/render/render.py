"""Fixed-camera frames of the warehouse (Cycles in Blender, then the lot package's camera model). Optional.

    python src/dimos_worlds/warehouse/render/render.py --out DIR [--lights day,night] [--cams cam1,cam2,cam3,cam4]
        [--samples 96] [--exposure-day 1.0] [--exposure-night 1.0] [--captured-at ISO8601] [--state STATE.json]

Needs numpy + Pillow + Blender (DIMOS_WORLDS_BLENDER, default `blender` on PATH; a GPU makes it practical). Builds
assets/cache/warehouse.blend first if missing (needs robots.npz and the asset cache: assets/fetch.py).
Writes DIR/<light>_<cam>.jpg (1280x720, JPEG q80) and DIR/manifest.json.

--state poses the scene as a recorded moment (tote poses, the G1, what each body holds, the arm's tool tip; format in
render/build_scene.py), with its own robots npz (robots_fk.py --iiwa-tip) and its own .blend.

UNVERIFIED in this package: ported with the paths changed; not run since (no Blender run for this release).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]  # src/dimos_worlds/warehouse
for p in (HERE.parent / "lot",):
    if (p / "camera_model.py").exists():
        sys.path.insert(0, str(p))
import camera_model as CM  # noqa: E402

BLEND = HERE / "assets" / "cache" / "warehouse.blend"
BLENDER = os.environ.get("DIMOS_WORLDS_BLENDER", "blender")
LAYOUT = json.loads((HERE / "layout.json").read_text())


def blender(*args) -> None:
    subprocess.run([BLENDER, "-b", "--factory-startup", "--disable-autoexec", *args], check=True)


def newest_source() -> float:
    return max(p.stat().st_mtime for p in [*(HERE / "render").glob("*.py"), HERE / "layout.json"])


def ensure_blend(state: Path | None = None) -> Path:
    if state is None:
        blend, npz, extra = BLEND, HERE / "assets" / "cache" / "robots.npz", []
    else:
        blend = BLEND.with_name(f"warehouse-{state.stem}.blend")
        npz = BLEND.with_name(f"robots-{state.stem}.npz")
        extra = ["--state", str(state), "--robots", str(npz)]
    if blend.exists() and blend.stat().st_mtime > max(newest_source(), state.stat().st_mtime if state else 0):
        return blend
    if state is not None:
        tip = json.loads(state.read_text()).get("armTip")
        subprocess.run([sys.executable, str(HERE / "render" / "robots_fk.py"), "--out", str(npz),
                        *(["--iiwa-tip", *map(str, tip)] if tip else [])], check=True)
    elif not npz.exists():
        subprocess.run([sys.executable, str(HERE / "render" / "robots_fk.py")], check=True)
    blender("-P", str(HERE / "render" / "build_scene.py"), "--", "--out", str(blend), *extra)
    return blend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lights", default="day,night")
    ap.add_argument("--cams", default="cam1,cam2,cam3,cam4")
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--exposure-day", type=float, default=1.3)
    ap.add_argument("--exposure-night", type=float, default=2.2)
    ap.add_argument("--captured-at", default="2026-10-08T14:20:00-07:00")
    ap.add_argument("--state", default=None, help="pose totes and robots as in this recorded moment (JSON)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    blend = ensure_blend(Path(a.state).resolve() if a.state else None)
    keep = os.environ.get("DIMOS_WORLDS_KEEP_NPY")
    with tempfile.TemporaryDirectory() as td:
        blender(str(blend), "-P", str(HERE / "render" / "render_frames.py"), "--", "--out", td, "--lights", a.lights,
                "--cams", a.cams, "--samples", str(a.samples))
        manifest = []
        for light in a.lights.split(","):
            for cam in a.cams.split(","):
                src = np.load(Path(td) / f"{light}__{cam}.npy").astype(np.float32)
                lens = LAYOUT["cameras"][cam]["lens_mm"]
                exp = a.exposure_day if light == "day" else a.exposure_night
                rgb = CM.isp(src, lens, light, seed_parts=(cam, light, a.captured_at), exposure=exp)
                path = out / f"{light}_{cam}.jpg"
                path.write_bytes(CM.to_jpeg(rgb, 80))
                manifest.append({"light": light, "camera": cam, "lens_mm": lens, "path": path.name,
                                 "captured_at": a.captured_at})
                if keep:
                    np.save(out / f"{light}_{cam}.npy", src.astype(np.float16))
    (out / "manifest.json").write_text(json.dumps({"frames": manifest}, indent=1) + "\n")
    print("RENDER_OK", len(manifest))


if __name__ == "__main__":
    main()
