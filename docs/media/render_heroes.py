"""Regenerate the README's Blender stills (Cycles; a GPU makes it practical).

    python src/dimos_worlds/lot/assets/fetch.py && python src/dimos_worlds/warehouse/assets/fetch.py
    DIMOS_WORLDS_BLENDER=/path/to/blender python docs/media/render_heroes.py [--samples 384] [--only NAME]

Builds what is missing first: warehouse/assets/cache/robots.npz (render/robots_fk.py: Menagerie G1, Go2, iiwa posed
by forward kinematics), warehouse.blend and lot.blend (render/build_scene.py). Then renders each shot with
hero_blender.py and writes docs/media/<name>.jpg (JPEG q86, under 1.5 MB). Robots are placed, not simulated: these
are stills of the scenes, not frames of a run. Verified with Blender 5.2.2 on Ubuntu 22.04 + RTX 3090 (OptiX).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image

OUT = Path(__file__).resolve().parent
PKG = OUT.parents[1] / "src" / "dimos_worlds"
BLENDER = os.environ.get("DIMOS_WORLDS_BLENDER", "blender")
WH_CACHE = PKG / "warehouse" / "assets" / "cache"
LOT_CACHE = PKG / "lot" / "assets" / "cache"
MAX_BYTES = 1_500_000

# name: (scene, light, robot placements "robot:x,y,yaw_deg", camera xyz, look-at xyz, horizontal fov deg, exposure)
SHOTS = {
    "lot-night-go2": ("lot", "night", ["go2:-8,-8.2,25"], "-3.5,-7.0,0.9", "-12,-8.6,0.5", 55, 0.0),
    "lot-night-rows": ("lot", "night", ["go2:-8,-8.2,-10"], "-14,-6.5,1.6", "-4,-8.5,0.4", 60, 0.3),
    "warehouse-go2-g1": ("warehouse", "day", ["g1:6.0,11.0,0", "go2:9.2,11.7,-20"], "13.5,11.0,1.15",
                         "6,11.4,0.75", 50, 0.0),
}


def run(*cmd: str, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def ensure_built() -> None:
    for cache, what in ((LOT_CACHE, "lot"), (WH_CACHE, "warehouse")):
        if not (cache / "kloppenheim_02_puresky").is_dir():
            sys.exit(f"no {what} asset cache: run python src/dimos_worlds/{what}/assets/fetch.py first")
    if not (WH_CACHE / "menagerie_unitree_go2" / ".done").is_file():
        sys.exit("no Menagerie Go2 in the warehouse cache: rerun python src/dimos_worlds/warehouse/assets/fetch.py")
    npz = WH_CACHE / "robots.npz"
    if not npz.exists():
        run(sys.executable, str(PKG / "warehouse" / "render" / "robots_fk.py"))
    for scene, cache in (("warehouse", WH_CACHE), ("lot", LOT_CACHE)):
        blend = cache / f"{scene}.blend"
        if not blend.exists():
            run(BLENDER, "-b", "--factory-startup", "--disable-autoexec", "-P",
                str(PKG / scene / "render" / "build_scene.py"), "--", "--out", str(blend), cwd=PKG / scene)


def shot(name: str, samples: int, size: str) -> Path:
    scene, light, places, cam, look, hfov, exposure = SHOTS[name]
    blend = (LOT_CACHE if scene == "lot" else WH_CACHE) / f"{scene}.blend"
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / f"{name}.png"
        cmd = [BLENDER, "-b", str(blend), "--disable-autoexec", "-P", str(OUT / "hero_blender.py"), "--",
               "--scene", scene, "--robots", str(WH_CACHE / "robots.npz"), "--cam", cam, "--look", look,
               "--hfov", str(hfov), "--light", light, "--samples", str(samples), "--exposure", str(exposure),
               "--size", size, "--out", str(png)]
        for p in places:
            cmd += ["--place", p]
        run(*cmd)
        img = Image.open(png).convert("RGB")
        path = OUT / f"{name}.jpg"
        for q in (86, 80, 74, 68):
            img.save(path, quality=q, optimize=True, progressive=True)
            if path.stat().st_size < MAX_BYTES:
                break
    print(f"WROTE {path} {path.stat().st_size // 1024} KiB", flush=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=384)
    ap.add_argument("--size", default="1600x900")
    ap.add_argument("--only", choices=sorted(SHOTS), default=None)
    a = ap.parse_args()
    ensure_built()
    for name in [a.only] if a.only else SHOTS:
        shot(name, a.samples, a.size)


if __name__ == "__main__":
    main()
