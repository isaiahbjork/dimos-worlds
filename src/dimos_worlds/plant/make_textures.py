"""Textures drawn here (not downloaded): every gauge's dial face, the valve position plate, equipment tag plates.

    python make_textures.py        # needs pillow + numpy; writes assets/cache/gen/*.png (untracked)

Blender maps them (render/build_scene.py); the needles and pointers are 3D objects posed from plant_state.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from dimos_worlds.plant import gauges, plant_state

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "cache" / "gen"


def tag_plate(text: str, px_w: int = 512, px_h: int = 160, bg=(240, 200, 30), fg=(15, 15, 15)) -> Image.Image:
    img = Image.new("RGB", (px_w, px_h), bg)
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.load_default(size=int(px_h * 0.6))
    except TypeError:
        f = ImageFont.load_default()
    d.rectangle([4, 4, px_w - 5, px_h - 5], outline=fg, width=6)
    d.text((px_w / 2, px_h / 2), text, fill=fg, font=f, anchor="mm")
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for tag, spec in plant_state.instruments().items():
        if spec["kind"] == "gauge":
            gauges.draw_face(spec, 1024).save(OUT / f"dial_{tag}.png")
            n += 1
        if spec["kind"] in ("gauge", "valve", "motor"):
            tag_plate(tag).save(OUT / f"plate_{tag}.png")
    gauges.draw_valve_plate(512).save(OUT / "valve_plate.png")
    for name in ("T-201", "T-202", "P-101", "P-102", "P-103", "C-301", "CONTROL ROOM", "NO STEP - STAIRS"):
        bg = (200, 40, 30) if "STAIRS" in name else (240, 240, 235)
        fg = (255, 255, 255) if "STAIRS" in name else (20, 20, 20)
        tag_plate(name, 1024 if len(name) > 8 else 512, 160, bg=bg, fg=fg).save(OUT / f"sign_{name.replace(' ', '_')}.png")
    print(f"TEXTURES_OK {n} dials -> {OUT}")


if __name__ == "__main__":
    main()
