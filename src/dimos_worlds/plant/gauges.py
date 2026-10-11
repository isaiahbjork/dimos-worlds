"""Gauge dials and valve indicators: the one drawing both the Blender scene and the 2D fallback use, plus a
classic-CV needle reader.

    draw_face(spec, px)            dial face without the needle (Blender maps it on the dial disc; the needle is 3D)
    draw_gauge(spec, value, px)    face + needle at needle_angle(value): the 2D fallback close-up
    draw_valve_plate(px)           OPEN/SHUT position plate (gate valves); the pointer is 3D in Blender
    draw_valve(spec, state, px)    plate + pointer (fallback)
    read_needle(rgb, cx, cy, r, spec, squash=(sx, sy))  -> (value, angle_deg, contrast)

The reader polar-samples darkness between 0.2 r and 0.5 r (inside the numerals, past the needle's short tail) and
takes the darkest ray. `squash` undoes the foreshortening of a dial seen off-axis (a caller that knows the camera and dial
poses passes cos of the horizontal / vertical view angles). Needs pillow + numpy.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dimos_worlds.plant.plant_state import START_DEG, SWEEP_DEG, needle_angle, value_at

FACE = (246, 245, 240)
INK = (18, 18, 20)
GREEN = (40, 150, 70)
RED = (200, 40, 30)
NEEDLE = (25, 25, 28)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: bitmap font only
        return ImageFont.load_default()


def _pt(cx: float, cy: float, r: float, ang_deg: float) -> tuple[float, float]:
    a = math.radians(ang_deg)
    return cx + r * math.cos(a), cy - r * math.sin(a)  # image y grows down


def _ticks(spec: dict) -> tuple[float, float]:
    lo, hi = spec["range"]
    span = hi - lo
    for major in (1, 2, 5, 10, 20, 25, 50, 100):
        if span / major <= 12:
            return float(major), float(major) / 5.0
    return span / 10.0, span / 50.0


def _arc(draw: ImageDraw.ImageDraw, cx, cy, r, a0, a1, width, color) -> None:
    """Arc between maths angles a0 > a1 (clockwise on the dial)."""
    steps = max(4, int(abs(a0 - a1) / 1.5))
    pts = [_pt(cx, cy, r, a0 + (a1 - a0) * k / steps) for k in range(steps + 1)]
    draw.line(pts, fill=color, width=width, joint="curve")


def draw_face(spec: dict, px: int = 512) -> Image.Image:
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = px / 2
    R = px * 0.49
    d.ellipse([c - R, c - R, c + R, c + R], fill=(40, 40, 44, 255))  # bezel
    Rf = px * 0.455
    d.ellipse([c - Rf, c - Rf, c + Rf, c + Rf], fill=FACE + (255,))
    lo, hi = spec["range"]
    nlo, nhi = spec["normal"]
    band_r = Rf * 0.80
    bw = max(3, int(px * 0.035))
    _arc(d, c, c, band_r, needle_angle(spec, lo), needle_angle(spec, nlo), bw, RED)
    _arc(d, c, c, band_r, needle_angle(spec, nlo), needle_angle(spec, nhi), bw, GREEN)
    _arc(d, c, c, band_r, needle_angle(spec, nhi), needle_angle(spec, hi), bw, RED)
    major, minor = _ticks(spec)
    v = lo
    fnt = _font(max(10, int(px * 0.075)))
    while v <= hi + 1e-6:
        a = needle_angle(spec, v)
        is_major = abs((v - lo) / major - round((v - lo) / major)) < 1e-6
        r0 = Rf * (0.84 if is_major else 0.88)
        d.line([_pt(c, c, r0, a), _pt(c, c, Rf * 0.96, a)], fill=INK, width=max(2, int(px * (0.012 if is_major else 0.005))))
        if is_major:
            tx, ty = _pt(c, c, Rf * 0.67, a)
            label = f"{v:g}"
            d.text((tx, ty), label, fill=INK, font=fnt, anchor="mm")
        v = round(v + minor, 6)
    small = _font(max(9, int(px * 0.06)))
    d.text((c, c + Rf * 0.34), spec["unit"], fill=INK, font=small, anchor="mm")  # both in the dead zone at the bottom
    d.text((c, c + Rf * 0.62), spec["tag"], fill=INK, font=small, anchor="mm")
    return img


def _needle(d: ImageDraw.ImageDraw, c: float, Rf: float, ang: float, px: int) -> None:
    tip = _pt(c, c, Rf * 0.86, ang)
    tail = _pt(c, c, Rf * 0.14, ang + 180)
    w = max(3, int(px * 0.018))
    d.line([tail, tip], fill=NEEDLE, width=w)
    d.line([_pt(c, c, Rf * 0.70, ang), tip], fill=RED, width=w)
    hub = px * 0.035
    d.ellipse([c - hub, c - hub, c + hub, c + hub], fill=NEEDLE)


def draw_gauge(spec: dict, value: float, px: int = 512) -> Image.Image:
    img = draw_face(spec, px)
    _needle(ImageDraw.Draw(img), px / 2, px * 0.455, needle_angle(spec, value), px)
    return img


PLATE_OPEN_DEG = 135.0  # pointer angle (maths) at OPEN / SHUT on the position plate
PLATE_SHUT_DEG = 45.0


def draw_valve_plate(px: int = 256) -> Image.Image:
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = px / 2
    R = px * 0.48
    d.ellipse([c - R, c - R, c + R, c + R], fill=(235, 232, 220, 255), outline=INK, width=max(2, px // 64))
    d.pieslice([c - R * 0.92, c - R * 0.92, c + R * 0.92, c + R * 0.92], 180, 270, fill=GREEN)  # upper-left: OPEN
    d.pieslice([c - R * 0.92, c - R * 0.92, c + R * 0.92, c + R * 0.92], 270, 360, fill=RED)  # upper-right: SHUT
    # labels out near the rim, past the tip of the (short) pointer, so the pointer never covers the word it points at
    f = _font(max(10, int(px * 0.11)))
    d.text(_pt(c, c, R * 0.70, PLATE_OPEN_DEG), "OPEN", fill=(255, 255, 255), font=f, anchor="mm")
    d.text(_pt(c, c, R * 0.70, PLATE_SHUT_DEG), "SHUT", fill=(255, 255, 255), font=f, anchor="mm")
    return img


def draw_valve(spec: dict, state: str, px: int = 256) -> Image.Image:
    if spec.get("style") == "ball":  # lever: along the pipe (vertical in this view) = open, across = closed
        img = Image.new("RGBA", (px, px), (120, 122, 120, 255))
        d = ImageDraw.Draw(img)
        c = px / 2
        d.rectangle([c - px * 0.08, 0, c + px * 0.08, px], fill=(150, 152, 155))  # pipe, running away from camera
        d.ellipse([c - px * 0.16, c - px * 0.16, c + px * 0.16, c + px * 0.16], fill=(170, 160, 120))
        if state == "open":
            d.rectangle([c - px * 0.04, c - px * 0.42, c + px * 0.04, c], fill=(230, 190, 20))
        else:
            d.rectangle([c, c - px * 0.04, c + px * 0.42, c + px * 0.04], fill=(230, 190, 20))
        return img
    img = draw_valve_plate(px)
    d = ImageDraw.Draw(img)
    c = px / 2
    ang = PLATE_OPEN_DEG if state == "open" else PLATE_SHUT_DEG
    d.line([(c, c), _pt(c, c, px * 0.24, ang)], fill=INK, width=max(3, px // 28))
    return img


def read_needle(rgb: np.ndarray, cx: float, cy: float, r: float, spec: dict,
                squash: tuple[float, float] = (1.0, 1.0)) -> tuple[float, float, float]:
    """(value, angle_deg, contrast). rgb: H x W x 3 uint8. (cx, cy, r): dial face centre and radius in pixels, as seen
    head-on; squash: (sx, sy) scale of the dial's x / y axes in the image (cos of the view angles)."""
    gray = rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    h, w = gray.shape
    sx, sy = squash
    best_a, best_v, vals = 0.0, -1e9, []
    radii = np.linspace(0.2, 0.5, 24) * r
    for a10 in range(0, 3600, 5):
        a = a10 / 10.0
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        xs = np.clip(np.round(cx + radii * ca * sx).astype(int), 0, w - 1)
        ys = np.clip(np.round(cy - radii * sa * sy).astype(int), 0, h - 1)
        dark = 255.0 - float(gray[ys, xs].mean())
        vals.append(dark)
        if dark > best_v:
            best_v, best_a = dark, a
    med = float(np.median(vals))
    contrast = best_v - med
    # the needle is several rays wide: take the darkness-weighted centre of the peak, not its first ray
    arr = np.asarray(vals)
    angs = np.arange(len(arr)) * 0.5
    near = np.abs((angs - best_a + 180.0) % 360.0 - 180.0) <= 8.0
    wts = np.where(near, np.clip(arr - (med + 0.5 * contrast), 0, None), 0.0)
    if wts.sum() > 0:
        off = ((angs - best_a + 180.0) % 360.0) - 180.0
        best_a = (best_a + float((off * wts).sum() / wts.sum())) % 360.0
    return round(value_at(spec, best_a), 3), round(best_a, 2), round(contrast, 1)


def sweep_check() -> list[tuple[float, float]]:
    """Self-test: draw each gauge at several values and read it back (errors in units)."""
    from plant_state import instruments

    errs = []
    for spec in instruments().values():
        if spec["kind"] != "gauge":
            continue
        lo, hi = spec["range"]
        for k in range(11):
            v = lo + (hi - lo) * k / 10
            img = np.asarray(draw_gauge(spec, v, 256).convert("RGB"))
            got, _a, _c = read_needle(img, 128, 128, 256 * 0.455, spec)
            errs.append((v, got))
    return errs


__all__ = ["draw_face", "draw_gauge", "draw_valve", "draw_valve_plate", "read_needle", "START_DEG", "SWEEP_DEG"]
