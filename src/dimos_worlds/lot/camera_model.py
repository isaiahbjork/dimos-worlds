"""Fixed CCTV camera model: sensor, lens (with barrel distortion), noise, ISP and JPEG.

The renderer produces a linear, pinhole ("source") image a bit wider than the camera sees. This module
turns it into what a typical fixed IP camera would send: barrel-distorted, vignetted, slightly soft,
with blooming around lamps, signal-dependent sensor noise smeared by the camera's noise reduction,
an sRGB-like tone curve with lifted blacks, edge sharpening, and JPEG compression.

Typical specs modelled (common for 2 MP bullet/turret cameras; UNVERIFIED for any specific product):
  * sensor 1/2.8" CMOS, 1920x1080 at about 2.9 um pitch: active area about 5.57 x 3.13 mm
  * lens 2.8 mm: about 100 deg horizontal field of view; 4 mm: about 80 deg (datasheet-style figures,
    which include the lens's barrel distortion, so the pinhole field of view is narrower)
  * output here is the 1280x720 sub-stream snapshot, JPEG

Pure numpy (+ Pillow for JPEG). Blender imports only the geometry helpers (source_fov), so keep
Pillow imports inside functions.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

OUT_W, OUT_H = 1280, 720
SRC_W, SRC_H = 1664, 936  # pinhole render size before distortion (covers the distorted corners)
SENSOR_W_MM, SENSOR_H_MM = 5.57, 3.13

# lens focal length (mm) -> radial distortion k1 (r_u = r_d * (1 + k1 r_d^2), normalised by focal length).
# k1 is chosen so the horizontal field of view matches the typical datasheet figure.
LENSES = {2.8: {"k1": 0.20, "hfov_deg": 100.0}, 4.0: {"k1": 0.42, "hfov_deg": 80.0}}
SRC_MARGIN = 1.03


def _xd_max(lens_mm: float) -> tuple[float, float]:
    """Half-extent of the output image in normalised distorted coordinates (x, y)."""
    return (SENSOR_W_MM / 2) / lens_mm, (SENSOR_H_MM / 2) / lens_mm


def _undistort_r(rd, k1):
    return rd * (1.0 + k1 * rd * rd)


def _distort_r(ru, k1, iters: int = 12):
    """Invert r_u = r_d (1 + k1 r_d^2) by Newton's method (monotonic for k1 > 0)."""
    rd = np.asarray(ru, dtype=np.float64).copy()
    for _ in range(iters):
        f = rd * (1 + k1 * rd * rd) - ru
        fp = 1 + 3 * k1 * rd * rd
        rd = rd - f / fp
    return rd


def source_extent(lens_mm: float) -> tuple[float, float]:
    """Half-extent (x, y) of the pinhole source image in normalised undistorted coordinates."""
    k1 = LENSES[lens_mm]["k1"]
    xm, ym = _xd_max(lens_mm)
    s = _undistort_r(math.hypot(xm, ym), k1) / math.hypot(xm, ym)  # corner scale factor
    x_u = xm * s * SRC_MARGIN
    return x_u, x_u * SRC_H / SRC_W


def source_fov(lens_mm: float) -> tuple[float, float]:
    """(horizontal, vertical) field of view of the pinhole source render, radians."""
    xu, yu = source_extent(lens_mm)
    return 2 * math.atan(xu), 2 * math.atan(yu)


def effective_hfov_deg(lens_mm: float) -> float:
    k1 = LENSES[lens_mm]["k1"]
    xm, _ = _xd_max(lens_mm)
    return 2 * math.degrees(math.atan(_undistort_r(xm, k1)))


def project(lens_mm: float, cam_space_xyz) -> tuple[float, float] | None:
    """Output pixel (u, v) of a point in camera space (x right, y up, -z forward). None if behind."""
    x, y, z = cam_space_xyz
    if -z <= 1e-6:
        return None
    xu, yu = x / -z, y / -z
    ru = math.hypot(xu, yu)
    k1 = LENSES[lens_mm]["k1"]
    rd = float(_distort_r(ru, k1)) if ru > 0 else 0.0
    s = rd / ru if ru > 0 else 1.0
    xd, yd = xu * s, yu * s
    xm, ym = _xd_max(lens_mm)
    return OUT_W / 2 + xd / xm * (OUT_W / 2), OUT_H / 2 - yd / ym * (OUT_H / 2)


# ---- image pipeline -------------------------------------------------------------------------


def _bilinear(img: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    sx = np.clip(sx, 0, w - 1.001)
    sy = np.clip(sy, 0, h - 1.001)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    fx = (sx - x0)[..., None]
    fy = (sy - y0)[..., None]
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def distort(src: np.ndarray, lens_mm: float) -> np.ndarray:
    """Remap a pinhole source image (SRC_H x SRC_W, linear) to the distorted 1280x720 camera image."""
    k1 = LENSES[lens_mm]["k1"]
    xm, ym = _xd_max(lens_mm)
    xu_max, yu_max = source_extent(lens_mm)
    u = (np.arange(OUT_W) + 0.5) / OUT_W * 2 - 1
    v = (np.arange(OUT_H) + 0.5) / OUT_H * 2 - 1
    xd, yd = np.meshgrid(u * xm, -v * ym)
    rd2 = xd * xd + yd * yd
    f = 1 + k1 * rd2
    xu, yu = xd * f, yd * f
    sh, sw = src.shape[:2]
    sx = (xu / xu_max + 1) / 2 * sw - 0.5
    sy = (1 - yu / yu_max) / 2 * sh - 0.5
    return _bilinear(src, sx, sy).astype(np.float32)


def _box_blur(img: np.ndarray, r: int, axis: int) -> np.ndarray:
    if r <= 0:
        return img
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r + 1, r)
    p = np.pad(img, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    n = img.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-ish blur: three box passes per axis."""
    if sigma <= 0.25:
        return img
    r = max(1, int(round(math.sqrt(12 * sigma * sigma / 3 + 1) / 2)))
    out = img
    for axis in (0, 1):
        for _ in range(3):
            out = _box_blur(out, r, axis)
    return out


def _downblur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Wide blur via a 4x downsample (for bloom)."""
    h, w = img.shape[:2]
    small = img[: h // 4 * 4, : w // 4 * 4].reshape(h // 4, 4, w // 4, 4, -1).mean(axis=(1, 3))
    small = blur(small, sigma / 4)
    sy = (np.arange(h) + 0.5) / 4 - 0.5
    sx = (np.arange(w) + 0.5) / 4 - 0.5
    gx, gy = np.meshgrid(sx, sy)
    return _bilinear(small, gx, gy).astype(np.float32)


# Per-mode exposure and noise. gain maps the renderer's linear radiance to sensor linear (0..1 = white).
MODES = {
    "night": {"gain": 0.9, "shot": 0.002, "read": 0.0025, "chroma": 0.5, "sat": 0.78, "black": 0.022,
              "dnr_sigma": 0.9},
    "day": {"gain": 0.55, "shot": 0.0006, "read": 0.0012, "chroma": 0.25, "sat": 1.0, "black": 0.01,
            "dnr_sigma": 0.6},
}


def _seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


def isp(src_linear: np.ndarray, lens_mm: float | None, mode: str, seed_parts=(), exposure: float = 1.0) -> np.ndarray:
    """Linear source render -> 8-bit RGB as the camera would output it (before JPEG).

    lens_mm from LENSES: barrel distortion to 1280x720. None: a rectilinear camera (the drone's, whose
    distortion is corrected on board); output keeps the source size."""
    cfg = MODES[mode]
    src = src_linear[..., :3].astype(np.float32)
    img = (distort(src, lens_mm) if lens_mm is not None else src) * (cfg["gain"] * exposure)
    # optics: vignetting, slight softness, bloom/flare around bright sources
    h, w = img.shape[:2]
    u = np.linspace(-1, 1, w)[None, :]
    v = np.linspace(-1, 1, h)[:, None] * (h / w)
    r2 = (u * u + v * v) / (1 + (h / w) ** 2)
    img *= (1 - (0.28 if lens_mm is not None else 0.15) * r2)[..., None]
    bright = np.clip(img - 0.9, 0, 40)
    img = blur(img, 0.6) + _downblur(bright, 6.0) * 0.18 + _downblur(bright, 22.0) * 0.06
    # sensor noise: Poisson-Gaussian in linear, spatially correlated by demosaic + 3D-DNR
    rng = np.random.default_rng(_seed(*seed_parts))
    lin = np.clip(img, 0, None)
    sigma = np.sqrt(cfg["shot"] * lin + cfg["read"] ** 2)
    luma_n = rng.standard_normal(img.shape[:2]).astype(np.float32)
    chroma_n = rng.standard_normal(img.shape).astype(np.float32)
    luma_n = blur(luma_n[..., None], cfg["dnr_sigma"])[..., 0] * 1.9
    chroma_n = blur(chroma_n, 2.0) * 3.0 * cfg["chroma"]
    img = img + sigma * (luma_n[..., None] + chroma_n)
    # tone: soft shoulder for lamps, sRGB-like curve, lifted blacks, reduced night saturation
    img = np.clip(img, 0, None)
    img = img * (1 + img / 16.0) / (1 + img)
    img = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(np.maximum(img, 1e-8), 1 / 2.4) - 0.055)
    luma = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    img = luma[..., None] + (img - luma[..., None]) * cfg["sat"]
    img = cfg["black"] + (1 - cfg["black"]) * img
    # ISP sharpening on luma (gives the slight halo of IP cameras)
    luma = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    detail = luma - blur(luma[..., None], 1.3)[..., 0]
    img = img + 0.55 * detail[..., None]
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def to_jpeg(rgb8: np.ndarray, quality: int) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb8).save(buf, "JPEG", quality=quality, subsampling="4:2:0")
    return buf.getvalue()
