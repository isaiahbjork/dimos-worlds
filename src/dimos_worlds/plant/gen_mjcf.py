"""layout.json -> plant.xml: the MuJoCo world of the Demo plant (collision model, fixed cameras, thermal ids).

    python3 gen_mjcf.py          # stdlib only; rewrites scene/plant.xml

Written so DimOS's Go2 loader takes it as is (go2_scene.py): angles in radians, no <option>/<default>, friction on
every geom, no joints. Every geom is named after its layout solid (pipes: "<pipe id>#<segment>"), which is how a
segmentation render finds instruments.
Overhead geometry (pipe-rack beams and pipes above 3 m, roofs) is visual only (contype 0).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import quoteattr

HERE = Path(__file__).resolve().parent
LAYOUT = HERE / "layout.json"
OUT = HERE / "scene" / "plant.xml"
FRICTION = "0.9 0.005 0.0001"

RGBA = {
    "cr_wall": "0.78 0.76 0.70 1", "roof_metal": "0.55 0.56 0.58 1", "panel_grey": "0.72 0.73 0.72 1",
    "desk": "0.45 0.40 0.34 1", "console": "0.15 0.16 0.18 1", "concrete_plinth": "0.62 0.61 0.58 1",
    "steel_dark": "0.22 0.23 0.24 1", "motor_blue": "0.13 0.30 0.55 1", "pump_blue": "0.16 0.36 0.62 1",
    "guard_yellow": "0.90 0.72 0.10 1", "steel_painted": "0.40 0.42 0.45 1", "steel_rack": "0.35 0.36 0.38 1",
    "steel_galv": "0.62 0.64 0.66 1", "grating": "0.40 0.40 0.38 1", "concrete_wall": "0.66 0.65 0.62 1",
    "tank_green": "0.36 0.48 0.38 1", "tank_roof": "0.50 0.52 0.50 1", "comp_grey": "0.48 0.50 0.52 1",
    "receiver_grey": "0.70 0.72 0.72 1", "eyewash_green": "0.15 0.60 0.30 1", "extinguisher_red": "0.75 0.10 0.08 1",
    "pipe_grey": "0.55 0.56 0.57 1", "pipe_green": "0.30 0.50 0.32 1", "pipe_yellow": "0.85 0.70 0.15 1",
    "pipe_blue": "0.20 0.40 0.70 1", "pipe_red": "0.70 0.15 0.12 1",
}


def f(*vals: float) -> str:
    return " ".join(f"{v:.4g}" for v in vals)


def load() -> dict:
    return json.loads(LAYOUT.read_text())


def camera_xyaxes(mount: list[float], target: list[float]) -> str:
    """MuJoCo camera frame: x right, y up, looking along -z."""
    fx, fy, fz = (target[i] - mount[i] for i in range(3))
    n = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / n, fy / n, fz / n
    rx, ry = fy, -fx  # right = forward x up(0,0,1), horizontal
    rn = math.hypot(rx, ry)
    rx, ry = rx / rn, ry / rn
    ux, uy, uz = ry * fz - 0 * fy, 0 * fx - rx * fz, rx * fy - ry * fx  # up = right x forward
    return f(rx, ry, 0.0, ux, uy, uz)


def vfov_deg(hfov_deg: float, aspect: float = 16 / 9) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov_deg) / 2) / aspect))


def build(lay: dict) -> str:
    out: list[str] = []
    w = out.append
    w('<mujoco model="demo-plant">')
    w('  <compiler angle="radian" autolimits="true"/>')
    w("  <visual>")
    w('    <global offwidth="1280" offheight="720" azimuth="90" elevation="-20"/>')
    w('    <headlight ambient="0.35 0.35 0.35" diffuse="0.35 0.35 0.35" specular="0 0 0"/>')
    w('    <quality shadowsize="4096" offsamples="4"/>')
    w("  </visual>")
    w("  <asset>")
    w('    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.62 0.70" rgb2="0.25 0.28 0.32" '
      'width="512" height="512"/>')
    w('    <texture name="slab" type="2d" builtin="checker" rgb1="0.56 0.55 0.52" rgb2="0.54 0.53 0.50" width="256" '
      'height="256" mark="random" markrgb="0.45 0.45 0.43" random="0.04"/>')
    w('    <material name="slab" texture="slab" texrepeat="28 18" texuniform="true" reflectance="0.02"/>')
    w("  </asset>")
    w("  <worldbody>")
    w('    <light name="sun" directional="true" dir="0.3 0.5 -1" diffuse="0.7 0.7 0.68" specular="0 0 0" '
      'castshadow="true"/>')
    s = lay["slab"]
    w(f'    <geom name="ground" type="plane" pos="0 0 0" size="200 200 1" rgba="0.42 0.42 0.40 1" '
      f'friction="{FRICTION}"/>')
    w(f'    <geom name="slab" type="box" pos="{f((s["x0"] + s["x1"]) / 2, (s["y0"] + s["y1"]) / 2, 0.001)}" '
      f'size="{f((s["x1"] - s["x0"]) / 2, (s["y1"] - s["y0"]) / 2, 0.001)}" material="slab" contype="0" '
      'conaffinity="0"/>')
    for wk in lay["walkways"]:  # painted, visual only
        xs = [p[0] for p in wk["poly"]]
        ys = [p[1] for p in wk["poly"]]
        w(f'    <geom name={quoteattr(wk["id"])} type="box" pos="{f((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0.003)}" '
          f'size="{f((max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2, 0.001)}" rgba="0.60 0.60 0.55 1" '
          'contype="0" conaffinity="0"/>')
    for so in lay["solids"]:
        col = "" if so["collide"] else ' contype="0" conaffinity="0"'
        rgba = RGBA.get(so["material"], "0.6 0.6 0.6 1")
        if so["shape"] == "box":
            sx, sy, sz = so["size"]
            w(f'    <geom name={quoteattr(so["id"])} type="box" pos="{f(*so["center"])}" size="{f(sx / 2, sy / 2, sz / 2)}" '
              f'rgba="{rgba}" friction="{FRICTION}"{col}/>')
        else:
            w(f'    <geom name={quoteattr(so["id"])} type="cylinder" fromto="{f(*so["from"], *so["to"])}" '
              f'size="{f(so["radius"])}" rgba="{rgba}" friction="{FRICTION}"{col}/>')
    for p in lay["pipes"]:
        rgba = RGBA.get(p["material"], "0.6 0.6 0.6 1")
        pts = p["points"]
        for k in range(len(pts) - 1):
            a, b = pts[k], pts[k + 1]
            low = min(a[2], b[2]) - p["radius"] < 3.0
            col = "" if (p["collide"] and low) else ' contype="0" conaffinity="0"'
            w(f'    <geom name={quoteattr(p["id"] + "#" + str(k))} type="capsule" fromto="{f(*a, *b)}" '
              f'size="{f(p["radius"])}" rgba="{rgba}" friction="{FRICTION}"{col}/>')
    for ins in lay["instruments"]:  # dial discs / valve indicators: visual only (the thermal and coverage ids)
        if ins["kind"] in ("gauge", "valve"):
            x, y, z = ins["pos"]
            fx, fy = ins["facing"]
            r = ins.get("dial_m", 0.14) / 2 + 0.01
            w(f'    <geom name={quoteattr(ins["tag"])} type="cylinder" fromto="{f(x - fx * 0.03, y - fy * 0.03, z, x + fx * 0.01, y + fy * 0.01, z)}" '
              f'size="{f(r)}" rgba="0.92 0.92 0.90 1" contype="0" conaffinity="0"/>')
        elif ins["kind"] == "drip_tray":
            x, y, _ = ins["pos"]
            w(f'    <geom name={quoteattr(ins["tag"])} type="box" pos="{f(x, y, 0.02)}" size="0.45 0.2 0.02" '
              'rgba="0.30 0.30 0.30 1" contype="0" conaffinity="0"/>')
    for cid, c in lay["cameras"].items():  # pole under each outdoor camera
        mx, my, mz = c["mount"]
        if mz > 4.0:
            w(f'    <geom name="{cid}-pole" type="cylinder" fromto="{f(mx, my, 0, mx, my, mz)}" size="0.08" '
              f'rgba="0.62 0.64 0.66 1" friction="{FRICTION}"/>')
        w(f'    <camera name="{cid}" pos="{f(*c["mount"])}" xyaxes="{camera_xyaxes(c["mount"], c["target"])}" '
          f'fovy="{vfov_deg(c["hfov_deg"]):.2f}"/>')
    w("  </worldbody>")
    w("</mujoco>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    xml = build(load())
    OUT.write_text(xml)
    print(f"wrote {OUT} ({xml.count('<geom')} geoms)")
