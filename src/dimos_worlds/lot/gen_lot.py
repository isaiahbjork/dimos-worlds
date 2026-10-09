"""Generate the lot scene package: scene/lot.xml (collision-simplified MuJoCo model) and scene/scene.meta.json.

    python -m dimos_worlds.lot.gen_lot          # rewrites scene/lot.xml and scene/scene.meta.json

Same layout as the optional Blender render scene (render/build_scene.py), simplified for physics:
  * vehicles: a lower-body box, a cabin box and four wheel cylinders at real dimensions
  * fence: one thin box per straight run (2.44 m fabric + 0.35 m barbed-wire arms are treated as 2.8 m),
    gaps at the two gates
  * buildings, fuel tank, dumpsters, poles, bollards and wheel stops as boxes and cylinders
  * the person is a mocap body of capsules at the standing pose (crouch is rendered in Blender only)

The model has no joints at all. Movable things (the two gates, the 23 row vehicles, the person) are mocap bodies:
events.py moves them by editing mocap_pos/mocap_quat, so a robot loaded next to the lot keeps its own qpos layout
(DimOS's Go2/G1 policies read qpos[7:] as the robot's joints). Mocap bodies still collide.

The XML is written so it can be both a standalone scene and the scene half of DimOS's Go2 sim:
  * angles in radians (DimOS's robot files have no <compiler>)
  * no <option> and no <default>: they would change the robot model loaded next to it; friction is written
    on every geom instead
Cameras carry the pinhole field of view of the render source (camera_model.source_fov). lot.xml is generated;
do not hand-edit it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from dimos_worlds.lot import camera_model as CM
from dimos_worlds.lot import layout as L

SCENE_DIR = Path(__file__).resolve().parent / "scene"
OUT = SCENE_DIR / "lot.xml"
META = SCENE_DIR / "scene.meta.json"
PLACES_JSON = SCENE_DIR / "places.json"
FRICTION = "0.9 0.005 0.0001"

ASPHALT = (0.1, 0.1, 0.11, 1)
CONCRETE = (0.36, 0.36, 0.35, 1)
STEEL = (0.5, 0.52, 0.55, 1)
MESH = (0.55, 0.57, 0.6, 0.25)
STUCCO = (0.55, 0.5, 0.42, 1)
METAL_BLDG = (0.52, 0.54, 0.56, 1)
GLASS = (0.06, 0.07, 0.09, 1)
TIRE = (0.03, 0.03, 0.03, 1)
YELLOW = (0.65, 0.45, 0.02, 1)
PERSON = (0.3, 0.3, 0.33, 1)


def f(values) -> str:
    return " ".join(f"{float(v):.4g}" for v in values)


def yaw_quat(yaw_deg: float) -> str:
    h = math.radians(yaw_deg) / 2
    return f((math.cos(h), 0.0, 0.0, math.sin(h)))


def box(pos, half, rgba, name: str | None = None, mass: float | None = None, quat: str | None = None) -> str:
    n = f' name="{name}"' if name else ""
    m = f' mass="{mass:.4g}"' if mass else ""
    q = f' quat="{quat}"' if quat else ""
    return f'<geom{n} type="box" pos="{f(pos)}" size="{f(half)}" rgba="{f(rgba)}"{m}{q} friction="{FRICTION}"/>'


def cyl(a, b, r, rgba, mass: float | None = None) -> str:
    m = f' mass="{mass:.4g}"' if mass else ""
    return f'<geom type="cylinder" fromto="{f(a)} {f(b)}" size="{r:.4g}" rgba="{f(rgba)}"{m} friction="{FRICTION}"/>'


def capsule(a, b, r, rgba) -> str:
    return f'<geom type="capsule" fromto="{f(a)} {f(b)}" size="{r:.4g}" rgba="{f(rgba)}" friction="{FRICTION}"/>'


# ---- vehicles -------------------------------------------------------------------------------
# (lower body z0, z1, cabin y0, y1, cabin z1 as a fraction of H, wheel radius, wheelbase, front overhang)
VEH_SHAPE = {
    "sedan": (0.24, 0.98, -1.4, 0.75, 1.0, 0.34, 2.83, 0.95),
    "hatch": (0.25, 1.0, -2.0, 0.8, 1.0, 0.33, 2.70, 0.88),
    "suv": (0.36, 1.12, -2.3, 1.05, 1.0, 0.39, 2.95, 0.95),
    "suv_compact": (0.32, 1.05, -2.05, 0.9, 1.0, 0.36, 2.68, 0.92),
    "pickup": (0.48, 1.26, -0.55, 1.2, 1.0, 0.40, 3.68, 0.95),
    "minivan": (0.28, 1.02, -2.45, 1.55, 1.0, 0.36, 3.03, 0.98),
    "box_truck": (0.75, 3.4, 0.0, 0.0, 0.0, 0.42, 4.0, 1.05),
    "skid_steer": (0.3, 1.2, -0.85, 0.35, 1.0, 0.39, 0.95, 1.4),
    "trailer": (0.5, 0.62, -1.4, -1.4, 0.0, 0.34, 0.64, 2.2),
    "covered_car": (0.2, 0.95, -1.3, 0.6, 1.0, 0.3, 2.6, 0.9),
}
VEH_MASS = {"sedan": 1600, "hatch": 1400, "suv": 2200, "suv_compact": 1700, "pickup": 2400, "minivan": 2000,
            "box_truck": 6500, "skid_steer": 3200, "trailer": 700, "covered_car": 1500}


def vehicle_geoms(kind: str, paint) -> list[str]:
    length, width, height = L.VEHICLE_DIMS[kind]
    z0, z1, cy0, cy1, cz, r, wheelbase, f_over = VEH_SHAPE[kind]
    rgba = (*paint, 1)
    mass = VEH_MASS[kind]
    g = [box((0, 0, (z0 + z1) / 2), (width / 2, length / 2, (z1 - z0) / 2), rgba, mass=mass * 0.8)]
    if cz > 0 and cy1 > cy0 and height * cz > z1 + 0.02:
        g.append(box((0, (cy0 + cy1) / 2, (z1 + height * cz) / 2), (width / 2 - 0.08, (cy1 - cy0) / 2, (height * cz - z1) / 2),
                     GLASS, mass=mass * 0.1))
    yf = length / 2 - f_over
    for wy in (yf, yf - wheelbase):
        for side in (-1, 1):
            x = side * (width / 2 - 0.12)
            g.append(cyl((x - 0.11, wy, r), (x + 0.11, wy, r), r, TIRE, mass=mass * 0.025))
    return g


def vehicle_body(v: dict) -> str:
    inner = "\n      ".join(vehicle_geoms(v["kind"], L.paint_rgb(v["paint"])))
    mocap = ' mocap="true"' if v["movable"] else ""
    return (
        f'    <body name="{v["name"]}"{mocap} pos="{f((v["x"], v["y"], 0.0))}" quat="{yaw_quat(v["yaw"])}">\n'
        f"      {inner}\n"
        f"    </body>"
    )


# ---- fence, gates, structures -----------------------------------------------------------------

FENCE_COLLIDE_H = L.FENCE_H + L.BARB_H


def fence() -> list[str]:
    out = []
    sides = {
        "north": ((-L.HX, L.HY), (1, 0), 2 * L.HX),
        "east": ((L.HX, -L.HY), (0, 1), 2 * L.HY),
        "south": ((-L.HX, -L.HY), (1, 0), 2 * L.HX),
        "west": ((-L.HX, L.HY), (0, -1), 2 * L.HY),
    }
    for side, (start, d, length) in sides.items():
        cuts = [0.0]
        for g0, g1 in L.FENCE_GAPS.get(side, []):
            cuts += [g0, g1]
        cuts.append(length)
        for s0, s1 in zip(cuts[::2], cuts[1::2]):
            mid = (s0 + s1) / 2
            cx, cy = start[0] + d[0] * mid, start[1] + d[1] * mid
            half_len = (s1 - s0) / 2
            half = (abs(d[0]) * half_len + 0.025, abs(d[1]) * half_len + 0.025, FENCE_COLLIDE_H / 2)
            out.append(box((cx, cy, FENCE_COLLIDE_H / 2), half, MESH))
            n_posts = max(1, round((s1 - s0) / 3.0))
            for k in range(n_posts + 1):
                s = s0 + (s1 - s0) * k / n_posts
                px, py = start[0] + d[0] * s, start[1] + d[1] * s
                out.append(cyl((px, py, 0), (px, py, L.FENCE_H + 0.06), 0.03, STEEL))
    return out


def gates() -> list[str]:
    out = []
    h = L.FENCE_H - 0.1
    for gid, g in L.GATES.items():
        ox, oy = g["origin"]
        ax, ay = g["axis"]
        length = g["length"]
        tail = 2.4 if g["type"] == "slide" else 0.0
        s0, s1 = -tail, length
        c = ((s0 + s1) / 2 * ax, (s0 + s1) / 2 * ay, 0.08 + h / 2)
        half = (abs(ax) * (s1 - s0) / 2 + 0.03, abs(ay) * (s1 - s0) / 2 + 0.03, h / 2)
        out.append(
            f'    <body name="{gid}" mocap="true" pos="{f((ox, oy, 0.0))}">\n'
            f"      {box(c, half, (0.5, 0.52, 0.55, 0.6), mass=150)}\n"
            f"    </body>"
        )
    for x in (-10.0, -4.0):
        out.append("    " + cyl((x, -25.0, 0), (x, -25.0, 2.6), 0.05, YELLOW))
    for y in (-14.0, -8.0):
        out.append("    " + cyl((-40.0, y, 0), (-40.0, y, 2.6), 0.05, YELLOW))
    return out


def structures() -> list[str]:
    g = []
    o = L.OFFICE
    (cx, cy), (hx, hy), h = o["center"], o["half"], o["height"] + o["parapet"]
    g.append(box((cx, cy, h / 2), (hx, hy, h / 2), STUCCO, name="office"))
    for wx, ww in o["windows"]:
        lit = o["windows"].index((wx, ww)) in o["lit_windows"]
        g.append(box((wx, cy + hy + 0.015, 1.65), (ww / 2, 0.015, 0.75), (0.6, 0.46, 0.26, 1) if lit else GLASS))
    b = L.SERVICE_BAY
    (cx, cy), (hx, hy), h = b["center"], b["half"], b["height"]
    g.append(box((cx, cy, h / 2), (hx, hy, h / 2), METAL_BLDG, name="service-bay"))
    for y in b["doors"]:
        g.append(box((cx - hx - 0.02, y, b["door_h"] / 2), (0.03, b["door_w"] / 2, b["door_h"] / 2), (0.3, 0.32, 0.35, 1)))
    t = L.FUEL_TANK
    (tx, ty), r = t["center"], t["diameter"] / 2
    g.append(box((tx, ty, 0.08), (*t["pad_half"], 0.08), CONCRETE))
    g.append(f'<geom name="fuel-tank" type="cylinder" fromto="{f((tx - t["length"] / 2, ty, 0.3 + r))} '
             f'{f((tx + t["length"] / 2, ty, 0.3 + r))}" size="{r:.4g}" rgba="0.72 0.72 0.7 1" friction="{FRICTION}"/>')
    g.append(box((tx, ty, 0.15), (t["length"] * 0.35, r * 0.8, 0.15), (0.06, 0.06, 0.06, 1)))
    for bx, by in t["bollards"]:
        g.append(cyl((bx, by, 0), (bx, by, 1.1), 0.085, YELLOW))
    for x, y in L.DRUMS:
        g.append(cyl((x, y, 0), (x, y, 0.88), 0.29, (0.5, 0.1, 0.05, 1)))
    for i, (x, y, yaw, color) in enumerate(L.DUMPSTERS):
        g.append(box((x, y, 0.7), (0.92, 0.68, 0.7), (*color, 1), name=f"dumpster-{i}", quat=yaw_quat(yaw)))
    for x, y in L.TIRE_STACKS:
        g.append(cyl((x, y, 0), (x, y, 0.75), 0.3, TIRE))
    for x, y, _heads, _kind in L.POLE_LIGHTS:
        g.append(cyl((x, y, 0), (x, y, 0.7), 0.3, CONCRETE))
        g.append(box((x, y, (L.POLE_H + 0.7) / 2), (0.075, 0.075, (L.POLE_H - 0.7) / 2), (0.08, 0.07, 0.06, 1)))
    for (mx, my, mz), _target, _lens in L.CAMERAS.values():
        g.append(cyl((mx, my, 0), (mx, my, mz + 0.3), 0.06, STEEL))
    sx, sy = L.POLE_SIGN["pos"]
    for dx in (-1.3, 1.3):
        g.append(box((sx + dx, sy, L.POLE_SIGN["height"] / 2), (0.1, 0.1, L.POLE_SIGN["height"] / 2), (0.06, 0.06, 0.06, 1)))
    # wheel stops: 1.8 m precast concrete, 0.13 m tall, at the back of each stall
    for row, y in L.ROW_Y.items():
        back = y + (L.STALL_DEPTH / 2 - 0.55) * (1 if L.ROW_NOSE_YAW[row] == 180.0 else -1)
        for x in L.STALL_X:
            g.append(box((x, back, 0.065), (0.9, 0.08, 0.065), CONCRETE))
    return g


def painted_lines() -> list[str]:
    out = []
    for y in L.ROW_Y.values():
        xs = [L.STALL_X[0] - L.STALL_PITCH / 2 + k * L.STALL_PITCH for k in range(len(L.STALL_X) + 1)]
        for x in xs:
            out.append(f'<geom type="box" pos="{f((x, y, 0.002))}" size="0.05 {L.STALL_DEPTH / 2:.4g} 0.002" '
                       f'rgba="0.62 0.62 0.6 1" contype="0" conaffinity="0"/>')
    return out


def camera_xyaxes(mount, target) -> str:
    """MuJoCo cameras look down their local -z. Build z = -forward, x = up x z (level), y = z x x."""
    fx, fy, fz = (target[i] - mount[i] for i in range(3))
    n = math.sqrt(fx * fx + fy * fy + fz * fz)
    zx, zy, zz = -fx / n, -fy / n, -fz / n
    xx, xy, xz = -zy, zx, 0.0
    xn = math.hypot(xx, xy)
    xx, xy = xx / xn, xy / xn
    yx, yy, yz = zy * xz - zz * xy, zz * xx - zx * xz, zx * xy - zy * xx
    return f((xx, xy, xz, yx, yy, yz))


def camera_xml(name: str, mount, target, lens) -> str:
    fx, fy, fz = (target[i] - mount[i] for i in range(3))
    n = math.sqrt(fx * fx + fy * fy + fz * fz)
    pos = [mount[i] + 0.35 * fc / n for i, fc in enumerate((fx, fy, fz))]
    fovy = math.degrees(CM.source_fov(lens)[1])
    return f'    <camera name="{name}" pos="{f(pos)}" xyaxes="{camera_xyaxes(mount, target)}" fovy="{fovy:.4g}"/>'


def person() -> str:
    px, py = L.PERSON_PARK
    c = PERSON
    parts = [
        capsule((-0.11, 0, 0.12), (-0.11, 0, 0.85), 0.085, c),
        capsule((0.11, 0, 0.12), (0.11, 0, 0.85), 0.085, c),
        f'<geom type="ellipsoid" pos="0 0 1.17" size="0.2 0.13 0.28" rgba="{f(c)}" friction="{FRICTION}"/>',
        capsule((-0.27, 0, 1.36), (-0.3, 0, 0.84), 0.06, c),
        capsule((0.27, 0, 1.36), (0.3, 0, 0.84), 0.06, c),
        f'<geom type="sphere" pos="0 0 1.64" size="0.12" rgba="0.07 0.07 0.08 1" friction="{FRICTION}"/>',
    ]
    inner = "\n      ".join(parts)
    return f'    <body name="person" mocap="true" pos="{f((px, py, 0.0))}">\n      {inner}\n    </body>'


def build() -> str:
    lines: list[str] = []
    w = lines.append
    w('<mujoco model="night-vehicle-lot">')
    w('  <compiler angle="radian" autolimits="true"/>')
    w("  <visual>")
    w(f'    <global offwidth="{CM.SRC_W}" offheight="{CM.SRC_H}" azimuth="90" elevation="-20"/>')
    w('    <headlight ambient="0.3 0.3 0.32" diffuse="0.2 0.2 0.22" specular="0 0 0"/>')
    w('    <quality shadowsize="4096" offsamples="4"/>')
    w('    <map znear="0.05" zfar="400" force="0.1"/>')
    w("  </visual>")
    w("  <asset>")
    w('    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.06 0.07 0.10" rgb2="0.01 0.01 0.02" width="512" height="512"/>')
    w('    <texture name="asphalt" type="2d" builtin="checker" rgb1="0.11 0.11 0.12" rgb2="0.10 0.10 0.11" '
      'width="256" height="256" mark="random" markrgb="0.2 0.2 0.2" random="0.06"/>')
    w('    <material name="asphalt" texture="asphalt" texrepeat="18 14" texuniform="true" reflectance="0.04" '
      'specular="0.1" shininess="0.1"/>')
    w("  </asset>")
    w("  <worldbody>")
    w('    <light name="moon" directional="true" dir="0.3 0.4 -1" diffuse="0.05 0.06 0.09" specular="0 0 0" castshadow="false"/>')
    w(f'    <geom name="ground" type="plane" pos="0 0 0" size="200 200 1" material="asphalt" friction="{FRICTION}"/>')
    for i, (lx, ly, heads, kind) in enumerate(L.POLE_LIGHTS):
        col = "1 0.6 0.25" if kind == "hps" else "1 0.9 0.8"
        w(f'    <light name="pole-{i + 1}" pos="{f((lx, ly, L.POLE_H - 0.3))}" dir="0 0 -1" diffuse="{col}" '
          f'specular="0 0 0" cutoff="70" exponent="4" attenuation="1 0 0.006" castshadow="true"/>')
    for line in painted_lines():
        w("    " + line)
    for g in fence():
        w("    " + g)
    for g in gates():
        w(g)
    for g in structures():
        w("    " + g)
    for name, (mount, target, lens) in L.CAMERAS.items():
        w(camera_xml(name, mount, target, lens))
    for spot, (_row, x, y, yaw) in sorted(L.SPOTS.items()):
        w(f'    <site name="spot-{spot}" pos="{f((x, y, 0.0))}" quat="{yaw_quat(yaw)}" size="0.05" group="5"/>')
    for p in json.loads(PLACES_JSON.read_text()):
        x, y = p["pose"]["x"], p["pose"]["y"]
        w(f'    <site name="place-{p["id"]}" pos="{f((x, y, 0.02))}" size="0.15" rgba="0.2 0.6 1 0.5" group="5"/>')
    for v in L.vehicles():
        w(vehicle_body(v))
    w(person())
    w("  </worldbody>")
    w("</mujoco>")
    return "\n".join(lines) + "\n"


def write_meta() -> Path:
    """scene.meta.json in DimOS's scene-package format, written by DimOS's own ScenePackage class."""
    from dimos.simulation.scene_assets.spec import SceneMeshAlignment, ScenePackage

    package = ScenePackage(
        package_dir=SCENE_DIR,
        source_path=Path("dimos_worlds/lot/layout.py"),
        # lot.xml is authored in the DimOS world frame: metres, z up, x east, y north, origin at the lot centre
        alignment=SceneMeshAlignment(y_up=False),
        mujoco_scene_path=OUT,
        metadata_path=META,
        stats={
            "name": "night-vehicle-lot",
            "extent_m": [2 * L.HX, 2 * L.HY],
            "vehicles": len(L.vehicles()),
            "cameras": sorted(L.CAMERAS),
            "gates": sorted(L.GATES),
            "places": len(json.loads(PLACES_JSON.read_text())),
            "places_file": PLACES_JSON.name,  # named places for navigation (dimos-worlds extension)
        },
    )
    path = package.write_metadata()
    return path


def main() -> None:
    OUT.write_text(build())
    print(f"wrote {OUT}")
    print(f"wrote {write_meta()}")


if __name__ == "__main__":
    main()
