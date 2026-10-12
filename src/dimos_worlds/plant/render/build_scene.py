"""Build plant.blend from layout.json: the Demo plant for camera renders and robot POV frames.

    blender -b --factory-startup --disable-autoexec -P render/build_scene.py -- --out assets/cache/plant.blend

Needs assets/fetch.py (Poly Haven CC0 textures + HDRIs) and make_textures.py (dial faces, plates) first.
Everything else is procedural here. Object custom properties used downstream:
  grp      floor | structures | roof | stairs   
  tag      instrument / equipment tag
State objects (scene_state.apply_state poses them from plant_state.truth): needle:<gauge tag> (rotation about the
dial normal), pointer:<valve tag> and stem:<valve tag> (gate valves), lever:<valve tag> (ball valve),
puddle:<drip tray tag> (hidden unless leaking). Lights carry w_day / w_night; emissive materials em_day / em_night.
Fixed cameras cam1..cam6 use lot/camera_model's pinhole source field of view, like the lot's.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
ENV = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ENV.parent / "lot"))  # dimos_worlds/lot/camera_model.py

import bhelp as H  # noqa: E402
import camera_model as CM  # noqa: E402

LAY = json.loads((ENV / "layout.json").read_text())
GEN = H.CACHE / "gen"
T0 = time.time()


def log(msg: str) -> None:
    print(f"BUILD [{time.time() - T0:5.1f}s] {msg}", flush=True)


def arg(name, default=None):
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return argv[argv.index(name) + 1] if name in argv else default


def tagged(o, grp: str, **props):
    o["grp"] = grp
    for k, v in props.items():
        o[k] = v
    return o


# ---------------------------------------------------------------- materials
def materials() -> dict:
    t = H.textured
    p = H.principled
    M = {
        "slab": t("slab", "concrete_floor_02", "2k", 3.0, tint=(0.92, 0.91, 0.88), variation=0.12),
        "gravel": t("gravel", "gravel_concrete_03", "1k", 2.0, tint=(0.85, 0.83, 0.8)),
        "cr_wall": t("cr_wall", "painted_concrete", "1k", 2.5, tint=(0.95, 0.93, 0.86), sat=0.12, value=1.15),
        "roof_metal": t("roof_metal", "corrugated_iron", "1k", 2.0, metal=0.6, tint=(0.8, 0.8, 0.82)),
        "concrete_plinth": t("concrete_plinth", "concrete_wall_008", "1k", 1.5, tint=(0.9, 0.9, 0.88)),
        "concrete_wall": t("concrete_wall", "concrete_wall_008", "1k", 2.0, tint=(0.85, 0.85, 0.82), variation=0.1),
        "tank_green": t("tank_green", "green_metal_rust", "1k", 3.0, metal=0.3),
        "motor_blue": t("motor_blue", "blue_metal_plate", "1k", 0.8, metal=0.4),
        "pump_blue": t("pump_blue", "blue_metal_plate", "1k", 0.6, metal=0.4, tint=(0.9, 0.95, 1.0)),
        "steel_rack": t("steel_rack", "rusty_metal_02", "1k", 2.0, metal=0.5, tint=(0.75, 0.72, 0.7)),
        "pipe_grey": t("pipe_grey", "metal_plate", "1k", 1.0, metal=0.6, tint=(0.75, 0.76, 0.78)),
        "grating": t("grating", "metal_grate_rusty", "1k", 1.0, metal=0.6),
        "door": t("door", "painted_metal_shutter", "1k", 2.0, metal=0.3),
        "steel_dark": p("steel_dark", (0.06, 0.065, 0.07), 0.45, 0.8),
        "steel_galv": p("steel_galv", (0.55, 0.57, 0.6), 0.35, 0.9),
        "steel_painted": p("steel_painted", (0.25, 0.27, 0.3), 0.5, 0.2),
        "guard_yellow": p("guard_yellow", (0.85, 0.6, 0.02), 0.45, 0.1),
        "panel_grey": p("panel_grey", (0.55, 0.57, 0.56), 0.4, 0.2),
        "desk": p("desk", (0.32, 0.27, 0.21), 0.6),
        "console": p("console", (0.03, 0.035, 0.04), 0.3),
        "tank_roof": p("tank_roof", (0.4, 0.42, 0.4), 0.6, 0.3),
        "comp_grey": p("comp_grey", (0.3, 0.33, 0.35), 0.45, 0.3),
        "receiver_grey": p("receiver_grey", (0.62, 0.64, 0.63), 0.4, 0.3),
        "eyewash_green": p("eyewash_green", (0.05, 0.45, 0.15), 0.4),
        "extinguisher_red": p("extinguisher_red", (0.6, 0.03, 0.02), 0.3),
        "pipe_green": p("pipe_green", (0.12, 0.3, 0.14), 0.45, 0.2),
        "pipe_yellow": p("pipe_yellow", (0.8, 0.6, 0.03), 0.45, 0.2),
        "pipe_blue": p("pipe_blue", (0.05, 0.2, 0.5), 0.45, 0.2),
        "pipe_red": p("pipe_red", (0.55, 0.05, 0.03), 0.45, 0.2),
        "walk_fill": p("walk_fill", (0.18, 0.32, 0.2), 0.75),
        "walk_edge": p("walk_edge", (0.85, 0.65, 0.02), 0.6),
        "bezel": p("bezel", (0.04, 0.04, 0.045), 0.3, 0.6),
        "needle": p("needle", (0.02, 0.02, 0.02), 0.4),
        "needle_red": p("needle_red", (0.6, 0.03, 0.02), 0.4),
        "brass": p("brass", (0.55, 0.42, 0.18), 0.35, 0.9),
        "handwheel": p("handwheel", (0.55, 0.04, 0.03), 0.4, 0.2),
        "lever_yellow": p("lever_yellow", (0.85, 0.65, 0.02), 0.4),
        "oil": p("oil", (0.01, 0.01, 0.012), 0.05, 0.0, coat=1.0, coat_rough=0.02),
        "glass": p("glass", (0.6, 0.7, 0.75), 0.05, transmission=0.9),
        "black_rubber": p("black_rubber", (0.02, 0.02, 0.02), 0.8),
    }
    M["lamp_led"] = H.emissive("lamp_led", (1.0, 0.95, 0.88), 40.0)
    M["lamp_led"]["em_day"] = 0.0
    M["lamp_led"]["em_night"] = 40.0
    M["annunciator"] = H.emissive("annunciator", (0.2, 1.0, 0.3), 4.0)
    M["annunciator"]["em_day"] = 2.0
    M["annunciator"]["em_night"] = 4.0
    M["screen"] = H.emissive("screen", (0.25, 0.45, 0.9), 2.0)
    M["screen"]["em_day"] = 1.5
    M["screen"]["em_night"] = 2.0
    return M


def image_material(name: str, png: Path, rough: float = 0.35, alpha: bool = False) -> bpy.types.Material:
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = H.image(png)
    tex.interpolation = "Cubic"
    tex.extension = "CLIP"
    nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
    if alpha:
        nt.links.new(tex.outputs["Alpha"], b.inputs["Alpha"])
    b.inputs["Roughness"].default_value = rough
    return m


# ---------------------------------------------------------------- helpers
def obj_from(mb: H.MeshBuilder, name: str, col, grp: str, smooth: float | None = 35.0, **props):
    return tagged(mb.to_object(name, col, smooth_angle=smooth), grp, **props)


def disc_uv(name: str, radius: float, depth: float, mat, col, segs: int = 64):
    """A disc in the XY plane (front face at +z) with planar UVs over its face: u, v in [0, 1] across the diameter."""
    mb = H.MeshBuilder()
    mb.cylinder((0, 0, -depth / 2), radius, depth, mat, segs=segs)
    o = mb.to_object(name, col, smooth_angle=None)
    me = o.data
    uv = me.uv_layers.new(name="uv")
    for poly in me.polygons:
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            uv.data[li].uv = (0.5 + co.x / (2 * radius), 0.5 + co.y / (2 * radius))
    return o


def frame_matrix(pos, facing) -> Matrix:
    """Local frame at an instrument: X = viewer's right, Y = up, Z = the way the face points."""
    fx, fy = facing
    right = Vector((-fy, fx, 0.0))
    up = Vector((0.0, 0.0, 1.0))
    nrm = Vector((fx, fy, 0.0))
    m = Matrix((right, up, nrm)).transposed().to_4x4()
    m.translation = Vector(pos)
    return m


def empty_at(name: str, col, matrix: Matrix):
    o = bpy.data.objects.new(name, None)
    o.matrix_world = matrix
    col.objects.link(o)
    return o


def parent_keep(child, parent):
    child.parent = parent
    child.matrix_parent_inverse = Matrix.Identity(4)


# ---------------------------------------------------------------- build
def build_ground(M, col):
    s = LAY["slab"]
    mb = H.MeshBuilder()
    mb.quad([(-40, -40, -0.02), (100, -40, -0.02), (100, 80, -0.02), (-40, 80, -0.02)], M["gravel"])
    tagged(mb.to_object("yard", col, smooth_angle=None), "floor")
    mb = H.MeshBuilder()
    mb.box(((s["x0"] + s["x1"]) / 2, (s["y0"] + s["y1"]) / 2, -0.01),
           ((s["x1"] - s["x0"]) / 2, (s["y1"] - s["y0"]) / 2, 0.01 + 0.002), M["slab"])
    tagged(mb.to_object("slab", col, smooth_angle=None), "floor")
    mb = H.MeshBuilder()
    for wk in LAY["walkways"]:
        xs = [p[0] for p in wk["poly"]]
        ys = [p[1] for p in wk["poly"]]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        mb.box(((x0 + x1) / 2, (y0 + y1) / 2, 0.0035), ((x1 - x0) / 2 - 0.08, (y1 - y0) / 2 - 0.08, 0.001), M["walk_fill"])
        for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)):
            cx, cy = (ax + bx) / 2, (ay + by) / 2
            hx = max(abs(bx - ax) / 2, 0.05)
            hy = max(abs(by - ay) / 2, 0.05)
            mb.box((cx, cy, 0.004), (hx, hy, 0.0012), M["walk_edge"])
    tagged(mb.to_object("walkways", col, smooth_angle=None), "floor")


def build_solids(M, col):
    groups: dict[str, H.MeshBuilder] = {}
    for so in LAY["solids"]:
        if so["id"].startswith(("M-", "P-", "T-", "C-301")) and "-fan" not in so["id"] and "-roof" not in so["id"]:
            continue  # equipment with detail: built below
        grp = {"roof": "roof", "stairs": "stairs"}.get(so["group"], "structures")
        mb = groups.setdefault(grp, H.MeshBuilder())
        mat = M.get(so["material"], M["steel_painted"])
        if so["shape"] == "box":
            sx, sy, sz = so["size"]
            bev = 0.01 if min(sx, sy, sz) > 0.05 else 0.0
            mb.box(tuple(so["center"]), (sx / 2, sy / 2, sz / 2), mat, bevel=bev)
        else:
            mb.tube(so["from"], so["to"], so["radius"], mat, segs=32 if so["radius"] > 1 else 20)
    for grp, mb in groups.items():
        obj_from(mb, f"solids_{grp}", col, grp)


def build_pipes(M, col):
    mb = H.MeshBuilder()
    for p in LAY["pipes"]:
        mat = M.get(p["material"], M["pipe_grey"])
        pts = p["points"]
        r = p["radius"]
        for a, b in zip(pts, pts[1:]):
            mb.tube(a, b, r, mat, segs=20 if r > 0.05 else 12)
        for q in pts[1:-1]:  # elbows
            mb.cylinder(tuple(q), r * 1.02, r * 2.04, mat, segs=16)
        for q, nxt in ((pts[0], pts[1]), (pts[-1], pts[-2])):  # flanges at the ends
            d = Vector(nxt) - Vector(q)
            if d.length < 1e-6 or q[2] < 0:
                continue
            d.normalize()
            rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
            m = Matrix.Translation(Vector(q) + d * 0.02) @ rot
            mb.cylinder((0, 0, 0), r * 1.6, 0.03, M["steel_galv"], segs=20, matrix=m)
    obj_from(mb, "pipes", col, "structures")


def build_equipment(M, col):
    sol = {s["id"]: s for s in LAY["solids"]}
    mb = H.MeshBuilder()
    for sid, s in sol.items():
        if s.get("thermal") == "motor":
            a, b, r = Vector(s["from"]), Vector(s["to"]), s["radius"]
            mb.tube(a, b, r, M["motor_blue"], segs=32)
            axis = (b - a).normalized()
            for k in range(1, 12):  # cooling fins
                c = a + (b - a) * (k / 12)
                rot = axis.to_track_quat("Z", "Y").to_matrix().to_4x4()
                mb.cylinder((0, 0, 0), r * 1.07, 0.012, M["motor_blue"], segs=32, matrix=Matrix.Translation(c) @ rot)
            mid = (a + b) / 2
            mb.box((mid.x, mid.y, mid.z + r + 0.07), (0.1, 0.09, 0.07), M["motor_blue"], bevel=0.01)  # terminal box
            for side in (-1, 1):  # feet
                mb.box((mid.x + side * 0.22, mid.y, mid.z - r - 0.02), (0.06, r * 0.9, 0.04), M["steel_dark"])
        elif s.get("thermal") == "pump":
            a, b, r = Vector(s["from"]), Vector(s["to"]), s["radius"]
            mb.tube(a, b, r, M["pump_blue"], segs=36)
            mid = (a + b) / 2
            mb.tube((mid.x, mid.y, mid.z + r * 0.6), (mid.x, mid.y, mid.z + r + 0.12), 0.07, M["pump_blue"])
            mb.cylinder((mid.x, mid.y, mid.z + r + 0.13), 0.11, 0.03, M["steel_galv"])
            mb.box((mid.x, mid.y, mid.z - r - 0.04), (0.12, 0.14, 0.06), M["pump_blue"])
        elif s.get("thermal") == "tank":
            a, b, r = Vector(s["from"]), Vector(s["to"]), s["radius"]
            mb.tube(a, b, r, M["tank_green"], segs=64)
            mb.cylinder((b.x, b.y, b.z + 0.25), r * 0.99, 0.5, M["tank_roof"], segs=64, radius2=r * 0.25)
            for k in range(1, 6):  # shell course seams
                mb.cylinder((a.x, a.y, a.z + k * 1.5), r + 0.01, 0.03, M["tank_green"], segs=64)
        elif sid == "C-301":
            cx, cy, cz = s["center"]
            sx, sy, sz = s["size"]
            mb.box((cx, cy, cz), (sx / 2, sy / 2, sz / 2), M["comp_grey"], bevel=0.03)
            for k in range(4):  # cylinder heads with fins
                mb.box((cx - 0.3 + k * 0.2, cy, cz + sz / 2 + 0.12), (0.07, 0.25, 0.12), M["steel_dark"], bevel=0.01)
    # air receiver details
    rec = sol["comp-receiver"]
    top = Vector(rec["to"])
    mb.cylinder((top.x, top.y, top.z + 0.15), rec["radius"], 0.3, M["receiver_grey"], segs=40, radius2=0.08)
    obj_from(mb, "equipment", col, "structures")


def build_gauge(spec, M, col):
    m = frame_matrix(spec["pos"], spec["facing"])
    root = empty_at(f"gauge:{spec['tag']}", col, m)
    R = spec["dial_m"] / 2
    face_mat = image_material(f"dial_{spec['tag']}", GEN / f"dial_{spec['tag']}.png", rough=0.25)
    face = disc_uv(f"face:{spec['tag']}", R, 0.004, face_mat, col)
    parent_keep(face, root)
    tagged(face, "structures", tag=spec["tag"])
    mb = H.MeshBuilder()
    mb.cylinder((0, 0, -0.03), R * 1.0, 0.055, M["bezel"], segs=48)  # case behind the face
    for k in range(48):  # bezel ring around the face (a ring of short tubes, leaves the face open)
        a0, a1 = 2 * math.pi * k / 48, 2 * math.pi * (k + 1) / 48
        mb.tube((R * 1.01 * math.cos(a0), R * 1.01 * math.sin(a0), 0.003),
                (R * 1.01 * math.cos(a1), R * 1.01 * math.sin(a1), 0.003), 0.007, M["steel_galv"], segs=8)
    mb.cylinder((0, -R - 0.03, -0.03), 0.015, 0.06, M["brass"], axis="y")  # stem down to the process connection
    case = mb.to_object(f"case:{spec['tag']}", col)
    parent_keep(case, root)
    tagged(case, "structures", tag=spec["tag"])
    # needle (state): local +X points at the value; rotated about local Z by the needle angle
    mb = H.MeshBuilder()
    mb.box((R * 0.36, 0, 0.006), (R * 0.5, R * 0.025, 0.0025), M["needle"])
    mb.box((R * 0.79, 0, 0.0062), (R * 0.07, R * 0.02, 0.0024), M["needle_red"])
    mb.box((-R * 0.07, 0, 0.006), (R * 0.07, R * 0.03, 0.0025), M["needle"])
    mb.cylinder((0, 0, 0.008), R * 0.06, 0.006, M["needle"], segs=24)
    nd = mb.to_object(f"needle:{spec['tag']}", col, smooth_angle=None)
    parent_keep(nd, root)
    tagged(nd, "structures", tag=spec["tag"])
    plate(spec, M, col, root, offset=(0, -R - 0.07, -0.02))


def plate(spec, M, col, root, offset):
    png = GEN / f"plate_{spec['tag']}.png"
    if not png.exists():
        return
    mat = image_material(f"plate_{spec['tag']}", png, rough=0.5)
    mb = H.MeshBuilder()
    w, h = 0.12, 0.0375
    x, y, z = offset
    mb.quad([(x - w / 2, y - h / 2, z), (x + w / 2, y - h / 2, z), (x + w / 2, y + h / 2, z), (x - w / 2, y + h / 2, z)], mat)
    o = mb.to_object(f"plate:{spec['tag']}", col, smooth_angle=None)
    uv = o.data.uv_layers.new(name="uv")
    for li, (u, v) in zip(range(4), ((0, 0), (1, 0), (1, 1), (0, 1))):
        uv.data[li].uv = (u, v)
    parent_keep(o, root)
    tagged(o, "structures", tag=spec["tag"])


def build_gate_valve(spec, M, col):
    """Gate valve in a vertical (style gate) or horizontal (gate_h) line: body, bonnet, rising stem, handwheel, and
    an OPEN/SHUT position plate with a pointer facing the walkway."""
    fx, fy = spec["facing"]
    x, y, z = spec["pos"]
    mb = H.MeshBuilder()
    horizontal = spec["style"] == "gate_h"
    if horizontal:  # line runs along y; bonnet up
        mb.cylinder((x, y, z), 0.12, 0.26, M["steel_dark"], axis="y", segs=24)
        for dy in (-0.14, 0.14):
            mb.cylinder((x, y + dy, z), 0.15, 0.03, M["steel_dark"], axis="y", segs=24)
        mb.box((x, y, z + 0.2), (0.07, 0.07, 0.12), M["steel_dark"], bevel=0.01)
        wheel_c = Vector((x, y, z + 0.42))
        wheel_axis = "z"
        plate_m = frame_matrix((x + 0.24, y + fy * 0.21, z + 0.3), (fx, fy))  # on a bracket beside the bonnet
        mb.box((x + 0.15, y + fy * 0.15, z + 0.3), (0.1, 0.012, 0.012), M["steel_galv"])
    else:  # vertical riser; the stem comes out toward the walkway
        mb.cylinder((x, y + 0.0, z), 0.11, 0.26, M["steel_dark"], segs=24)
        for dz in (-0.14, 0.14):
            mb.cylinder((x, y, z + dz), 0.14, 0.03, M["steel_dark"], segs=24)
        mb.box((x + fx * 0.18, y + fy * 0.18, z), (0.07 if fy else 0.1, 0.1 if fy else 0.07, 0.07), M["steel_dark"])
        wheel_c = Vector((x + fx * 0.36, y + fy * 0.36, z))
        wheel_axis = "y" if fy else "x"
        # clear of the handwheel's disc as the payload sees it from the walkway (at 0.17 the wheel hid the plate when
        # the valve sits at camera height, XV-301)
        plate_m = frame_matrix((x + 0.30, y + fy * 0.12, z + 0.06), (fx, fy))
        mb.box((x + 0.19, y + fy * 0.1, z + 0.06), (0.11, 0.012, 0.012), M["steel_galv"])  # bracket (half sizes)
    obj_from(mb, f"valve:{spec['tag']}", col, "structures", tag=spec["tag"])
    # handwheel (torus approximated by a ring of tubes) with spokes
    mb = H.MeshBuilder()
    rw = 0.16
    n = 20
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        if wheel_axis == "z":
            p0 = wheel_c + Vector((rw * math.cos(a0), rw * math.sin(a0), 0))
            p1 = wheel_c + Vector((rw * math.cos(a1), rw * math.sin(a1), 0))
        elif wheel_axis == "y":
            p0 = wheel_c + Vector((rw * math.cos(a0), 0, rw * math.sin(a0)))
            p1 = wheel_c + Vector((rw * math.cos(a1), 0, rw * math.sin(a1)))
        else:
            p0 = wheel_c + Vector((0, rw * math.cos(a0), rw * math.sin(a0)))
            p1 = wheel_c + Vector((0, rw * math.cos(a1), rw * math.sin(a1)))
        mb.tube(p0, p1, 0.014, M["handwheel"], segs=8)
    for k in range(3):
        a = 2 * math.pi * k / 3
        off = {"z": Vector((math.cos(a), math.sin(a), 0)), "y": Vector((math.cos(a), 0, math.sin(a))),
               "x": Vector((0, math.cos(a), math.sin(a)))}[wheel_axis] * rw
        mb.tube(wheel_c, wheel_c + off, 0.01, M["handwheel"], segs=6)
    obj_from(mb, f"handwheel:{spec['tag']}", col, "structures", tag=spec["tag"])
    # rising stem (state): length scaled by valve position; from the wheel outward
    mb = H.MeshBuilder()
    mb.cylinder((0, 0, 0.06), 0.016, 0.12, M["steel_galv"], segs=12)
    stem = mb.to_object(f"stem:{spec['tag']}", col, smooth_angle=None)
    d = {"z": Vector((0, 0, 1)), "y": Vector((0, fy, 0)), "x": Vector((fx, 0, 0))}[wheel_axis]
    stem.matrix_world = Matrix.Translation(wheel_c) @ d.to_track_quat("Z", "Y").to_matrix().to_4x4()
    tagged(stem, "structures", tag=spec["tag"])
    # position plate + pointer (state)
    root = empty_at(f"vplate:{spec['tag']}", col, plate_m)
    face = disc_uv(f"vface:{spec['tag']}", 0.075, 0.004, image_material("valve_plate", GEN / "valve_plate.png", 0.4),
                   col)
    parent_keep(face, root)
    tagged(face, "structures", tag=spec["tag"])
    mb = H.MeshBuilder()
    mb.box((0.016, 0, 0.006), (0.021, 0.006, 0.003), M["needle"])  # half the plate radius: stops short of the labels
    mb.cylinder((0, 0, 0.007), 0.01, 0.006, M["needle"], segs=16)
    ptr = mb.to_object(f"pointer:{spec['tag']}", col, smooth_angle=None)
    parent_keep(ptr, root)
    tagged(ptr, "structures", tag=spec["tag"])
    plate(spec, M, col, root, offset=(0, -0.11, 0.0))


def build_ball_valve(spec, M, col):
    x, y, z = spec["pos"]
    mb = H.MeshBuilder()
    mb.cylinder((x, y, z), 0.075, 0.18, M["steel_galv"], axis="y", segs=24)
    mb.cylinder((x, y, z), 0.09, 0.09, M["brass"], axis="y", segs=24)
    mb.cylinder((x, y, z + 0.1), 0.02, 0.06, M["steel_galv"], segs=12)
    obj_from(mb, f"valve:{spec['tag']}", col, "structures", tag=spec["tag"])
    mb = H.MeshBuilder()  # lever (state): local +X along its length, rotated about z
    mb.box((0.14, 0, 0), (0.15, 0.018, 0.008), M["lever_yellow"], bevel=0.004)
    mb.box((0.27, 0, 0), (0.05, 0.026, 0.014), M["black_rubber"], bevel=0.004)
    lv = mb.to_object(f"lever:{spec['tag']}", col, smooth_angle=None)
    lv.location = (x, y, z + 0.135)
    tagged(lv, "structures", tag=spec["tag"])
    root = empty_at(f"bplate:{spec['tag']}", col, frame_matrix((x, y - 0.11, z + 0.02), spec["facing"]))
    plate(spec, M, col, root, offset=(0, 0.0, 0.0))


def build_trays(spec, M, col):
    x, y, _ = spec["pos"]
    mb = H.MeshBuilder()
    mb.box((x, y, 0.03), (0.45, 0.2, 0.03), M["steel_galv"])
    obj_from(mb, f"tray:{spec['tag']}", col, "structures", tag=spec["tag"])
    mb = H.MeshBuilder()  # puddle (state): irregular dark film spreading south of the tray
    n = 28
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        rr = 0.55 + 0.18 * math.sin(3 * a + 1.3) + 0.1 * math.sin(7 * a)
        pts.append((x + rr * 1.3 * math.cos(a), y - 0.45 + rr * 0.7 * math.sin(a), 0.006))
    mb.quad(pts, M["oil"])
    pd = mb.to_object(f"puddle:{spec['tag']}", col, smooth_angle=None)
    tagged(pd, "structures", tag=spec["tag"])
    pd.hide_render = True


def build_control_room_details(M, col):
    mb = H.MeshBuilder()
    for i in range(6):  # annunciator lamps on the panel face
        for j in range(2):
            mb.box((7.1 + i * 0.36, 33.24, 1.65 + j * 0.16), (0.07, 0.006, 0.05), M["annunciator"])
    for i in range(3):  # console screens
        mb.box((4.1 + i * 0.9, 29.64, 1.05), (0.33, 0.01, 0.2), M["screen"])
    for wx in (4.0, 11.0):  # windows on the south wall (the doorway stays open)
        mb.box((wx, 26.0, 1.6), (1.0, 0.14, 0.5), M["glass"])
    obj_from(mb, "cr_details", col, "structures")


def build_signs(M, col):
    signs = [("T-201", (40.0, 30.6 - 3.22, 3.0), (0, -1), 1.6), ("T-202", (48.0, 30.6 - 3.22, 3.0), (0, -1), 1.6),
             ("CONTROL ROOM", (8.0, 25.86, 2.9), (0, -1), 2.0), ("C-301", (47.2, 6.48, 1.0), (0, -1), 0.5),
             ("NO STEP - STAIRS", (44.0, 24.8, 0.6), (0, -1), 1.2), ("NO STEP - STAIRS", (50.5, 17.25, 0.9), (0, -1), 1.2)]
    for n in ("101", "102", "103"):
        px = {"101": 22.0, "102": 27.0, "103": 32.0}[n]
        signs.append((f"P-{n}", (px - 0.6, 13.98, 0.35), (0, -1), 0.4))
    for i, (txt, pos, facing, width) in enumerate(signs):
        png = GEN / f"sign_{txt.replace(' ', '_')}.png"
        mat = image_material(f"sign_{txt}", png, 0.5)
        aspect = (1024 if len(txt) > 8 else 512) / 160
        m = frame_matrix(pos, facing)
        mb = H.MeshBuilder()
        w, h = width, width / aspect
        mb.quad([(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)], mat)
        o = mb.to_object(f"sign:{i}", col, smooth_angle=None)
        uv = o.data.uv_layers.new(name="uv")
        for li, (u, v) in zip(range(4), ((0, 0), (1, 0), (1, 1), (0, 1))):
            uv.data[li].uv = (u, v)
        o.matrix_world = m
        tagged(o, "structures")


def build_lights(M, col):
    for lt in LAY["lights"]:
        x, y, z = lt["pos"]
        kelvin_rgb = (1.0, 0.92, 0.82) if lt["kelvin"] <= 4000 else (1.0, 0.97, 0.94)
        if lt["kind"] == "pole":
            mb = H.MeshBuilder()
            mb.box((x + 0.35, y, z), (0.4, 0.18, 0.05), M["steel_painted"], bevel=0.01)
            mb.box((x + 0.35, y, z - 0.055), (0.3, 0.12, 0.006), M["lamp_led"])
            mb.tube((x, y, z - 0.1), (x + 0.1, y, z), 0.04, M["steel_galv"])
            obj_from(mb, f"lamp:{lt['id']}", col, "structures")
            ld = bpy.data.lights.new(lt["id"], "SPOT")
            ld.spot_size = math.radians(140)
            ld.spot_blend = 0.5
            ld.shadow_soft_size = 0.15
            ld.color = kelvin_rgb
            o = bpy.data.objects.new(lt["id"], ld)
            o.location = (x + 0.35, y, z - 0.08)
            o.rotation_euler = (0, 0, 0)  # spot points -Z (down)
            o["w_night"] = 2600.0
            o["w_day"] = 0.0
        elif lt["kind"] == "wallpack":
            ld = bpy.data.lights.new(lt["id"], "SPOT")
            ld.spot_size = math.radians(120)
            ld.spot_blend = 0.6
            ld.color = kelvin_rgb
            o = bpy.data.objects.new(lt["id"], ld)
            o.location = (x, y - 0.25, z)
            o.rotation_euler = (math.radians(35), 0, 0)
            o["w_night"] = 700.0
            o["w_day"] = 0.0
        else:  # indoor / shelter: area lights on in both modes
            ld = bpy.data.lights.new(lt["id"], "AREA")
            ld.size = 2.0
            ld.color = kelvin_rgb
            o = bpy.data.objects.new(lt["id"], ld)
            o.location = (x, y, z)
            o["w_night"] = 500.0
            o["w_day"] = 500.0 if lt["id"] == "cr-ceiling" else 0.0
        col.objects.link(o)
    sun = bpy.data.lights.new("sun", "SUN")
    sun.angle = math.radians(1.0)
    sun.color = (1.0, 0.96, 0.9)
    so = bpy.data.objects.new("sun", sun)
    so.rotation_euler = (math.radians(50), math.radians(10), math.radians(-35))
    so["w_day"] = 3.2
    so["w_night"] = 0.0
    col.objects.link(so)


def build_world():
    w = bpy.data.worlds.new("plant_world")
    bpy.context.scene.world = w
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.name = "env"
    day = H.image(H.CACHE / "kloofendal_overcast_puresky" / "kloofendal_overcast_puresky_1k.hdr")
    night = H.image(H.CACHE / "kloppenheim_02_puresky" / "kloppenheim_02_puresky_1k.hdr")
    env.image = day
    day.use_fake_user = True
    night.use_fake_user = True  # unused until night mode: keep it in the saved .blend
    nt.links.new(env.outputs["Color"], nt.nodes["Background"].inputs["Color"])
    w["day_hdri"] = day.name
    w["night_hdri"] = night.name


def build_cameras(col):
    for cid, c in LAY["cameras"].items():
        cd = bpy.data.cameras.new(cid)
        hfov, _ = CM.source_fov(c["lens_mm"])
        cd.sensor_fit = "HORIZONTAL"
        cd.angle = hfov
        cd.clip_start = 0.05
        cd.clip_end = 400
        o = bpy.data.objects.new(cid, cd)
        o.location = c["mount"]
        d = Vector(c["target"]) - Vector(c["mount"])
        o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        col.objects.link(o)
        # housing on its mount
        mb = H.MeshBuilder()
        mb.box((0, 0, 0.12), (0.06, 0.06, 0.16), H.principled("cam_housing", (0.85, 0.85, 0.83), 0.4))
        h = mb.to_object(f"housing:{cid}", col)
        h.matrix_world = o.matrix_world
        tagged(h, "structures")


def main() -> None:
    out = Path(arg("--out", str(H.CACHE / "plant.blend")))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    col = H.collection("plant")
    M = materials()
    log("materials")
    build_world()
    build_ground(M, col)
    build_solids(M, col)
    build_pipes(M, col)
    build_equipment(M, col)
    for spec in LAY["instruments"]:
        if spec["kind"] == "gauge":
            build_gauge(spec, M, col)
        elif spec["kind"] == "valve" and spec["style"] in ("gate", "gate_h"):
            build_gate_valve(spec, M, col)
        elif spec["kind"] == "valve":
            build_ball_valve(spec, M, col)
        elif spec["kind"] == "drip_tray":
            build_trays(spec, M, col)
    build_control_room_details(M, col)
    build_signs(M, col)
    build_lights(M, col)
    build_cameras(col)
    log(f"objects: {len(bpy.data.objects)}")
    sys.path.insert(0, str(HERE))
    import scene_state  # noqa: PLC0415

    scene_state.set_mode(scene, "day")
    scene_state.apply_state([])
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True)
    log(f"BUILD_OK {out}")


if __name__ == "__main__":
    main()
