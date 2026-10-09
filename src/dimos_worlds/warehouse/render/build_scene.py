"""Build the warehouse scene from layout.json and the cached CC0 assets; save assets/cache/warehouse.blend.
Optional, render-only: nothing in the MuJoCo scene package or the sims needs it.

    blender -b --factory-startup --disable-autoexec -P render/build_scene.py -- --out assets/cache/warehouse.blend \
        [--state STATE.json] [--robots assets/cache/robots.npz]

(paths relative to src/dimos_worlds/warehouse/). Needs assets/cache/robots.npz (python render/robots_fk.py) and the
Poly Haven cache (python assets/fetch.py). Metres, origin = SW interior corner, x east, y north, z up (layout.json).

Totes, the G1 and the iiwa come from cell.py (the same layout the MuJoCo scene uses): totes at their initial slots,
the G1 in its charging spot. --state poses them as a recorded moment instead (render/render.py --state):
{"totes": {id: [x, y, z_centre, yaw]}, "g1": [x, y, yaw], "go2": [x, y, yaw], "g1Holding": id|null, "armHolding": id|null,
 "armTip": [x, y, z]}; the iiwa pose for armTip is solved by robots_fk.py (--robots).

Built on Ubuntu 22.04 with Blender 5.2.2 (1398 objects with the Go2). --state is ported but has not been run here.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "lot"))  # the lot package's camera model (lens field of view)

import bhelp as H  # noqa: E402
import camera_model as CM  # noqa: E402

sys.path.insert(0, str(HERE.parents[2]))
from dimos_worlds.warehouse import cell as CELL  # noqa: E402  (pure python: the layout's places, slots and totes)

LAY = json.loads((HERE.parent / "layout.json").read_text())
STATE: dict = {}
W, D, HT = LAY["building"]["size"]
WT = 0.25  # wall thickness (outside the interior rectangle)
RNG = random.Random(11)
ROOT = None  # root collection


def col(name):
    return H.collection(name, ROOT)


def mk(mb, name, grp, smooth_angle=35.0):
    o = mb.to_object(name, ROOT, smooth_angle)
    o["grp"] = grp
    return o


# ---- materials -------------------------------------------------------------------------------


def materials() -> dict:
    M = {}
    M["slab"] = H.textured("slab", "concrete_floor_worn_001", "2k", 3.0, tint=(0.9, 0.9, 0.88), sat=0.35, value=1.7,
                           rough_add=-0.12, variation=0.3)
    M["wall_low"] = H.principled("wall_low", (0.5, 0.49, 0.45), rough=0.75)
    M["wall_up"] = H.textured("wall_up", "box_profile_metal_sheet", "1k", 2.0, tint=(0.85, 0.87, 0.9), sat=0.0,
                              value=2.2, metal=0.1)
    M["roof"] = H.textured("roof_deck", "box_profile_metal_sheet", "1k", 2.0, tint=(0.95, 0.95, 0.95), sat=0.0, value=3.4,
                           metal=0.0)
    M["ground"] = H.textured("ground_out", "concrete_floor_02", "1k", 3.0, sat=0.4, value=1.0)
    M["wood_a"] = H.textured("pallet_wood_a", "rough_wood", "1k", 0.5, tint=(1.0, 0.85, 0.65), sat=0.8, value=1.5)
    M["wood_b"] = H.textured("pallet_wood_b", "rough_wood", "1k", 0.5, tint=(0.8, 0.68, 0.52), sat=0.7, value=1.2)
    M["ply"] = H.textured("table_ply", "plywood", "1k", 0.6, tint=(1, 0.92, 0.8), sat=0.8, value=1.3)
    M["blue"] = H.principled("rack_blue", (0.015, 0.07, 0.4), rough=0.4, metal=0.2)
    M["orange"] = H.principled("rack_orange", (0.75, 0.16, 0.01), rough=0.4, metal=0.2)
    M["galv"] = H.principled("galvanized", (0.5, 0.52, 0.53), rough=0.4, metal=0.85)
    M["dark"] = H.principled("dark_steel", (0.04, 0.04, 0.045), rough=0.55, metal=0.6)
    M["grey_steel"] = H.principled("grey_steel", (0.2, 0.21, 0.22), rough=0.5, metal=0.5)
    M["yellow"] = H.principled("safety_yellow", (0.7, 0.5, 0.02), rough=0.5)
    M["red_pipe"] = H.principled("red_pipe", (0.5, 0.02, 0.015), rough=0.5, metal=0.2)
    M["duct"] = H.principled("duct", (0.55, 0.56, 0.58), rough=0.35, metal=0.9)
    M["door_metal"] = H.principled("door_metal", (0.6, 0.62, 0.63), rough=0.4, metal=0.7)
    M["leaf"] = H.principled("door_leaf", (0.35, 0.38, 0.36), rough=0.5, metal=0.3)
    M["carton"] = H.principled("carton_proxy", (0.36, 0.25, 0.14), rough=0.8)
    M["carton_lite"] = H.principled("carton_proxy_lite", (0.42, 0.31, 0.18), rough=0.8)
    M["wrap"] = H.principled("stretch_wrap", (0.8, 0.84, 0.88), rough=0.12, alpha=0.2)
    M["wrap_proxy"] = H.principled("stretch_wrap_proxy", (0.55, 0.58, 0.62), rough=0.2)
    M["tote_blue"] = H.principled("tote_blue", (0.015, 0.07, 0.3), rough=0.35)
    M["tote_grey"] = H.principled("tote_grey", (0.12, 0.13, 0.14), rough=0.35)
    M["tote_green"] = H.principled("tote_green", (0.02, 0.18, 0.06), rough=0.35)
    M["paint_y"] = H.principled("paint_yellow", (0.62, 0.45, 0.02), rough=0.55)
    M["paint_w"] = H.principled("paint_white", (0.6, 0.6, 0.57), rough=0.55)
    M["paint_g"] = H.principled("paint_green", (0.02, 0.35, 0.08), rough=0.55)
    M["paint_b"] = H.principled("paint_blue", (0.025, 0.1, 0.3), rough=0.55)
    M["skid"] = H.principled("tyre_mark", (0.0, 0.0, 0.0), rough=0.5, alpha=0.12)
    M["oil"] = H.principled("oil_stain", (0.005, 0.005, 0.005), rough=0.35, alpha=0.45)
    M["label"] = H.principled("bin_label", (0.7, 0.7, 0.68), rough=0.5)
    M["text_k"] = H.principled("text_black", (0.01, 0.01, 0.01), rough=0.6)
    M["text_w"] = H.principled("text_white", (0.8, 0.8, 0.78), rough=0.5)
    M["sign_blue"] = H.principled("sign_blue", (0.01, 0.07, 0.4), rough=0.4)
    M["sign_yel"] = H.principled("sign_yellow", (0.75, 0.52, 0.01), rough=0.4)
    M["sign_red"] = H.principled("sign_red", (0.5, 0.02, 0.02), rough=0.4)
    M["sign_grn"] = H.principled("sign_green", (0.02, 0.35, 0.1), rough=0.4)
    M["roller"] = H.principled("roller_zinc", (0.6, 0.62, 0.64), rough=0.3, metal=0.9)
    M["motor"] = H.principled("motor_blue", (0.02, 0.1, 0.35), rough=0.4, metal=0.3)
    M["rubber"] = H.principled("belt_rubber", (0.015, 0.015, 0.015), rough=0.8)
    M["white_plastic"] = H.principled("white_plastic", (0.7, 0.7, 0.68), rough=0.35)
    M["camera"] = H.principled("cam_housing", (0.7, 0.7, 0.68), rough=0.4)
    M["glass"] = H.principled("lens_glass", (0.01, 0.012, 0.015), rough=0.05, spec=1.0)
    M["hazard"] = hazard_material()
    for name, col_, a, b in (("led_a", (1.0, 0.95, 0.88), 30.0, 3.0), ("led_b", (1.0, 0.95, 0.88), 30.0, 0.0)):
        m = H.emissive(name, col_, a)
        m["em_day"], m["em_night"] = a, b
        M[name] = m
    sky = H.emissive("skylight_panel", (0.85, 0.92, 1.0), 3.0)
    sky["em_day"], sky["em_night"] = 3.0, 0.0
    M["skylight"] = sky
    ex = H.emissive("exit_sign", (0.1, 1.0, 0.3), 6.0)
    ex["em_day"] = ex["em_night"] = 6.0
    M["exit"] = ex
    led = H.emissive("status_led", (0.1, 1.0, 0.2), 8.0)
    led["em_day"] = led["em_night"] = 8.0
    M["status_led"] = led
    return M


def hazard_material():
    m = bpy.data.materials.new("hazard_stripes")
    m.use_nodes = True
    nt = m.node_tree
    n, ln = nt.nodes, nt.links
    b = n["Principled BSDF"]
    tc = n.new("ShaderNodeTexCoord")
    wv = n.new("ShaderNodeTexWave")
    wv.wave_type = "BANDS"
    wv.bands_direction = "DIAGONAL"
    wv.wave_profile = "SAW"
    wv.inputs["Scale"].default_value = 5.0
    wv.inputs["Distortion"].default_value = 0.0
    ln.new(tc.outputs["Object"], wv.inputs["Vector"])
    cr = n.new("ShaderNodeValToRGB")
    cr.color_ramp.interpolation = "CONSTANT"
    cr.color_ramp.elements[0].color = (0.01, 0.01, 0.01, 1)
    cr.color_ramp.elements[1].position = 0.5
    cr.color_ramp.elements[1].color = (0.62, 0.45, 0.02, 1)
    ln.new(wv.outputs["Fac"], cr.inputs["Fac"])
    ln.new(cr.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.55
    return m


# ---- mesh helpers -----------------------------------------------------------------------------


def fquad(mb, x0, y0, x1, y1, z, mat):
    return mb.quad([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], mat)


def line(mb, x0, y0, x1, y1, w, mat, z=0.003):
    """Floor line from (x0,y0) to (x1,y1), width w (axis-aligned or not)."""
    d = Vector((x1 - x0, y1 - y0, 0))
    n = Vector((-d.y, d.x, 0)).normalized() * (w / 2)
    a, b = Vector((x0, y0, z)), Vector((x1, y1, z))
    return mb.quad([a - n, b - n, b + n, a + n], mat)


def rect_outline(mb, x0, y0, x1, y1, w, mat):
    line(mb, x0, y0 + w / 2, x1, y0 + w / 2, w, mat)
    line(mb, x0, y1 - w / 2, x1, y1 - w / 2, w, mat)
    line(mb, x0 + w / 2, y0 + w, x0 + w / 2, y1 - w, w, mat)
    line(mb, x1 - w / 2, y0 + w, x1 - w / 2, y1 - w, w, mat)


def wall_rot(side):  # text rotation so the text faces into the room
    return {"S": (90, 0, 180), "N": (90, 0, 0), "W": (90, 0, 90), "E": (90, 0, -90)}[side]


def wall_pt(side, a, z, off):
    """Point on the interior face of a wall, `off` metres into the room; a = coordinate along the wall."""
    return {"S": (a, off, z), "N": (a, D - off, z), "W": (off, a, z), "E": (W - off, a, z)}[side]


def wall_sign(M, side, a, z, w, h, bg, body, txt, size, grp="roof", tcol=None):  # on the wall above the cutaway
    mb = H.MeshBuilder()
    cx, cy, cz = wall_pt(side, a, z, 0.015)
    ax = (w / 2, 0.012, h / 2) if side in ("S", "N") else (0.012, w / 2, h / 2)
    mb.box((cx, cy, cz), ax, M[bg], bevel=0.004)
    mk(mb, f"sign_{body[:10]}", grp)
    px, py, pz = wall_pt(side, a, z, 0.03)
    for i, line_ in enumerate(body.split("\n")):
        dz = (len(body.split("\n")) - 1) / 2 * size * 1.4 - i * size * 1.4
        loc = (px, py, pz + dz)
        t = H.text(f"txt_{body[:8]}{i}", line_, size, M[tcol or txt], ROOT, loc, wall_rot(side), 0.002)
        t["grp"] = grp


# ---- assets -----------------------------------------------------------------------------------


def load_proto(asset: str, res: str = "1k"):
    """Join a Poly Haven model into one mesh, origin at the bottom centre. Returns (mesh, (sx, sy, sz))."""
    blend = H.CACHE / asset / f"{asset}_{res}.blend"
    objs = H.append_objects(blend)
    tmp = bpy.data.collections.new("tmp_" + asset)
    bpy.context.scene.collection.children.link(tmp)
    for o in objs:
        tmp.objects.link(o)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    parts = []
    for o in objs:
        if o.type != "MESH":
            continue
        ev = o.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
        me.transform(o.matrix_world)
        no = bpy.data.objects.new(asset + "_p", me)
        tmp.objects.link(no)
        parts.append(no)
    for o in objs:
        bpy.data.objects.remove(o)
    active = parts[0]
    with bpy.context.temp_override(active_object=active, selected_editable_objects=parts, selected_objects=parts):
        bpy.ops.object.join()
    me = active.data
    vs = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", vs)
    vs = vs.reshape(-1, 3)
    lo, hi = vs.min(axis=0), vs.max(axis=0)
    me.transform(Matrix.Translation((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
    size = tuple(float(x) for x in (hi - lo))
    bpy.data.objects.remove(active)
    bpy.data.collections.remove(tmp)
    me.name = asset
    print(f"PROTO {asset} size {size} mats {[m.name for m in me.materials if m]} tris {len(me.polygons)}", flush=True)
    return me, size


def instance(me, loc, yaw_deg=0.0, scale=(1, 1, 1), name="inst", grp="render_only"):
    o = bpy.data.objects.new(name, me)
    o.location = loc
    o.rotation_euler = (0, 0, math.radians(yaw_deg))
    o.scale = scale
    ROOT.objects.link(o)
    o["grp"] = grp
    return o


# ---- building shell ----------------------------------------------------------------------------

LOW_TOP = 3.2
WEB_CUT = 1.0  # the web GLB is a cutaway: walls end here (above is baked into the light, not exported)


def shell(M):
    low, mid, up = H.MeshBuilder(), H.MeshBuilder(), H.MeshBuilder()
    walls = {"S": [], "N": [], "W": [], "E": []}
    for d in LAY["doors"]:
        r = d["rect"]
        w = d["wall"][0].upper()
        a0, a1 = (r[0], r[2]) if w in "SN" else (r[1], r[3])
        walls[w].append((a0, a1, d["height"]))

    def emit(side, a0, a1, z0, z1):
        for lo_, hi_, mat_mb in ((z0, min(z1, WEB_CUT), low), (max(z0, WEB_CUT), min(z1, LOW_TOP), mid),
                                 (max(z0, LOW_TOP), z1, up)):
            if hi_ - lo_ < 1e-4:
                continue
            if side == "S":
                mat_mb.box(((a0 + a1) / 2, -WT / 2, (lo_ + hi_) / 2), ((a1 - a0) / 2, WT / 2, (hi_ - lo_) / 2),
                           M["wall_up"] if mat_mb is up else M["wall_low"])
            elif side == "N":
                mat_mb.box(((a0 + a1) / 2, D + WT / 2, (lo_ + hi_) / 2), ((a1 - a0) / 2, WT / 2, (hi_ - lo_) / 2),
                           M["wall_up"] if mat_mb is up else M["wall_low"])
            elif side == "W":
                mat_mb.box((-WT / 2, (a0 + a1) / 2, (lo_ + hi_) / 2), (WT / 2, (a1 - a0) / 2, (hi_ - lo_) / 2),
                           M["wall_up"] if mat_mb is up else M["wall_low"])
            else:
                mat_mb.box((W + WT / 2, (a0 + a1) / 2, (lo_ + hi_) / 2), (WT / 2, (a1 - a0) / 2, (hi_ - lo_) / 2),
                           M["wall_up"] if mat_mb is up else M["wall_low"])

    for side, ops in walls.items():
        full = (-WT, W + WT) if side in "SN" else (0.0, D)
        cur = full[0]
        for a0, a1, h in sorted(ops):
            emit(side, cur, a0, 0.0, HT)
            emit(side, a0, a1, h, HT)
            cur = a1
        emit(side, cur, full[1], 0.0, HT)
    # pilasters on N and S walls (clear of the roll-up door) and a 15 cm dark base rail
    for x in (3.0, 9.0, 15.0, 21.0):
        for side in ("S", "N"):
            if side == "S" and 23.5 < x < 29:
                continue
            y = 0.15 if side == "S" else D - 0.15
            mid.box((x, y, LOW_TOP / 2 + 0.1), (0.25, 0.15, LOW_TOP / 2 + 0.1), M["wall_low"])
    mk(low, "walls_lower", "shell")
    # Above WEB_CUT the walls are in the renders and in the bake's light, not in the web GLB: a map seen from above
    # at an angle would otherwise look at the south wall instead of into the building ("roof" = never exported).
    mk(mid, "walls_middle", "roof")
    mk(up, "walls_upper", "roof")
    # roof deck, trusses, purlins (never in the web GLB; they bounce light)
    rf = H.MeshBuilder()
    rf.box((W / 2, D / 2, HT + 0.125), (W / 2 + WT, D / 2 + WT, 0.125), M["roof"])
    for x in (2.5, 8.0, 13.5, 19.0, 24.5, 28.5):
        rf.box((x, D / 2, HT - 0.45), (0.12, D / 2, 0.45), M["grey_steel"])
        for k in range(10):  # web members
            y = 1.0 + k * 2.0
            rf.tube((x, y, HT - 0.9), (x, y + 1.0, HT - 0.05), 0.025, M["grey_steel"], segs=6)
            rf.tube((x, y + 1.0, HT - 0.05), (x, y + 2.0, HT - 0.9), 0.025, M["grey_steel"], segs=6)
    for y in (2.0, 10.0, 18.0):
        rf.box((W / 2, y, HT - 0.1), (W / 2, 0.05, 0.08), M["grey_steel"])
    mk(rf, "roof", "roof")
    # slab (floor group: also the base of the web floor bake)
    sb = H.MeshBuilder()
    sb.box((W / 2, D / 2, -0.15), (W / 2 + WT, D / 2 + WT, 0.15), M["slab"])
    mk(sb, "slab", "floor", None)
    # exterior ground
    gb = H.MeshBuilder()
    fquad(gb, -40, -40, 70, 60, -0.03, M["ground"])
    mk(gb, "ground_out", "render_only", None)


def doors(M):
    for d in LAY["doors"]:
        r = d["rect"]
        if d["type"] == "roll_up":
            x0, x1, h = r[0], r[2], d["height"]
            w = x1 - x0
            mb = H.MeshBuilder()
            for s in (x0 - 0.1, x1 + 0.1):  # guide rails
                mb.box((s, 0.08, h / 2 + 0.2), (0.07, 0.08, h / 2 + 0.2), M["dark"])
            mb.box(((x0 + x1) / 2, 0.16, h + 0.25), (w / 2 + 0.2, 0.16, 0.25), M["dark"])  # coil housing
            for b in LAY.get("bollards", []):  # layout bollards
                mb.cylinder((b["x"], b["y"], b["height"] / 2), b["r"], b["height"], M["yellow"], segs=12)
            mk(mb, "rollup_frame", "roof")  # rails and coil up the wall: renders only (web walls stop at WEB_CUT)
            for mode, openf in (("day", d["open_fraction"]["day"]), ("night", d["open_fraction"]["night"])):
                cb = H.MeshBuilder()
                z_low = h * openf
                n = int((h - z_low) / 0.075)
                for k in range(n):
                    cb.box(((x0 + x1) / 2, 0.05, h - 0.0375 - k * 0.075), (w / 2, 0.022, 0.034), M["door_metal"])
                cb.box(((x0 + x1) / 2, 0.05, z_low + 0.07), (w / 2, 0.035, 0.07), M["yellow"])
                mk(cb, f"rollup_curtain_{mode}", f"door_{mode}")
        else:
            wall = d["wall"][0].upper()
            a0, a1 = (r[0], r[2]) if wall in "SN" else (r[1], r[3])
            mb = H.MeshBuilder()
            c = (a0 + a1) / 2
            h = d["height"]
            if wall == "W":
                mb.box((0.04, c, h / 2), (0.04, (a1 - a0) / 2 - 0.02, h / 2 - 0.02), M["leaf"])
                mb.box((0.1, a1 - 0.15, 1.0), (0.025, 0.12, 0.02), M["galv"])  # push bar
                ex_pos = (0.05, c, h + 0.35)
            else:
                mb.box((W - 0.04, c, h / 2), (0.04, (a1 - a0) / 2 - 0.02, h / 2 - 0.02), M["leaf"])
                mb.box((W - 0.1, a0 + 0.15, 1.0), (0.025, 0.12, 0.02), M["galv"])
                ex_pos = (W - 0.05, c, h + 0.35)
            mk(mb, f"door_{d['id']}", "roof")
            eb = H.MeshBuilder()
            eb.box(ex_pos, (0.03, 0.22, 0.1), M["exit"])
            mk(eb, f"exit_sign_{d['id']}", "roof")
            t = H.text(f"EXIT_{d['id']}", "EXIT", 0.11, M["text_k"], ROOT,
                       (ex_pos[0] + (0.035 if wall == "W" else -0.035), ex_pos[1], ex_pos[2]),
                       (90, 0, 90 if wall == "W" else -90), 0.004)
            t["grp"] = "roof"


# ---- racking, pallets, loads -------------------------------------------------------------------


class Protos:
    carton = tote = None


def pallet(mb, cx, cy, z, M, rng, size=(1.2, 1.0, 0.144)):
    """Wooden pallet (GMA 1.2 x 1.0 x 0.144 by default; the arm's half pallet is 0.8 x 0.6 x 0.14), bottom at z."""
    mat = M["wood_a"] if rng.random() < 0.6 else M["wood_b"]
    sx, sy, sz = size
    hb = (sz - 0.025) / 2  # block height / 2
    for dy in (-(sy / 2 - 0.05), 0.0, sy / 2 - 0.05):
        mb.box((cx, cy + dy, z + 0.0125 + hb), (sx / 2, 0.045, hb), mat)
    nb = max(4, round(sx / 0.173))
    for k in range(nb):
        mb.box((cx - sx / 2 + 0.08 + k * (sx - 0.16) / (nb - 1), cy, z + sz - 0.0125), (0.07, sy / 2, 0.0125), mat)
    for k in range(3):
        mb.box((cx - sx / 2 + 0.1 + k * (sx - 0.2) / 2, cy, z + 0.0125), (0.07, sy / 2, 0.0125), mat)


def carton_stack(cx, cy, z0, h_target, rng, M, proxy_mb):
    """Cartons on a 1.2 x 1.0 pallet footprint: PH instances for renders, one box proxy for the web."""
    cs = Protos.carton_size
    ch = cs[2]
    long_, short = max(cs[0], cs[1]), min(cs[0], cs[1])
    nx, ny = int(1.2 // long_), int(1.0 // short)
    layers = max(1, int(h_target // ch))
    gx, gy = nx * long_, ny * short
    full_layers = layers
    for ly in range(layers):
        cnt = nx * ny if ly < layers - 1 or rng.random() < 0.7 else rng.randint(1, nx * ny)
        slots = [(i, j) for i in range(nx) for j in range(ny)]
        rng.shuffle(slots)
        for i, j in slots[:cnt]:
            x = cx - gx / 2 + (i + 0.5) * long_ + rng.uniform(-0.01, 0.01)
            y = cy - gy / 2 + (j + 0.5) * short + rng.uniform(-0.01, 0.01)
            yaw = (90 if cs[1] > cs[0] else 0) + (180 if rng.random() < 0.5 else 0) + rng.uniform(-1.5, 1.5)
            instance(Protos.carton, (x, y, z0 + ly * ch), yaw, name="carton")
    proxy_mb.box((cx, cy, z0 + full_layers * ch / 2), (gx / 2, gy / 2, full_layers * ch / 2), M["carton"] if rng.random() < 0.6 else M["carton_lite"], bevel=0.01, segments=1)
    return full_layers * ch


def tote_stack(cx, cy, z0, h_target, rng, M, proxy_mb):
    ts = Protos.tote_size
    nx, ny = int(1.2 // ts[0]), int(1.0 // ts[1])
    layers = max(1, int(h_target // ts[2]))
    gx, gy = nx * ts[0], ny * ts[1]
    for ly in range(layers):
        for i in range(nx):
            for j in range(ny):
                instance(Protos.tote, (cx - gx / 2 + (i + 0.5) * ts[0], cy - gy / 2 + (j + 0.5) * ts[1], z0 + ly * ts[2]),
                         rng.uniform(-1.0, 1.0), name="tote")
    mat = rng.choice([M["tote_blue"], M["tote_grey"], M["tote_green"]])
    proxy_mb.box((cx, cy, z0 + layers * ts[2] / 2), (gx / 2, gy / 2, layers * ts[2] / 2), mat, bevel=0.01, segments=1)
    return layers * ts[2]


def wrapped_stack(cx, cy, z0, h_target, rng, M, proxy_mb, wrap_mb):
    h = carton_stack(cx, cy, z0, h_target, rng, M, H.MeshBuilder())  # cartons inside (render only), proxy discarded
    wrap_mb.box((cx, cy, z0 + h / 2), (0.62, 0.52, h / 2 + 0.005), M["wrap"])
    proxy_mb.box((cx, cy, z0 + h / 2), (0.62, 0.52, h / 2 + 0.005), M["wrap_proxy"], bevel=0.012, segments=1)
    return h


def put_load(kind, cx, cy, z0, h_target, rng, M, mbs):
    pm, wm = mbs["proxy"], mbs["wrap"]
    if kind == "cartons":
        return carton_stack(cx, cy, z0, h_target, rng, M, pm)
    if kind == "totes":
        return tote_stack(cx, cy, z0, h_target, rng, M, pm)
    if kind == "wrapped":
        return wrapped_stack(cx, cy, z0, h_target, rng, M, pm, wm)
    return 0.0


def racking(M):
    blue, orange, labels, pal, plate = H.MeshBuilder(), H.MeshBuilder(), H.MeshBuilder(), H.MeshBuilder(), H.MeshBuilder()
    guards, decks = H.MeshBuilder(), H.MeshBuilder()
    mbs = {"proxy": H.MeshBuilder(), "wrap": H.MeshBuilder()}
    kinds = ["cartons"] * 5 + ["totes"] * 2 + ["wrapped"] * 3
    fill = {1: 0.88, 2: 0.72, 3: 0.55}
    for sid, sh in LAY["shelves"].items():
        x0, y0, x1, y1 = sh["rect"]
        bay = sh["bay_length"]
        n = int(round((x1 - x0) / bay))
        fh = sh["frame_height"]
        face_south = sh["pick_side"] == "south"
        for k in range(n + 1):
            x = x0 + k * bay
            ys = (y0 + 0.04, y1 - 0.04)
            for y in ys:
                blue.box((x, y, fh / 2), (0.04, 0.04, fh / 2), M["blue"])
            for hz in (0.3, 1.5, 2.7, 3.9, 4.9):
                blue.tube((x, ys[0], hz), (x, ys[1], hz), 0.012, M["blue"], segs=6)
            for i, hz in enumerate((0.3, 1.5, 2.7, 3.9)):
                a, b = (ys[0], ys[1]) if i % 2 == 0 else (ys[1], ys[0])
                blue.tube((x, a, hz), (x, b, hz + 1.2), 0.012, M["blue"], segs=6)
            if k in (0, n):  # column guards
                for y in ys:
                    guards.box((x, y, 0.3), (0.12, 0.12, 0.3), M["yellow"])
        pick_bay = sh["pick_bay"]["bay"]
        for lvl, bz in enumerate(sh["beam_z"]):
            for bayi in range(n):
                if bayi + 1 == pick_bay and lvl == 0:
                    continue  # the pick bay's ground level has tote decks instead
                cx = x0 + (bayi + 0.5) * bay
                for y in (y0 + 0.04, y1 - 0.04):
                    orange.box((cx, y, bz - 0.06), (bay / 2 - 0.04, 0.025, 0.06), M["orange"])
        # pick bay: two wire decks on their own beams, a place sign on the aisle side
        cx = x0 + (pick_bay - 0.5) * bay
        for dz in sh["pick_bay"]["deck_z"]:
            for y in (y0 + 0.04, y1 - 0.04):
                orange.box((cx, y, dz - 0.05), (bay / 2 - 0.04, 0.025, 0.045), M["orange"])
            for k in range(9):
                decks.box((cx - bay / 2 + 0.1 + k * (bay - 0.2) / 8, (y0 + y1) / 2, dz - 0.012),
                          (0.012, (y1 - y0) / 2 - 0.05, 0.012), M["galv"])
            for j in range(12):
                decks.box((cx, y0 + 0.08 + j * (y1 - y0 - 0.16) / 11, dz - 0.002), (bay / 2 - 0.06, 0.003, 0.003),
                          M["galv"])
        fy = y0 - 0.006 if face_south else y1 + 0.006
        plate.box((cx, fy, 1.32), (0.32, 0.006, 0.11), M["sign_blue"])
        t = H.text(f"pick_{sid}", f"SHELF {sid}", 0.12, M["text_w"], ROOT, (cx, fy - 0.008 if face_south else fy + 0.008, 1.27),
                   (90, 0, 0 if face_south else 180), 0.002)
        t["grp"] = "interior"
        for b in sh["bins"]:
            bx, by, bz = b["center"]
            lvl = b["level"]
            fy = y0 + 0.012 if face_south else y1 - 0.012
            labels.box((bx, fy - 0.003 if face_south else fy + 0.003, bz - 0.06), (0.16, 0.002, 0.035), M["label"])
            t = H.text(f"lbl_{b['id']}", b["id"], 0.04, M["text_k"], ROOT,
                       (bx, fy - 0.006 if face_south else fy + 0.006, bz - 0.06), (90, 0, 0 if face_south else 180), 0.0005)
            t["grp"] = "render_only"
            for px in (bx - 0.64, bx + 0.64):
                if RNG.random() > fill[lvl]:
                    continue
                pallet(pal, px, by, bz, M, RNG)
                kind = RNG.choice(kinds)
                put_load(kind, px, by, bz + 0.144, RNG.uniform(0.7, 1.3) if lvl < 3 else RNG.uniform(0.6, 1.2), RNG, M, mbs)
        # row letter plate on the east end frame, facing +x; capacity plate near it
        plate.box((x1 + 0.03, (y0 + y1) / 2, 4.3), (0.012, 0.26, 0.26), M["sign_blue"])
        t = H.text(f"row_{sid}", sid, 0.38, M["text_w"], ROOT, (x1 + 0.045, (y0 + y1) / 2, 4.3), (90, 0, 90), 0.003)
        t["grp"] = "interior"
        plate.box((x1 + 0.03, (y0 + y1) / 2, 1.9), (0.012, 0.2, 0.12), M["sign_yel"])
        t = H.text(f"cap_{sid}", "MAX 1000 KG\nPER LEVEL", 0.05, M["text_k"], ROOT, (x1 + 0.045, (y0 + y1) / 2, 1.9),
                   (90, 0, 90), 0.002)
        t["grp"] = "interior"
    mk(blue, "racks_frames", "interior")
    mk(orange, "racks_beams", "interior")
    mk(labels, "bin_labels", "interior")
    mk(pal, "pallets_rack", "interior")
    mk(guards, "column_guards", "interior")
    mk(decks, "pick_decks", "interior")
    mk(plate, "rack_plates", "interior")
    return mbs


# ---- cell equipment ---------------------------------------------------------------------------


def staging(M, mbs):
    pal = H.MeshBuilder()
    for i, p in enumerate(LAY["pallets"]):
        cx, cy = p["center"]
        pallet(pal, cx, cy, 0.0, M, RNG, tuple(p["size"]))
        hl = p["load_height"]
        if hl > 0:
            put_load(["wrapped", "cartons"][i % 2], cx, cy, p["size"][2], hl, RNG, M, mbs)
    mk(pal, "pallets_staging", "interior")


def totes(M):
    """The sim's totes (cell.py initial slots, or --state): Poly Haven crates scaled to the sim tote."""
    ts = Protos.tote_size
    sx, sy, sz = CELL.TOTE_SIZE
    scale = (sx / ts[0], sy / ts[1], sz / ts[2])
    poses = {tid: list(v) for tid, v in CELL.initial_poses().items()}
    poses.update({tid: list(v) for tid, v in STATE.get("totes", {}).items()})
    for tid, (x, y, zc, yaw) in poses.items():
        if tid == STATE.get("g1Holding") and STATE.get("g1"):
            gx, gy, gyaw = STATE["g1"]
            x, y, zc, yaw = gx + 0.32 * math.cos(gyaw), gy + 0.32 * math.sin(gyaw), 0.78, gyaw
        elif tid == STATE.get("armHolding") and STATE.get("armTip"):
            x, y, tz = STATE["armTip"]
            zc = tz - sz / 2
        instance(Protos.tote, (x, y, zc - sz / 2), math.degrees(yaw), scale, name=tid)


def conveyor(M, mbs):
    c = LAY["conveyor"]
    x0, y0, x1, y1 = c["rect"]
    z = c["top_z"]
    mb = H.MeshBuilder()
    for y in (y0 + 0.03, y1 - 0.03):
        mb.box(((x0 + x1) / 2, y, z - 0.07), ((x1 - x0) / 2, 0.03, 0.07), M["galv"])
    for x in np.arange(x0 + 0.1, x1, 0.12):
        mb.cylinder((x, (y0 + y1) / 2, z - 0.03), 0.025, y1 - y0 - 0.08, M["roller"], axis="y", segs=8)
    for x in np.arange(x0 + 0.3, x1, 1.2):
        for y in (y0 + 0.04, y1 - 0.04):
            mb.box((x, y, (z - 0.14) / 2), (0.025, 0.025, (z - 0.14) / 2), M["grey_steel"])
        mb.box((x, (y0 + y1) / 2, 0.25), (0.015, (y1 - y0) / 2 - 0.04, 0.015), M["grey_steel"])
    mb.box((x1 - 0.45, (y0 + y1) / 2, z - 0.32), (0.2, 0.14, 0.12), M["motor"])  # drive unit under the belt
    mb.box((x1 - 0.05, (y0 + y1) / 2, z + 0.1), (0.02, (y1 - y0) / 2, 0.1), M["yellow"])  # end stop
    mk(mb, "conveyor", "interior")  # its totes are the sim's (totes())


def pick_table(M, mbs):
    t = LAY["pick_table"]
    x0, y0, x1, y1 = t["rect"]
    z = t["top_z"]
    mb = H.MeshBuilder()
    mb.box(((x0 + x1) / 2, (y0 + y1) / 2, z - 0.02), ((x1 - x0) / 2, (y1 - y0) / 2, 0.02), M["ply"], bevel=0.003)
    for x in (x0 + 0.06, x1 - 0.06):
        for y in (y0 + 0.06, y1 - 0.06):
            mb.box((x, y, (z - 0.04) / 2), (0.025, 0.025, (z - 0.04) / 2), M["grey_steel"])
    mb.box(((x0 + x1) / 2, (y0 + y1) / 2, 0.25), ((x1 - x0) / 2 - 0.06, (y1 - y0) / 2 - 0.06, 0.012), M["galv"])
    for y in (y0 + 0.06, y1 - 0.06):
        mb.box(((x0 + x1) / 2, y, z - 0.12), ((x1 - x0) / 2 - 0.06, 0.02, 0.05), M["grey_steel"])
    mk(mb, "pick_table", "interior")  # its totes are the sim's (totes())


def arm_cell(M):
    a = LAY["arm_cell"]
    cx, cy = a["pedestal_center"]
    h = a["pedestal_height"]
    hx, hy = a["pedestal_size"][0] / 2, a["pedestal_size"][1] / 2
    mb = H.MeshBuilder()
    mb.box((cx, cy, 0.01), (hx + 0.05, hy + 0.05, 0.01), M["dark"])
    mb.box((cx, cy, h / 2), (hx - 0.02, hy - 0.02, h / 2 - 0.02), M["grey_steel"], bevel=0.004)
    mb.box((cx, cy, h - 0.01), (hx, hy, 0.01), M["dark"])
    mk(mb, "arm_pedestal", "interior")
    # keep-out hazard band + a light curtain post pair
    hb = H.MeshBuilder()
    x0, y0, x1, y1 = LAY["markings"]["arm_cell_keepout"]
    bw = 0.2
    fquad(hb, x0, y0, x1, y0 + bw, 0.003, M["hazard"])
    fquad(hb, x0, y1 - bw, x1, y1, 0.003, M["hazard"])
    _, cy0, _, cy1 = LAY["conveyor"]["rect"]
    fquad(hb, x0, y0 + bw, x0 + bw, cy0 - 0.05, 0.003, M["hazard"])  # west band, open where the conveyor comes in
    fquad(hb, x0, cy1 + 0.05, x0 + bw, y1 - bw, 0.003, M["hazard"])
    fquad(hb, x1 - bw, y0 + bw, x1, y1 - bw, 0.003, M["hazard"])
    mk(hb, "keepout_hazard", "floor", None)
    pb = H.MeshBuilder()
    for (x, y) in ((x0 - 0.1, y0 - 0.1), (x1 + 0.1, y1 + 0.1)):
        pb.box((x, y, 0.55), (0.04, 0.04, 0.55), M["yellow"])
        pb.box((x, y, 1.12), (0.05, 0.05, 0.04), M["status_led"])
    mk(pb, "keepout_posts", "interior")


def charging(M):
    c = LAY["charging"]
    x0, y0, x1, y1 = c["rect"]
    mb = H.MeshBuilder()
    mb.box((0.18, c["center"][1], 0.7), (0.15, 0.2, 0.7), M["white_plastic"], bevel=0.01)
    mb.box((0.34, c["center"][1], 1.0), (0.01, 0.05, 0.1), M["status_led"])
    mb.box((0.32, c["center"][1], 0.15), (0.12, 0.3, 0.15), M["dark"])  # charge pad
    mk(mb, "charging_dock", "interior")


APRON = (23.6, 0.9, 28.9, 2.2)  # dock apron (floor tape), inside the roll-up door


def floor_marks(M):
    mb = H.MeshBuilder()
    for aid, off in (("aisle-AB", 0.2), ("aisle-CD", 0.2)):
        a = next(x for x in LAY["aisles"] if x["id"] == aid)
        x0, y0, x1, y1 = a["rect"]
        line(mb, x0, y0 + off, x1, y0 + off, 0.1, M["paint_y"])
        line(mb, x0, y1 - off, x1, y1 - off, 0.1, M["paint_y"])
    line(mb, 2.0, 9.9, 22.5, 9.9, 0.1, M["paint_y"])
    line(mb, 2.0, 12.8, 22.5, 12.8, 0.1, M["paint_y"])
    line(mb, 2.0, 4.8, 21.5, 4.8, 0.1, M["paint_y"])
    line(mb, 21.1, 0.8, 21.1, 19.4, 0.1, M["paint_y"])
    # green walkway along the north aisle
    line(mb, 2.0, 17.9, 29.4, 17.9, 0.12, M["paint_g"])
    line(mb, 2.0, 19.4, 29.4, 19.4, 0.12, M["paint_g"])
    # staging area and pallet bays
    rect_outline(mb, 22.5, 2.3, 28.2, 4.9, 0.1, M["paint_w"])
    for p in LAY["pallets"]:
        cx, cy = p["center"]
        hx, hy = p["size"][0] / 2 + 0.15, p["size"][1] / 2 + 0.15
        rect_outline(mb, cx - hx, cy - hy, cx + hx, cy + hy, 0.05, M["paint_w"])
    rect_outline(mb, 0.4, 10.4, 1.6, 11.6, 0.08, M["paint_g"])
    rect_outline(mb, *APRON, 0.08, M["paint_y"])  # dock apron
    # tyre marks and oil
    for (x0, y0, x1, y1) in ((2.5, 11.0, 22.0, 11.0), (2.5, 11.9, 22.0, 11.9), (3, 7.0, 20, 7.0), (3, 7.6, 20, 7.6),
                             (2.8, 15.2, 20, 15.2), (22.0, 1.4, 22.0, 18.0), (21.6, 1.4, 21.6, 18.0)):
        line(mb, x0, y0 + RNG.uniform(-0.05, 0.05), x1, y1 + RNG.uniform(-0.05, 0.05), 0.2, M["skid"], z=0.002)
    for _ in range(14):
        x, y = RNG.uniform(3, 28), RNG.uniform(1.5, 18.5)
        r = RNG.uniform(0.08, 0.25)
        mb.cylinder((x, y, 0.0025), r, 0.002, M["oil"], segs=10)
    mk(mb, "floor_marks", "floor", None)
    for body, loc, size, rot in (("A | B", (19.4, 15.35, 0.004), 0.55, 0), ("C | D", (19.4, 7.35, 0.004), 0.55, 0),
                                 ("STAGING", (25.6, 5.4, 0.004), 0.5, 0), ("DOCK 1", (24.55, 1.55, 0.004), 0.45, 0),
                                 ("CHARGE", (1.0, 9.95, 0.004), 0.25, 0), ("WALKWAY", (15.0, 18.65, 0.004), 0.3, 0)):
        t = H.text("floor_" + body, body, size, M["paint_w"] if body != "WALKWAY" else M["paint_g"], ROOT, loc, (0, 0, rot), 0.001)
        t["grp"] = "floor"


def signage(M):
    wall_sign(M, "N", 9.0, 2.4, 1.6, 0.8, "sign_yel", "CAUTION\nFORKLIFT TRAFFIC", "text_k", 0.11)
    wall_sign(M, "N", 20.0, 2.4, 0.9, 0.7, "sign_blue", "PPE\nREQUIRED", "text_w", 0.12)
    wall_sign(M, "N", 14.5, 2.4, 0.9, 0.7, "sign_grn", "FIRST AID", "text_w", 0.1)
    wall_sign(M, "E", 12.6, 2.2, 1.7, 0.85, "sign_yel", "ROBOT WORK CELL\nAUTHORISED ONLY", "text_k", 0.1)
    wall_sign(M, "E", 3.2, 2.4, 0.9, 0.6, "sign_red", "NO\nSMOKING", "text_w", 0.1)
    wall_sign(M, "S", 17.0, 2.2, 1.4, 0.7, "sign_blue", "DOCK DOOR 1\nKEEP CLEAR", "text_w", 0.1)
    wall_sign(M, "W", 4.0, 2.4, 1.2, 0.6, "sign_grn", "ASSEMBLY\nPOINT", "text_w", 0.1)
    wall_sign(M, "S", 8.0, 2.2, 1.2, 0.5, "sign_red", "FIRE EXIT\nKEEP CLEAR", "text_w", 0.09)
    # fire equipment: Poly Haven extinguishers and pull stations on the pilasters
    me, size = load_proto("korean_fire_extinguisher_01")
    fb = H.MeshBuilder()
    for (x, y, yaw) in ((9.0, 0.32, 0), (15.0, 0.32, 0), (9.0, D - 0.32, 180), (21.0, D - 0.32, 180)):
        o = instance(me, (x, y, 0.0), yaw, name="extinguisher")
        o.scale = (0.9, 0.9, 0.9)
        o["grp"] = "interior"  # small enough to keep in the web bake
        sgn = 1 if y < 5 else -1
        fb.box((x, y + sgn * 0.17, 1.9), (0.22, 0.01, 0.2), M["sign_red"])
    mk(fb, "extinguisher_signs", "roof")
    fa, _ = load_proto("fire_alarm")
    for (x, y, yaw) in ((9.0, 0.32, 0), (21.0, D - 0.32, 180)):
        o = instance(fa, (x + 0.45, y, 1.3), yaw, name="fire_alarm")
        o["grp"] = "roof"
    ws, _ = load_proto("WetFloorSign_01")
    o = instance(ws, (22.4, 1.0, 0.0), 30, (0.9, 0.9, 0.9), name="wet_sign")
    o["grp"] = "interior"


def overhead(M):
    lt = LAY["lights"]["high_bay_grid"]
    lc = col("lights")
    hb, lens_a, lens_b = H.MeshBuilder(), H.MeshBuilder(), H.MeshBuilder()
    idx = 0
    for yi, y in enumerate(lt["y"]):
        for xi, x in enumerate(lt["x"]):
            z = lt["z"]
            on_night = (xi + yi) % 2 == 0
            hb.box((x, y, z + 0.09), (0.66, 0.16, 0.05), M["dark"])
            for dx in (-0.5, 0.5):
                hb.tube((x + dx, y, z + 0.1), (x + dx, y, HT - 0.45), 0.006, M["dark"], segs=4)
            (lens_a if on_night else lens_b).box((x, y, z + 0.02), (0.6, 0.12, 0.02), M["led_a"] if on_night else M["led_b"])
            ld = bpy.data.lights.new(f"hb_{idx}", "AREA")
            ld.shape = "RECTANGLE"
            ld.size, ld.size_y = 1.2, 0.25
            ld.spread = math.radians(130)
            ld.color = (1.0, 0.95, 0.88)
            o = bpy.data.objects.new(f"hb_{idx}", ld)
            o.location = (x, y, z - 0.01)
            o["w_day"] = 150.0
            o["w_night"] = 55.0 if on_night else 0.0
            lc.objects.link(o)
            idx += 1
    mk(hb, "highbay_bodies", "interior")
    mk(lens_a, "highbay_lens_a", "interior")
    mk(lens_b, "highbay_lens_b", "interior")
    # skylights (day only): emissive roof panels with matching area lights
    sk = H.MeshBuilder()
    for i, (x, y) in enumerate([(10.75, 5.0), (10.75, 15.0), (21.75, 5.0), (21.75, 15.0), (16.25, 10.0), (27.0, 10.0)]):
        fquad(sk, x - 1.2, y - 1.5, x + 1.2, y + 1.5, HT - 0.03, M["skylight"])
        ld = bpy.data.lights.new(f"sky_{i}", "AREA")
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = 2.4, 3.0
        ld.spread = math.radians(160)
        ld.color = (0.85, 0.92, 1.0)
        o = bpy.data.objects.new(f"sky_{i}", ld)
        o.location = (x, y, HT - 0.04)
        o["w_day"] = 260.0
        o["w_night"] = 0.0
        lc.objects.link(o)
    mk(sk, "skylights", "roof", None)
    # day sun through the open roll-up door
    sd = bpy.data.lights.new("sun", "SUN")
    sd.angle = math.radians(1.0)
    so = bpy.data.objects.new("sun", sd)
    so.rotation_euler = Vector((0.22, 0.9, -0.5)).to_track_quat("-Z", "Y").to_euler()
    so["w_day"] = 6.0
    so["w_night"] = 0.0
    lc.objects.link(so)
    # services: duct, sprinkler pipes, cable tray
    sv = H.MeshBuilder()
    sv.cylinder((15.0, 11.3, 7.0), 0.3, 27.0, M["duct"], axis="x", segs=20)
    for y in (3.0, 7.4, 11.8, 15.3, 18.8):
        sv.cylinder((15.0, y, 7.3), 0.04, 28.0, M["red_pipe"], axis="x", segs=8)
    sv.cylinder((1.0, 10.0, 7.3), 0.08, 19.0, M["red_pipe"], axis="y", segs=10)
    sv.box((21.7, 10.0, 6.4), (0.15, 9.0, 0.03), M["galv"])
    mk(sv, "services", "roof")  # overhead: in renders and the bake, not on the map (it hid the arm cell from above)


def cameras(M):
    cc = col("cameras")
    mb = H.MeshBuilder()
    for name, c in LAY["cameras"].items():
        mount, target, lens = Vector(c["mount"]), Vector(c["target"]), c["lens_mm"]
        cd = bpy.data.cameras.new(name)
        hfov, _ = CM.source_fov(lens)
        cd.sensor_fit = "HORIZONTAL"
        cd.angle = hfov
        cd.clip_start, cd.clip_end = 0.1, 200
        o = bpy.data.objects.new(name, cd)
        d = (target - mount).normalized()
        o.location = mount + d * 0.3
        o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        o["lens_mm"] = lens
        cc.objects.link(o)
        # housing: bracket to the nearest wall, bullet body along the view direction
        wx = 0.0 if mount.x < W / 2 else W
        wy = 0.0 if mount.y < D / 2 else D
        mb.tube((wx, mount.y, mount.z), (mount.x, mount.y, mount.z), 0.025, M["camera"], segs=8)
        mb.tube((mount.x, wy, mount.z), (mount.x, mount.y, mount.z), 0.025, M["camera"], segs=8)
        rot = d.to_track_quat("Y", "Z").to_matrix().to_4x4()
        m = Matrix.Translation(mount + d * 0.1) @ rot
        mb.box((0, 0, 0), (0.05, 0.13, 0.05), M["camera"], matrix=m, bevel=0.02)
        mb.box((0, 0.02, 0.06), (0.065, 0.16, 0.008), M["camera"], matrix=m)
        mb.cylinder((0, 0.131, 0), 0.032, 0.004, M["glass"], axis="y", matrix=m)
    mk(mb, "camera_housings", "roof")  # wall brackets at 6 m: the cutaway web GLB has no wall up there


# ---- robots -----------------------------------------------------------------------------------


def robots(M, npz=None):
    data = np.load(npz or H.CACHE / "robots.npz")
    names = sorted({k.split("/")[0] for k in data.files})

    def build(robot, loc, yaw, grp, decim_tris=None):
        ks = sorted({int(k.split("/")[1]) for k in data.files if k.startswith(robot + "/")})
        groups = {}
        for k in ks:
            rgba = tuple(round(float(x), 3) for x in data[f"{robot}/{k}/rgba"])
            groups.setdefault(rgba, []).append(k)
        total = sum(len(data[f"{robot}/{k}/f"]) for k in ks)
        out = []
        for rgba, klist in groups.items():
            vs, fs, off = [], [], 0
            for k in klist:
                v, f = data[f"{robot}/{k}/v"], data[f"{robot}/{k}/f"]
                vs.append(v)
                fs.append(f + off)
                off += len(v)
            v, f = np.concatenate(vs), np.concatenate(fs)
            me = bpy.data.meshes.new(f"{robot}_{len(out)}")
            me.from_pydata(v.tolist(), [], f.tolist())
            me.update()
            H.smooth(me, 40)
            name = f"rob_{robot}_{rgba}"
            mat = bpy.data.materials.get(name) or H.principled(name, rgba[:3], rough=0.45, metal=0.15 if rgba[3] >= 0.99 else 0.0)
            me.materials.append(mat)
            o = bpy.data.objects.new(f"{robot}_{len(out)}", me)
            o.location = loc
            o.rotation_euler = (0, 0, math.radians(yaw))
            o["grp"] = grp
            ROOT.objects.link(o)
            if decim_tris:
                md = o.modifiers.new("decim", "DECIMATE")
                md.ratio = min(1.0, decim_tris / max(total, 1))
            out.append(o)
        return total

    gx, gy, gyaw = STATE.get("g1") or CELL.HUMANOID_HOME
    a = LAY["arm_cell"]
    t1 = build("g1", (gx, gy, 0.0), math.degrees(gyaw), "render_only")  # the live map draws the G1 itself
    t2 = build("iiwa", (a["pedestal_center"][0], a["pedestal_center"][1], a["pedestal_height"]), a["base_yaw_deg"], "render_only")
    build("iiwa", (a["pedestal_center"][0], a["pedestal_center"][1], a["pedestal_height"]), a["base_yaw_deg"], "web_only", 20000)
    t3 = 0
    if "go2" in names:  # robots_fk.py dumps it when the Menagerie Go2 is in the cache; default: the fleet's start
        qx, qy, qyaw = STATE.get("go2") or (3.0, 11.4, 0.0)
        t3 = build("go2", (qx, qy, 0.0), math.degrees(qyaw), "render_only")
    print(f"ROBOTS g1 {t1} tris, iiwa {t2} tris, go2 {t3} tris", flush=True)


# ---- world -------------------------------------------------------------------------------------


def world():
    w = bpy.data.worlds.new("wh_world")
    w.use_nodes = True
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.name = "env"
    day = bpy.data.images.load(str(H.CACHE / "kloofendal_overcast_puresky" / "kloofendal_overcast_puresky_1k.hdr"))
    night = bpy.data.images.load(str(H.CACHE / "kloppenheim_02_puresky" / "kloppenheim_02_puresky_1k.hdr"))
    env.image = night
    day.use_fake_user = night.use_fake_user = True  # the unused one must survive save/load
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.name = "Background"
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    w["day_hdri"], w["night_hdri"] = day.name, night.name
    bpy.context.scene.world = w


def main() -> None:
    global ROOT
    argv = sys.argv[sys.argv.index("--") + 1:]
    out = Path(argv[argv.index("--out") + 1])
    if "--state" in argv:
        STATE.update(json.loads(Path(argv[argv.index("--state") + 1]).read_text()))
    npz = Path(argv[argv.index("--robots") + 1]) if "--robots" in argv else None
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ROOT = bpy.context.scene.collection
    M = materials()
    world()
    Protos.carton, Protos.carton_size = load_proto("cardboard_box_01")
    Protos.tote, Protos.tote_size = load_proto("plastic_crate_02")
    shell(M)
    doors(M)
    mbs = racking(M)
    staging(M, mbs)
    conveyor(M, mbs)
    pick_table(M, mbs)
    arm_cell(M)
    charging(M)
    floor_marks(M)
    signage(M)
    overhead(M)
    cameras(M)
    totes(M)
    robots(M, npz)
    mk(mbs["proxy"], "load_proxies", "web_only")
    mk(mbs["wrap"], "stretch_wrap", "render_only")
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print("BUILD_OK", out, "objects", len(bpy.data.objects), flush=True)


if __name__ == "__main__":
    main()
