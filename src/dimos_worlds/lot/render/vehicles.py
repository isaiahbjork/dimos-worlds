"""Procedural vehicles at real dimensions (layout.VEHICLE_DIMS). Runs inside Blender.

Why procedural: no CC0 vehicle set we can fetch without an account is at real proportions (Kenney/Quaternius
cars are toy-proportioned, about 2.5 m long). These are built from side profiles with wheel arches, a
tapered greenhouse, glass, clear-coated paint, tyres and rims, lamps, plates and mirrors. The one
asset vehicle is Poly Haven's covered_car (CC0).

Every vehicle is one mesh object named like its layout entry (veh-14, veh-x-box), origin on the ground at
the body centre, nose along +y before its yaw.
"""

from __future__ import annotations

import math
import random

import bpy
from mathutils import Matrix

import bhelp as H


def paint_material(paint_rgb, metallic: float, rough: float) -> bpy.types.Material:
    """Clear-coated paint with road grime on the lower body and light dust (object-space height gradient)."""
    key = "paint_%.3f_%.3f_%.3f_%d" % (*paint_rgb, int(metallic > 0))
    m = bpy.data.materials.get(key)
    if m:
        return m
    m = H.principled(key, paint_rgb, rough=rough, metal=metallic, coat=1.0, coat_rough=0.05)
    nt = m.node_tree
    n, ln = nt.nodes, nt.links
    b = n.get("Principled BSDF")
    tc = n.new("ShaderNodeTexCoord")
    sep = n.new("ShaderNodeSeparateXYZ")
    ln.new(tc.outputs["Object"], sep.inputs["Vector"])
    noise = n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.0
    noise.inputs["Detail"].default_value = 8.0
    ln.new(tc.outputs["Object"], noise.inputs["Vector"])
    # grime factor: 1 at the rocker (z < 0.35 m), fading out by 0.75 m, broken up by noise
    mr = n.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0.75
    mr.inputs["From Max"].default_value = 0.3
    ln.new(sep.outputs["Z"], mr.inputs["Value"])
    mul = n.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    ln.new(mr.outputs["Result"], mul.inputs[0])
    ln.new(noise.outputs["Fac"], mul.inputs[1])
    mul2 = n.new("ShaderNodeMath")
    mul2.operation = "MULTIPLY"
    mul2.use_clamp = True
    mul2.inputs[1].default_value = 1.4
    ln.new(mul.outputs[0], mul2.inputs[0])
    mix = n.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    ln.new(mul2.outputs[0], mix.inputs["Factor"])
    mix.inputs[6].default_value = (*paint_rgb, 1)
    mix.inputs[7].default_value = (0.09, 0.08, 0.07, 1)
    ln.new(mix.outputs[2], b.inputs["Base Color"])
    rmix = n.new("ShaderNodeMapRange")
    rmix.inputs["To Min"].default_value = rough
    rmix.inputs["To Max"].default_value = 0.85
    ln.new(mul2.outputs[0], rmix.inputs["Value"])
    ln.new(rmix.outputs["Result"], b.inputs["Roughness"])
    cr = n.new("ShaderNodeMapRange")
    cr.inputs["To Min"].default_value = 0.04
    cr.inputs["To Max"].default_value = 0.6
    ln.new(mul2.outputs[0], cr.inputs["Value"])
    ln.new(cr.outputs["Result"], b.inputs["Coat Roughness"])
    return m


def _mats(paint_rgb, seed: int) -> dict:
    rng = random.Random(seed)
    lum = sum(paint_rgb) / 3
    metallic = 0.0 if lum > 0.6 or rng.random() < 0.3 else 0.55
    return {
        "paint": paint_material(paint_rgb, metallic, 0.32 + 0.1 * rng.random()),
        "glass": H.principled("veh_glass", (0.015, 0.017, 0.02), rough=0.05, metal=0.0, spec=0.9),
        "tire": H.principled("veh_tire", (0.018, 0.018, 0.018), rough=0.85),
        "rim": H.principled("veh_rim", (0.55, 0.56, 0.58), rough=0.35, metal=1.0),
        "rim_dark": H.principled("veh_rim_dark", (0.06, 0.06, 0.065), rough=0.45, metal=0.8),
        "trim": H.principled("veh_trim", (0.025, 0.025, 0.027), rough=0.6),
        "chrome": H.principled("veh_chrome", (0.8, 0.8, 0.8), rough=0.12, metal=1.0),
        "head": H.principled("veh_headlamp", (0.7, 0.72, 0.75), rough=0.08, metal=0.3, spec=1.0),
        "tail": H.principled("veh_taillamp", (0.16, 0.006, 0.006), rough=0.12, spec=0.8),
        "amber": H.principled("veh_amber", (0.5, 0.2, 0.0), rough=0.15),
        "plate": H.principled("veh_plate", (0.75, 0.75, 0.7), rough=0.5),
        "under": H.principled("veh_under", (0.01, 0.01, 0.01), rough=0.9),
        "box": H.principled("veh_box_frp", (0.78, 0.78, 0.76), rough=0.45),
        "steel": H.principled("veh_steel_dark", (0.06, 0.06, 0.06), rough=0.55, metal=0.6),
        "deck": H.principled("veh_wood_deck", (0.16, 0.11, 0.07), rough=0.85),
    }


def _arched_profile(L, zb, top_pts, axles, r_arch, z_wheel):
    """Side profile (y, z): bottom edge front->rear with semicircular arches over each axle, then top rear->front."""
    pts = [(L / 2, zb)]
    t0 = math.asin(min(0.95, (zb - z_wheel) / r_arch)) if zb > z_wheel else 0.0
    for yc in sorted(axles, reverse=True):  # front axle first
        if t0 == 0.0:
            pts.append((yc + r_arch, zb))
        n = 10
        for i in range(n + 1):
            t = t0 + (math.pi - 2 * t0) * i / n
            pts.append((yc + r_arch * math.cos(t), z_wheel + r_arch * math.sin(t)))
        if t0 == 0.0:
            pts.append((yc - r_arch, zb))
    pts.append((-L / 2, zb))
    pts.extend(top_pts)  # rear -> front, must end above (L/2, zb)
    # drop consecutive duplicates
    out = []
    for p in pts:
        if not out or abs(out[-1][0] - p[0]) > 1e-4 or abs(out[-1][1] - p[1]) > 1e-4:
            out.append(p)
    return out


def _wheels(mb, M, axles, track, r, width=0.24, dark=False, dual_rear=False):
    for i, yc in enumerate(axles):
        for side in (-1, 1):
            xs = [side * track / 2]
            if dual_rear and i == len(axles) - 1:
                xs = [side * (track / 2 - 0.15), side * (track / 2 + 0.12)]
            for x in xs:
                mb.cylinder((x, yc, r), r, width, M["tire"], axis="x", segs=24, bevel=0.035)
                rim = M["rim_dark"] if dark else M["rim"]
                xo = x + side * (width / 2 + 0.004)
                mb.cylinder((xo, yc, r), r * 0.62, 0.02, rim, axis="x", segs=20)
                mb.cylinder((xo + side * 0.011, yc, r), r * 0.18, 0.012, M["trim"] if not dark else M["chrome"],
                            axis="x", segs=10)
                for k in range(5):  # dark gaps between spokes
                    a = 2 * math.pi * k / 5 + 0.3
                    cy, cz = yc + math.cos(a) * r * 0.4, r + math.sin(a) * r * 0.4
                    mb.cylinder((xo + side * 0.008, cy, cz), r * 0.11, 0.008, M["under"], axis="x", segs=8)


def sculpted(mb, build, voxel: float, mat_for_face, W: float, z_lo: float, z_hi: float, bulge: float = 0.05,
             dome: float = 0.03):
    """Build a part in a scratch mesh, voxel-remesh it (rounded edges), then shape it like sheet metal:
    sides bulge at mid height and tuck in at the top and bottom, upward faces dome across the width.
    Faces get materials from mat_for_face(normal) and are merged into `mb`."""
    import bmesh

    tmp = H.MeshBuilder()
    build(tmp)
    col = bpy.context.scene.collection
    o = tmp.to_object("_sculpt_tmp", col, smooth_angle=None)
    rm = o.modifiers.new("remesh", "REMESH")
    rm.mode = "VOXEL"
    rm.voxel_size = voxel
    rm.use_smooth_shade = True
    sm = o.modifiers.new("smooth", "CORRECTIVE_SMOOTH")
    sm.iterations = 4
    sm.use_only_smooth = True
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
    bm = bmesh.new()
    bm.from_mesh(me)
    zm, zh = (z_lo + z_hi) / 2, max(0.1, (z_hi - z_lo) / 2)
    bm.normal_update()
    for v in bm.verts:
        t = (v.co.z - zm) / zh
        v.co.x *= 1.0 - bulge * min(1.5, t * t)
        if v.normal.z > 0.3:
            v.co.z += dome * v.normal.z * max(0.0, 1 - (2 * v.co.x / W) ** 2)
    bm.normal_update()
    for f in bm.faces:
        f.material_index = mb.slot(mat_for_face(f.normal))
        f.smooth = True
    me2 = bpy.data.meshes.new("_sculpt_out")
    bm.to_mesh(me2)
    bm.free()
    for m in mb.mats:
        me2.materials.append(m)
    mb.bm.from_mesh(me2)
    for d in (me, me2):
        bpy.data.meshes.remove(d)
    mesh = o.data
    bpy.data.objects.remove(o)
    bpy.data.meshes.remove(mesh)


def _lamps(mb, M, L, W, z_head, z_tail, head_h=0.14, tail_h=0.22, inset=0.02):
    for side in (-1, 1):
        mb.box((side * (W / 2 - 0.22), L / 2 - inset, z_head), (0.2, 0.04, head_h / 2), M["head"], bevel=0.02)
        mb.box((side * (W / 2 - 0.13), -L / 2 + inset, z_tail), (0.12, 0.04, tail_h / 2), M["tail"], bevel=0.015)
    mb.box((0, L / 2 + 0.005, z_head - 0.2), (0.17, 0.01, 0.06), M["plate"])
    mb.box((0, -L / 2 - 0.005, z_tail - 0.18), (0.17, 0.01, 0.08), M["plate"])


def _mirrors(mb, M, W, y, z):
    for side in (-1, 1):
        mb.box((side * (W / 2 + 0.08), y, z), (0.09, 0.05, 0.07), M["paint"], bevel=0.02)


def _side_x(W, z, z_lo, z_hi, bulge):
    zm, zh = (z_lo + z_hi) / 2, max(0.1, (z_hi - z_lo) / 2)
    t = (z - zm) / zh
    return W / 2 * (1 - bulge * min(1.5, t * t)) + 0.003


def _door_seams(mb, M, W, ys, z_lo, z_hi, zb, belt, bulge):
    """Thin dark panel gaps on both sides, following the body's side curvature."""
    for y in ys:
        for side in (-1, 1):
            pts = [(zb + 0.12), (zb + belt) / 2, belt - 0.03]
            for za, zc in zip(pts, pts[1:]):
                mb.tube((side * _side_x(W, za, z_lo, z_hi, bulge), y, za),
                        (side * _side_x(W, zc, z_lo, z_hi, bulge), y, zc), 0.004, M["under"], segs=4)


def _belt_trim(mb, M, gw, y0, y1, belt):
    for side in (-1, 1):
        mb.tube((side * (gw / 2 + 0.008), y0, belt - 0.03), (side * (gw / 2 + 0.008), y1, belt - 0.03), 0.014,
                M["trim"], segs=6)


def car_like(mb, M, L, W, H, *, r, wheelbase, f_over, zb, hood_z, deck_z, belt, gh_rear_base, gh_front_base,
             roof_rear, roof_front, nose_z, tail_z, track=None, taper=0.80, rails=False, cladding=False):
    yf = L / 2 - f_over
    yr = yf - wheelbase
    top = [
        (-L / 2, tail_z - 0.25),
        (-L / 2 + 0.03, tail_z),
        (-L / 2 + 0.25, deck_z),
        (gh_rear_base, belt),
        (gh_front_base, belt),
        (L / 2 - 0.35, hood_z),
        (L / 2 - 0.03, nose_z),
        (L / 2, nose_z - 0.25),
    ]
    prof = _arched_profile(L, zb, top, [yf, yr], r + 0.06, r)
    sculpted(mb, lambda t: t.profile(prof, -W / 2, W / 2, M["paint"]), 0.04, lambda n: M["paint"], W, zb, belt + 0.1,
             bulge=0.06, dome=0.035)
    # greenhouse: rounded glass, the upward-facing part is the painted roof
    gw = W - 0.14
    gh = [(gh_rear_base, belt - 0.06), (gh_front_base, belt - 0.06), (roof_front, H), (roof_rear, H)]
    roof_n = 0.82
    sculpted(mb, lambda t: t.profile(gh, -gw / 2, gw / 2, M["glass"], taper_above=belt + 0.05, taper=taper), 0.035,
             lambda n: M["paint"] if n.z > roof_n else M["glass"], W, belt - 0.1, H + 0.6, bulge=0.0, dome=0.03)
    rw = gw * taper
    # A and C pillars in body colour, B pillar black
    for side in (-1, 1):
        xb = side * (gw / 2 - 0.01)
        xt = side * (rw / 2)
        mb.tube((xb, gh_front_base - 0.02, belt), (xt, roof_front, H - 0.04), 0.035, M["paint"])
        mb.tube((xb, gh_rear_base + 0.02, belt), (xt, roof_rear, H - 0.04), 0.05, M["paint"])
        ymid = (roof_front + roof_rear) / 2 - 0.1
        mb.tube((side * (gw / 2 + 0.005), ymid, belt), (side * (rw / 2 + 0.005), ymid, H - 0.04), 0.03, M["trim"])
    _door_seams(mb, M, W, [gh_front_base - 0.05, (roof_front + roof_rear) / 2 - 0.1, gh_rear_base + 0.3],
                zb, belt + 0.1, zb, belt, 0.06)
    _belt_trim(mb, M, gw, gh_rear_base + 0.1, gh_front_base - 0.05, belt)
    if rails:
        for side in (-1, 1):
            mb.box((side * (rw / 2 - 0.05), (roof_front + roof_rear) / 2, H + 0.03),
                   (0.025, (roof_front - roof_rear) / 2 - 0.1, 0.025), M["trim"])
    if cladding:
        mb.box((0, 0, zb + 0.08), (W / 2 + 0.01, L / 2 - 0.3, 0.08), M["trim"])
    # bumpers, grille
    mb.box((0, L / 2 - 0.05, zb + 0.15), (W / 2 - 0.05, 0.07, 0.12), M["trim"], bevel=0.03)
    mb.box((0, -L / 2 + 0.05, zb + 0.15), (W / 2 - 0.05, 0.07, 0.12), M["trim"], bevel=0.03)
    mb.box((0, L / 2 - 0.005, (nose_z + zb) / 2 + 0.1), (W * 0.28, 0.02, 0.1), M["trim"])
    _lamps(mb, M, L, W, nose_z - 0.12, tail_z - 0.1)
    _mirrors(mb, M, W, gh_front_base - 0.1, belt + 0.12)
    _wheels(mb, M, [yf, yr], (track or W - 0.2) - 0.24 + 0.24, r)
    mb.box((0, 0, zb + 0.02), (W / 2 - 0.1, L / 2 - 0.3, 0.05), M["under"])


def sedan(mb, M, L, W, H):
    car_like(mb, M, L, W, H, r=0.34, wheelbase=2.83, f_over=0.95, zb=0.24, hood_z=0.84, deck_z=1.0, belt=0.98,
             gh_rear_base=-1.4, gh_front_base=0.75, roof_rear=-0.9, roof_front=-0.05, nose_z=0.72, tail_z=0.95)


def hatch(mb, M, L, W, H):
    car_like(mb, M, L, W, H, r=0.33, wheelbase=2.70, f_over=0.88, zb=0.25, hood_z=0.9, deck_z=1.0, belt=1.0,
             gh_rear_base=-2.0, gh_front_base=0.8, roof_rear=-1.75, roof_front=0.0, nose_z=0.76, tail_z=1.0)


def suv(mb, M, L, W, H):
    car_like(mb, M, L, W, H, r=0.39, wheelbase=2.95, f_over=0.95, zb=0.36, hood_z=1.08, deck_z=1.12, belt=1.12,
             gh_rear_base=-2.3, gh_front_base=1.05, roof_rear=-2.22, roof_front=0.25, nose_z=1.0, tail_z=1.12,
             rails=True, cladding=True, taper=0.84)


def suv_compact(mb, M, L, W, H):
    car_like(mb, M, L, W, H, r=0.36, wheelbase=2.68, f_over=0.92, zb=0.32, hood_z=1.0, deck_z=1.05, belt=1.05,
             gh_rear_base=-2.05, gh_front_base=0.9, roof_rear=-1.95, roof_front=0.15, nose_z=0.92, tail_z=1.05,
             rails=True, cladding=True, taper=0.82)


def minivan(mb, M, L, W, H):
    car_like(mb, M, L, W, H, r=0.36, wheelbase=3.03, f_over=0.98, zb=0.28, hood_z=1.0, deck_z=1.05, belt=1.02,
             gh_rear_base=-2.45, gh_front_base=1.55, roof_rear=-2.35, roof_front=0.55, nose_z=0.85, tail_z=1.02,
             taper=0.86)


def pickup(mb, M, L, W, H):
    r, yf = 0.40, L / 2 - 0.95
    yr = yf - 3.68
    zb, belt = 0.48, 1.26
    cab_back, cab_front = -0.55, 1.2
    top = [
        (-L / 2, 0.62), (-L / 2, 0.92), (cab_back - 0.08, 0.92), (cab_back - 0.08, belt),
        (cab_front, belt), (L / 2 - 0.25, 1.2), (L / 2 - 0.02, 1.16), (L / 2, 0.75),
    ]
    prof = _arched_profile(L, zb, top, [yf, yr], r + 0.08, r)
    sculpted(mb, lambda t: t.profile(prof, -W / 2, W / 2, M["paint"]), 0.04, lambda n: M["paint"], W, zb, belt + 0.1,
             bulge=0.05, dome=0.03)
    # bed walls and tailgate
    bed_len = (-L / 2, cab_back - 0.08)
    yc = (bed_len[0] + bed_len[1]) / 2
    hl = (bed_len[1] - bed_len[0]) / 2
    for side in (-1, 1):
        mb.box((side * (W / 2 - 0.04), yc, 1.12), (0.04, hl, 0.2), M["paint"], bevel=0.02)
    mb.box((0, -L / 2 + 0.04, 1.12), (W / 2 - 0.04, 0.04, 0.2), M["paint"], bevel=0.02)
    mb.box((0, cab_back - 0.12, 1.12), (W / 2 - 0.04, 0.04, 0.2), M["paint"])
    mb.box((0, yc, 0.93), (W / 2 - 0.08, hl - 0.05, 0.01), M["trim"])  # bed liner
    gw = W - 0.16
    gh = [(cab_back, belt - 0.06), (cab_front, belt - 0.06), (0.35, H), (cab_back + 0.05, H)]
    sculpted(mb, lambda t: t.profile(gh, -gw / 2, gw / 2, M["glass"], taper_above=belt + 0.05, taper=0.88), 0.035,
             lambda n: M["paint"] if n.z > 0.82 else M["glass"], W, belt - 0.1, H + 0.6, bulge=0.0, dome=0.025)
    _door_seams(mb, M, W, [cab_front - 0.05, 0.1, cab_back + 0.02], zb, belt + 0.1, zb, belt, 0.05)
    _belt_trim(mb, M, gw, cab_back + 0.05, cab_front - 0.05, belt)
    for side in (-1, 1):
        mb.tube((side * (gw / 2 - 0.01), cab_front - 0.02, belt), (side * gw * 0.44, 0.35, H - 0.04), 0.04, M["paint"])
        mb.tube((side * (gw / 2 + 0.005), 0.2, belt), (side * (gw * 0.44 + 0.005), 0.0, H - 0.04), 0.035, M["trim"])
    # big grille, bumpers, lamps, mirrors, wheels
    mb.box((0, L / 2 - 0.005, 0.95), (W * 0.36, 0.02, 0.17), M["chrome"], bevel=0.01)
    mb.box((0, L / 2 - 0.02, zb + 0.12), (W / 2 - 0.02, 0.08, 0.13), M["chrome"], bevel=0.03)
    mb.box((0, -L / 2 - 0.02, 0.58), (W / 2 - 0.05, 0.09, 0.1), M["chrome"], bevel=0.03)
    _lamps(mb, M, L, W, 1.0, 1.05, head_h=0.18, tail_h=0.35)
    _mirrors(mb, M, W + 0.1, cab_front - 0.15, belt + 0.18)
    _wheels(mb, M, [yf, yr], W - 0.22, r, width=0.27, dark=True)
    mb.box((0, 0, zb + 0.02), (W / 2 - 0.15, L / 2 - 0.4, 0.06), M["under"])
    mb.box((0, 0, 0.42), (0.5, L / 2 - 0.6, 0.06), M["under"])  # frame rails under the body


def box_truck(mb, M, L, W, H):
    r = 0.42
    yf, yr = L / 2 - 1.05, L / 2 - 1.05 - 4.0
    cab_len, cab_h, cab_w = 1.75, 2.45, 2.0
    cab_front = L / 2
    # cab-over: tall cab with a raked windscreen
    cab = [(cab_front - cab_len, 0.75), (cab_front, 0.75), (cab_front, 1.55), (cab_front - 0.18, cab_h),
           (cab_front - cab_len, cab_h)]
    mb.profile(cab, -cab_w / 2, cab_w / 2, M["box"], bevel=0.08, segments=2)
    mb.box((0, cab_front + 0.005, 1.95), (cab_w / 2 - 0.12, 0.02, 0.38), M["glass"],
           matrix=Matrix.Translation((0, 0, 0)))
    for side in (-1, 1):
        mb.box((side * (cab_w / 2 + 0.005), cab_front - 0.6, 1.9), (0.01, 0.45, 0.35), M["glass"])
    mb.box((0, cab_front - 0.01, 1.05), (cab_w / 2 - 0.1, 0.03, 0.18), M["trim"])
    mb.box((0, cab_front + 0.02, 0.62), (cab_w / 2, 0.1, 0.14), M["trim"], bevel=0.03)
    # box body
    bx0, bx1 = -L / 2, cab_front - cab_len - 0.1
    mb.box((0, (bx0 + bx1) / 2, (0.95 + H) / 2), (W / 2, (bx1 - bx0) / 2, (H - 0.95) / 2), M["box"], bevel=0.03)
    for k in range(12):  # roll-up door ribs at the rear
        z = 1.05 + k * 0.2
        mb.box((0, bx0 - 0.01, z), (W / 2 - 0.12, 0.012, 0.012), M["steel"])
    mb.box((0, (bx0 + bx1) / 2, 0.85), (W / 2 - 0.05, (bx1 - bx0) / 2, 0.1), M["steel"])  # floor rail
    mb.box((0, 0, 0.62), (0.45, L / 2 - 0.3, 0.1), M["under"])  # chassis
    mb.box((0, bx0 + 0.15, 0.5), (W / 2 - 0.1, 0.05, 0.05), M["steel"])  # underride bar
    for side in (-1, 1):
        mb.box((side * (W / 2 - 0.15), bx0 + 0.02, 1.15), (0.09, 0.03, 0.12), M["tail"])
        mb.box((side * (cab_w / 2 - 0.2), cab_front + 0.01, 1.18), (0.15, 0.03, 0.07), M["head"])
        mb.box((side * (cab_w / 2 + 0.12), cab_front - 0.25, 1.95), (0.05, 0.08, 0.2), M["trim"])
    _wheels(mb, M, [yf, yr], 1.9, r, width=0.24, dark=False, dual_rear=True)
    mb.box((0, -L / 2 - 0.005, 0.75), (0.17, 0.01, 0.08), M["plate"])


def skid_steer(mb, M, L, W, H):
    r = 0.39
    paint, black = M["paint"], M["trim"]
    # chassis and engine hood (rear), wheels outside
    mb.box((0, -0.35, 0.72), (0.58, 1.0, 0.42), paint, bevel=0.04)
    mb.box((0, -1.3, 0.95), (0.6, 0.12, 0.45), paint, bevel=0.05)  # rear door / engine
    # ROPS cab: posts, roof, side screens
    cy0, cy1, cz0, cz1 = -0.85, 0.35, 1.12, H - 0.05
    for x in (-0.48, 0.48):
        for y in (cy0, cy1):
            mb.box((x, y, (cz0 + cz1) / 2), (0.04, 0.04, (cz1 - cz0) / 2), black)
        mb.box((x, (cy0 + cy1) / 2, (cz0 + cz1) / 2), (0.005, (cy1 - cy0) / 2, (cz1 - cz0) / 2 - 0.05), M["glass"])
    mb.box((0, (cy0 + cy1) / 2, H - 0.04), (0.55, 0.68, 0.05), black, bevel=0.02)
    mb.box((0, cy1, (cz0 + cz1) / 2), (0.46, 0.005, (cz1 - cz0) / 2 - 0.05), M["glass"])
    # lift arms from the rear towers to the front, bucket
    for side in (-1, 1):
        x = side * 0.7
        mb.box((x, -1.05, 1.35), (0.07, 0.15, 0.55), paint, bevel=0.02)
        mb.tube((x, -1.0, 1.85), (x, 0.55, 1.2), 0.08, paint)
        mb.tube((x, 0.55, 1.2), (x, 1.15, 0.45), 0.08, paint)
    mb.box((0, L / 2 - 0.3, 0.35), (W / 2 - 0.02, 0.3, 0.3), M["steel"], bevel=0.02)  # bucket
    mb.box((0, L / 2 - 0.02, 0.06), (W / 2 - 0.02, 0.02, 0.05), M["steel"])
    for y in (-0.7, 0.25):
        for side in (-1, 1):
            mb.cylinder((side * (W / 2 - 0.16), y, r), r, 0.3, M["tire"], axis="x", segs=20, bevel=0.04)
            mb.cylinder((side * (W / 2 - 0.01), y, r), r * 0.55, 0.02, paint, axis="x", segs=12)


def trailer(mb, M, L, W, H):
    deck_z = 0.58
    mb.box((0, 0, deck_z), (W / 2 - 0.25, L / 2 - 1.0, 0.04), M["deck"])
    for side in (-1, 1):
        mb.box((side * (W / 2 - 0.27), 0, deck_z - 0.06), (0.04, L / 2 - 1.0, 0.07), M["steel"])
        mb.box((side * (W / 2 - 0.27), -L / 2 + 1.0, deck_z + 0.25), (0.03, 0.03, 0.25), M["steel"])
        mb.box((side * (W / 2 - 0.1), -0.3, 0.55), (0.12, 0.65, 0.04), M["steel"])  # fender
        mb.tube((side * (W / 2 - 0.27), L / 2 - 1.0, deck_z - 0.05), (0, L / 2, 0.45), 0.04, M["steel"])  # tongue
    mb.box((0, -L / 2 + 1.0, deck_z + 0.5), (W / 2 - 0.25, 0.04, 0.5), M["steel"])  # ramp gate (up)
    for side in (-1, 1):
        for y in (-0.62, 0.02):
            mb.cylinder((side * (W / 2 - 0.12), y, 0.34), 0.34, 0.2, M["tire"], axis="x", segs=20, bevel=0.03)
            mb.cylinder((side * (W / 2 - 0.015), y, 0.34), 0.2, 0.02, M["rim"], axis="x", segs=12)
    mb.cylinder((0, L / 2 - 0.15, 0.3), 0.04, 0.6, M["steel"])  # jack


BUILDERS = {"sedan": sedan, "hatch": hatch, "suv": suv, "suv_compact": suv_compact, "pickup": pickup,
            "minivan": minivan, "box_truck": box_truck, "skid_steer": skid_steer, "trailer": trailer}


def build_vehicle(v: dict, dims: dict, paint_rgb, col) -> bpy.types.Object:
    L, W, Hh = dims[v["kind"]]
    M = _mats(paint_rgb, v["seed"])
    mb = H.MeshBuilder()
    BUILDERS[v["kind"]](mb, M, L, W, Hh)
    o = mb.to_object(v["name"], col, smooth_angle=40.0)
    o.location = (v["x"], v["y"], 0.0)
    o.rotation_euler = (0, 0, math.radians(v["yaw"]))
    o["kind"] = v["kind"]
    return o


def build_covered_car(v: dict, col) -> bpy.types.Object:
    blend = H.CACHE / "covered_car" / "covered_car_1k.blend"
    parts = H.append_objects(blend)
    root = H.empty(v["name"], col, (v["x"], v["y"], 0.0), v["yaw"])
    for p in parts:
        if p.type != "MESH":
            continue
        col.objects.link(p)
        p.parent = root
        # Poly Haven model is nose along +y already (measured 1.79 x 4.38 m)
    root["kind"] = "covered_car"
    return root
