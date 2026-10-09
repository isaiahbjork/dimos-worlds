"""The intruder: Blender's CC0 realistic male base mesh, rigged here, dressed by material, posed by name.

Source mesh: Human Base Meshes bundle v1.4.1 (Blender Studio, CC0), object GEO-body_male_realistic,
1.69 m, A-pose, facing -y. We add a 17-bone armature at measured joint positions, bind with automatic
(heat) weights, assign hoodie / jeans / shoes / skin by region, and add a hood shell on the head bone.

Poses are bone directions in armature space (facing -y, +x is the person's left), applied parent first,
so they do not depend on bone roll. Runs inside Blender.
"""

from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector

import bhelp as H

SCALE = 1.78 / 1.69  # base mesh is 1.69 m; scale to an average adult male

# Joint positions on the unscaled base mesh (measured from vertex bands; see README).
J = {
    "pelvis": (0.0, -0.02, 0.93), "spine": (0.0, -0.02, 1.08), "chest": (0.0, -0.01, 1.24),
    "neck": (0.0, 0.0, 1.42), "head": (0.0, -0.01, 1.50), "head_top": (0.0, -0.02, 1.69),
    "hip": (0.09, -0.01, 0.90), "knee": (0.12, -0.02, 0.48), "ankle": (0.165, 0.03, 0.085),
    "toe": (0.17, -0.11, 0.02),
    "clav": (0.03, 0.0, 1.38), "shoulder": (0.175, 0.01, 1.38), "elbow": (0.29, 0.03, 1.08),
    "wrist": (0.36, 0.0, 0.86), "hand_tip": (0.39, -0.01, 0.70),
}


def _mirror(p, side):
    return (p[0] * side, p[1], p[2])


BONES = [  # name, head joint, tail joint, parent, side (+1 left, -1 right, 0 centre)
    ("pelvis", "pelvis", "spine", None, 0),
    ("spine", "spine", "chest", "pelvis", 0),
    ("chest", "chest", "neck", "spine", 0),
    ("neck", "neck", "head", "chest", 0),
    ("head", "head", "head_top", "neck", 0),
]
for _s, _n in ((1, "L"), (-1, "R")):
    BONES += [
        (f"thigh.{_n}", "hip", "knee", "pelvis", _s),
        (f"shin.{_n}", "knee", "ankle", f"thigh.{_n}", _s),
        (f"foot.{_n}", "ankle", "toe", f"shin.{_n}", _s),
        (f"clavicle.{_n}", "clav", "shoulder", "chest", _s),
        (f"upperarm.{_n}", "shoulder", "elbow", f"clavicle.{_n}", _s),
        (f"forearm.{_n}", "elbow", "wrist", f"upperarm.{_n}", _s),
        (f"hand.{_n}", "wrist", "hand_tip", f"forearm.{_n}", _s),
    ]


def _v(a, b, c):
    v = Vector((a, b, c))
    return v.normalized()


def _limb(forward_deg, out=0.0):
    """Direction for a limb hanging down, swung forward (toward -y) by forward_deg; `out` adds +x (left side)."""
    a = math.radians(forward_deg)
    return (out, -math.sin(a), -math.cos(a))


# Bone directions per pose for the left side; the right side mirrors x unless given explicitly.
# pelvis_offset: armature-space translation of the pelvis (metres, unscaled mesh units).
POSES = {
    "stand": {
        "pelvis_offset": (0, 0, 0),
        "dirs": {
            "upperarm.L": _limb(4, 0.18), "upperarm.R": _limb(4, -0.18),
            "forearm.L": _limb(14, 0.12), "forearm.R": _limb(14, -0.12),
            "hand.L": _limb(12, 0.05), "hand.R": _limb(12, -0.05),
            "thigh.L": _limb(0, 0.04), "thigh.R": _limb(0, -0.04),
            "shin.L": _limb(-2, 0.05), "shin.R": _limb(-2, -0.05),
        },
    },
    "walk": {
        "pelvis_offset": (0, 0, -0.025),
        "dirs": {
            "spine": _v(0, -0.08, 1), "chest": _v(0, -0.06, 1),
            "thigh.L": _limb(24, 0.03), "shin.L": _limb(10, 0.03), "foot.L": _v(0, -1, 0.15),
            "thigh.R": _limb(-16, -0.03), "shin.R": _limb(-45, -0.03), "foot.R": _v(0, -0.6, -0.8),
            "upperarm.L": _limb(-18, 0.16), "forearm.L": _limb(-4, 0.1), "hand.L": _limb(-4, 0.05),
            "upperarm.R": _limb(22, -0.16), "forearm.R": _limb(45, -0.1), "hand.R": _limb(45, -0.05),
        },
    },
    "crouch": {
        "pelvis_offset": (0, 0.24, -0.41),
        "dirs": {
            "spine": _v(0, -0.55, 0.84), "chest": _v(0, -0.45, 0.89), "neck": _v(0, -0.2, 1), "head": _v(0, -0.05, 1),
            "thigh.L": _v(0.3, -0.95, -0.06), "shin.L": _v(0.05, 0.33, -0.94), "foot.L": _v(0.05, -1, -0.15),
            "thigh.R": _v(-0.3, -0.95, -0.06), "shin.R": _v(-0.05, 0.33, -0.94), "foot.R": _v(-0.05, -1, -0.15),
            "upperarm.L": _v(0.15, -0.75, -0.65), "forearm.L": _v(0.0, -0.9, -0.1), "hand.L": _v(0, -1, 0),
            "upperarm.R": _v(-0.15, -0.7, -0.7), "forearm.R": _v(0.0, -0.85, -0.45), "hand.R": _v(0, -0.8, -0.6),
        },
    },
}


def _region_material(center: Vector, mats: dict):
    x, y, z = center
    ax = abs(x)
    if z > 1.46:
        return mats["skin"]
    if z < 0.10:
        return mats["shoe"]
    if ax > 0.3 and z < 0.89:  # hands below the cuff
        return mats["skin"]
    if z < 0.97 and ax < 0.26:
        return mats["jeans"]
    return mats["hoodie"]


def build_person(col) -> bpy.types.Object:
    blend = H.CACHE / "human_base_meshes" / "human-base-meshes-bundle-v1.4.1" / "human_base_meshes_bundle.blend"
    body = H.append_objects(blend, ["GEO-body_male_realistic"])[0]
    col.objects.link(body)
    for m in list(body.modifiers):
        body.modifiers.remove(m)
    body.parent = None
    body.location = (0, 0, 0)
    body.rotation_euler = (0, 0, 0)
    body.scale = (1, 1, 1)
    body.name = "person_body"

    mats = {
        "hoodie": H.principled("person_hoodie", (0.03, 0.031, 0.034), rough=0.95, spec=0.2),  # black hoodie
        "jeans": H.principled("person_jeans", (0.035, 0.045, 0.08), rough=0.9, spec=0.2),  # dark jeans
        "shoe": H.principled("person_shoe", (0.02, 0.02, 0.02), rough=0.7),
        "skin": H.principled("person_skin", (0.32, 0.2, 0.14), rough=0.55, spec=0.35),
    }
    me = body.data
    me.materials.clear()
    order = ["hoodie", "jeans", "shoe", "skin"]
    for k in order:
        me.materials.append(mats[k])
    for p in me.polygons:
        p.material_index = order.index(next(k for k in order if mats[k] == _region_material(p.center, mats)))
    H.smooth(me, 80.0)

    # armature
    arm_data = bpy.data.armatures.new("person_rig")
    arm = bpy.data.objects.new("person", arm_data)
    col.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for name, hj, tj, parent, side in BONES:
        b = arm_data.edit_bones.new(name)
        b.head = _mirror(J[hj], side or 1)
        b.tail = _mirror(J[tj], side or 1)
        b.roll = 0.0
        if parent:
            b.parent = arm_data.edit_bones[parent]
            b.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")

    # bind with heat weights, fall back to envelopes if the solver fails
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    body.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    except RuntimeError:
        bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
    if not body.vertex_groups or max((len(g.name) for g in body.vertex_groups), default=0) == 0:
        bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")

    # hood: a shell around the head with the face opening cut out, parented to the head bone
    mb = H.MeshBuilder()
    import bmesh
    res = bmesh.ops.create_uvsphere(mb.bm, u_segments=20, v_segments=12, radius=1.0)
    m = Matrix.Translation((0, 0.025, 1.585)) @ Matrix.Diagonal((0.118, 0.14, 0.135, 1))
    bmesh.ops.transform(mb.bm, matrix=m, verts=res["verts"])
    kill = [f for f in mb.bm.faces if f.calc_center_median().y < -0.07 and 1.50 < f.calc_center_median().z < 1.66]
    bmesh.ops.delete(mb.bm, geom=kill, context="FACES")
    for f in mb.bm.faces:
        f.material_index = mb.slot(mats["hoodie"])
    hood = mb.to_object("person_hood", col, smooth_angle=80.0)
    sol = hood.modifiers.new("thick", "SOLIDIFY")
    sol.thickness = 0.015
    hood.parent = arm
    hood.parent_type = "BONE"
    hood.parent_bone = "head"
    # bone parenting puts the child at the bone tail; compensate so it stays where it was modelled
    bpy.context.view_layer.update()
    hood.matrix_world = Matrix.Identity(4)

    arm.scale = (SCALE, SCALE, SCALE)
    arm["yaw_offset"] = 180.0  # the mesh faces -y; layout headings assume a model facing +y
    return arm


def apply_pose(arm: bpy.types.Object, pose: str) -> None:
    """Set the armature to a named pose. Bone directions are in armature space (unscaled)."""
    spec = POSES[pose]
    rest_dirs = {}
    for b in arm.data.bones:
        rest_dirs[b.name] = (b.tail_local - b.head_local).normalized()
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
    bpy.context.view_layer.update()
    dirs = spec["dirs"]
    # parents before children: BONES is ordered that way
    for name, *_ in BONES:
        pb = arm.pose.bones[name]
        target = dirs.get(name)
        if name == "pelvis":
            off = Vector(spec["pelvis_offset"])
            mat = pb.matrix.copy()
            if target is not None:
                q = rest_dirs[name].rotation_difference(Vector(target).normalized())
                mat = Matrix.Translation(mat.translation) @ q.to_matrix().to_4x4() @ Matrix.Translation(-mat.translation) @ mat
            mat.translation = mat.translation + off
            pb.matrix = mat
            bpy.context.view_layer.update()
            continue
        if target is None:
            continue
        cur = pb.matrix.copy()
        cur_dir = (cur.to_3x3() @ Vector((0, 1, 0))).normalized()
        q = cur_dir.rotation_difference(Vector(target).normalized())
        head = cur.translation.copy()
        new = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head) @ cur
        pb.matrix = new
        bpy.context.view_layer.update()
