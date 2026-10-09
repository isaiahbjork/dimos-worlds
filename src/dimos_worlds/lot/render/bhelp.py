"""Small bpy/bmesh helpers shared by the build scripts. Runs inside Blender only."""

from __future__ import annotations

import math
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

LOT_DIR = Path(__file__).resolve().parents[1]
CACHE = LOT_DIR / "assets" / "cache"


# ---- collections / objects ----------------------------------------------------


def collection(name: str, parent: bpy.types.Collection | None = None) -> bpy.types.Collection:
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if col.name not in parent.children:
        parent.children.link(col)
    return col


def link(obj: bpy.types.Object, col: bpy.types.Collection) -> bpy.types.Object:
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def empty(name: str, col, loc=(0, 0, 0), yaw_deg: float = 0.0) -> bpy.types.Object:
    o = bpy.data.objects.new(name, None)
    o.location = loc
    o.rotation_euler = (0, 0, math.radians(yaw_deg))
    col.objects.link(o)
    return o


def smooth(me: bpy.types.Mesh, angle_deg: float = 35.0) -> None:
    for p in me.polygons:
        p.use_smooth = True
    try:
        me.set_sharp_from_angle(angle=math.radians(angle_deg))
    except AttributeError:  # older API
        pass


# ---- materials ------------------------------------------------------------------


def _new_mat(name: str) -> tuple[bpy.types.Material, bpy.types.Node, bpy.types.NodeTree]:
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    return m, bsdf, nt


def principled(name: str, color=(0.5, 0.5, 0.5), rough=0.5, metal=0.0, coat=0.0, coat_rough=0.05,
               emission=None, emission_strength=0.0, alpha=1.0, transmission=0.0, ior=1.45,
               spec=0.5) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    m, b, _ = _new_mat(name)
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["IOR"].default_value = ior
    b.inputs["Specular IOR Level"].default_value = spec
    if coat:
        b.inputs["Coat Weight"].default_value = coat
        b.inputs["Coat Roughness"].default_value = coat_rough
    if transmission:
        b.inputs["Transmission Weight"].default_value = transmission
    if emission is not None:
        b.inputs["Emission Color"].default_value = (*emission, 1.0)
        b.inputs["Emission Strength"].default_value = emission_strength
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
    m.diffuse_color = (*color, 1.0)
    return m


def image(path: Path, non_color: bool = False) -> bpy.types.Image:
    img = bpy.data.images.load(str(path), check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def textured(name: str, asset_id: str, res: str, size_m: float, tint=(1, 1, 1), sat: float = 1.0,
             value: float = 1.0, rough_add: float = 0.0, box: bool = True, normal_strength: float = 1.0,
             metal: float = 0.0, coords: str = "Object", variation: float = 0.0) -> bpy.types.Material:
    """PBR material from a Poly Haven texture set, mapped in metres (object coordinates, box projection)."""
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    m, b, nt = _new_mat(name)
    nodes, links = nt.nodes, nt.links
    root = CACHE / asset_id
    tc = nodes.new("ShaderNodeTexCoord")
    mp = nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1 / size_m, 1 / size_m, 1 / size_m)
    links.new(tc.outputs[coords], mp.inputs["Vector"])

    def tex(kind: str, non_color: bool):
        t = nodes.new("ShaderNodeTexImage")
        t.image = image(root / f"{asset_id}_{kind}_{res}.jpg", non_color)
        if box:
            t.projection = "BOX"
            t.projection_blend = 0.25
        links.new(mp.outputs["Vector"], t.inputs["Vector"])
        return t

    diff = tex("diff", False)
    hsv = nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = sat
    hsv.inputs["Value"].default_value = value
    links.new(diff.outputs["Color"], hsv.inputs["Color"])
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    links.new(hsv.outputs["Color"], mix.inputs[6])
    mix.inputs[7].default_value = (*tint, 1)
    out_color = mix.outputs[2]
    if variation:
        # large-scale wear and patching: a 6-15 m noise darkens and lightens the texture
        nz = nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 0.09
        nz.inputs["Detail"].default_value = 3.0
        links.new(tc.outputs[coords], nz.inputs["Vector"])
        mr = nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = 0.3
        mr.inputs["From Max"].default_value = 0.7
        mr.inputs["To Min"].default_value = 1 - variation
        mr.inputs["To Max"].default_value = 1 + variation * 0.6
        links.new(nz.outputs["Fac"], mr.inputs["Value"])
        mul = nodes.new("ShaderNodeMix")
        mul.data_type = "RGBA"
        mul.blend_type = "MULTIPLY"
        mul.inputs["Factor"].default_value = 1.0
        links.new(out_color, mul.inputs[6])
        cmb = nodes.new("ShaderNodeCombineXYZ")
        for k in ("X", "Y", "Z"):
            links.new(mr.outputs["Result"], cmb.inputs[k])
        links.new(cmb.outputs[0], mul.inputs[7])
        out_color = mul.outputs[2]
    links.new(out_color, b.inputs["Base Color"])
    rough = tex("rough", True)
    if rough_add:
        add = nodes.new("ShaderNodeMath")
        add.operation = "ADD"
        add.use_clamp = True
        add.inputs[1].default_value = rough_add
        links.new(rough.outputs["Color"], add.inputs[0])
        links.new(add.outputs[0], b.inputs["Roughness"])
    else:
        links.new(rough.outputs["Color"], b.inputs["Roughness"])
    nor = tex("nor_gl", True)
    nm = nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = normal_strength
    links.new(nor.outputs["Color"], nm.inputs["Color"])
    links.new(nm.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Metallic"].default_value = metal
    return m


def emissive(name: str, color, strength: float) -> bpy.types.Material:
    """Pure emitter (lamp lenses, lit windows, sign faces). Strength is scaled at render time by mode."""
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    m, b, nt = _new_mat(name)
    b.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1)
    b.inputs["Emission Color"].default_value = (*color, 1)
    b.inputs["Emission Strength"].default_value = strength
    b.inputs["Roughness"].default_value = 0.3
    m["night_emission"] = strength  # render_frames.py turns these down in day mode
    return m


# ---- mesh building ---------------------------------------------------------------


class MeshBuilder:
    """Accumulate parts into one bmesh with per-part material slots, then emit one object."""

    def __init__(self) -> None:
        self.bm = bmesh.new()
        self.mats: list[bpy.types.Material] = []

    def slot(self, mat: bpy.types.Material) -> int:
        if mat not in self.mats:
            self.mats.append(mat)
        return self.mats.index(mat)

    def _finish(self, geom_verts, faces, mat, bevel: float, segments: int, bevel_angle: float = 30.0):
        idx = self.slot(mat)
        for f in faces:
            f.material_index = idx
        if bevel > 0:
            edges = set()
            for f in faces:
                for e in f.edges:
                    if len(e.link_faces) == 2 and e.calc_face_angle(0) > math.radians(bevel_angle):
                        edges.add(e)
            if edges:
                res = bmesh.ops.bevel(self.bm, geom=list(edges), offset=bevel, segments=segments,
                                      affect="EDGES", profile=0.5, clamp_overlap=True)
                for f in res.get("faces", []):
                    f.material_index = idx
        return faces

    def box(self, center, half, mat, bevel: float = 0.0, segments: int = 2, matrix: Matrix | None = None):
        res = bmesh.ops.create_cube(self.bm, size=1.0)
        verts = res["verts"]
        sx, sy, sz = (2 * h for h in half)
        m = Matrix.Translation(Vector(center)) @ Matrix.Diagonal((sx, sy, sz, 1.0))
        if matrix is not None:
            m = matrix @ m
        bmesh.ops.transform(self.bm, matrix=m, verts=verts)
        faces = list({f for v in verts for f in v.link_faces})
        return self._finish(verts, faces, mat, bevel, segments)

    def cylinder(self, center, radius, depth, mat, axis: str = "z", segs: int = 16, bevel: float = 0.0,
                 matrix: Matrix | None = None, radius2: float | None = None):
        res = bmesh.ops.create_cone(self.bm, cap_ends=True, segments=segs, radius1=radius,
                                    radius2=radius if radius2 is None else radius2, depth=depth)
        verts = res["verts"]
        rot = {"z": Matrix.Identity(4), "x": Matrix.Rotation(math.pi / 2, 4, "Y"),
               "y": Matrix.Rotation(math.pi / 2, 4, "X")}[axis]
        m = Matrix.Translation(Vector(center)) @ rot
        if matrix is not None:
            m = matrix @ m
        bmesh.ops.transform(self.bm, matrix=m, verts=verts)
        faces = list({f for v in verts for f in v.link_faces})
        return self._finish(verts, faces, mat, bevel, 2)

    def tube(self, a, b, radius, mat, segs: int = 8):
        a, b = Vector(a), Vector(b)
        d = b - a
        length = d.length
        if length < 1e-6:
            return []
        rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4()
        m = Matrix.Translation((a + b) / 2) @ rot
        return self.cylinder((0, 0, 0), radius, length, mat, segs=segs, matrix=m)

    def profile(self, pts_yz, x0: float, x1: float, mat, bevel: float = 0.0, segments: int = 2,
                taper_above: float | None = None, taper: float = 1.0, matrix: Matrix | None = None,
                bevel_angle: float = 30.0):
        """Extrude a side profile (list of (y, z), any winding) across x in [x0, x1]."""
        verts0 = [self.bm.verts.new((x0, y, z)) for y, z in pts_yz]
        face = self.bm.faces.new(verts0)
        ext = bmesh.ops.extrude_face_region(self.bm, geom=[face])
        new_verts = [g for g in ext["geom"] if isinstance(g, bmesh.types.BMVert)]
        bmesh.ops.translate(self.bm, vec=(x1 - x0, 0, 0), verts=new_verts)
        all_verts = verts0 + new_verts
        if taper_above is not None:
            xc = (x0 + x1) / 2
            for v in all_verts:
                if v.co.z > taper_above:
                    v.co.x = xc + (v.co.x - xc) * taper
        faces = list({f for v in all_verts for f in v.link_faces})
        bmesh.ops.recalc_face_normals(self.bm, faces=faces)
        if matrix is not None:
            bmesh.ops.transform(self.bm, matrix=matrix, verts=all_verts)
        return self._finish(all_verts, faces, mat, bevel, segments, bevel_angle)

    def quad(self, corners, mat):
        vs = [self.bm.verts.new(c) for c in corners]
        f = self.bm.faces.new(vs)
        f.material_index = self.slot(mat)
        return [f]

    def to_object(self, name: str, col, smooth_angle: float | None = 35.0) -> bpy.types.Object:
        me = bpy.data.meshes.new(name)
        self.bm.normal_update()
        self.bm.to_mesh(me)
        self.bm.free()
        for m in self.mats:
            me.materials.append(m)
        if smooth_angle is not None:
            smooth(me, smooth_angle)
        o = bpy.data.objects.new(name, me)
        col.objects.link(o)
        return o


def append_objects(blend: Path, names: list[str] | None = None) -> list[bpy.types.Object]:
    """Append objects (all, or by name) from a .blend. Scripts in the file are not run (--disable-autoexec)."""
    with bpy.data.libraries.load(str(blend), link=False) as (src, dst):
        dst.objects = [n for n in src.objects if names is None or n in names]
    return [o for o in dst.objects if o is not None]


def curve_wire(name: str, pts, radius: float, mat, col) -> bpy.types.Object:
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 1
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts):
        p.co = (*c, 1.0)
    o = bpy.data.objects.new(name, cu)
    cu.materials.append(mat)
    col.objects.link(o)
    return o


def text(name: str, body: str, size: float, mat, col, loc, rot_deg=(90, 0, 0), extrude: float = 0.005,
         align: str = "CENTER") -> bpy.types.Object:
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.extrude = extrude
    cu.align_x = align
    cu.align_y = "CENTER"
    o = bpy.data.objects.new(name, cu)
    o.location = loc
    o.rotation_euler = tuple(math.radians(a) for a in rot_deg)
    cu.materials.append(mat)
    col.objects.link(o)
    return o
