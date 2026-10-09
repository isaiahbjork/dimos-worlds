"""Build the lot render scene from layout.py and the cached CC0 assets; save assets/cache/lot.blend.

    cd src/dimos_worlds/lot && blender -b --factory-startup --disable-autoexec -P render/build_scene.py -- --out assets/cache/lot.blend

Re-run it after editing layout.py or render/*.py. Needs Blender (not a Python dependency) and the CC0 assets
from assets/fetch.py.
"""

from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bhelp as H  # noqa: E402
import layout as L  # noqa: E402
import person as P  # noqa: E402
import vehicles as V  # noqa: E402

CACHE = H.CACHE
RNG = random.Random(7)


def source_hash() -> str:
    here = Path(__file__).resolve().parent
    files = sorted(here.glob("*.py")) + [here.parent / "layout.py", here.parent / "camera_model.py"]
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ---- materials ------------------------------------------------------------------


def materials() -> dict:
    M = {}
    M["asphalt"] = H.textured("lot_asphalt", "worn_asphalt", "2k", 2.0, sat=0.2, value=2.2, rough_add=0.05,
                               variation=0.35)
    add_crack_sealant(M["asphalt"])
    M["street"] = H.textured("street_asphalt", "asphalt_02", "2k", 3.0, sat=0.3, value=0.8, variation=0.25)
    M["gravel"] = H.textured("surround_gravel", "asphalt_02", "2k", 2.2, tint=(0.75, 0.66, 0.55), sat=0.6, value=1.6,
                             rough_add=0.1)
    M["concrete"] = H.textured("concrete", "concrete_floor_worn_001", "1k", 3.0, sat=0.5, value=3.0)
    M["plaster"] = H.textured("office_plaster", "painted_plaster_wall", "1k", 2.0, tint=(0.82, 0.76, 0.66), sat=0.4)
    M["metal_wall"] = H.textured("bay_metal", "corrugated_iron_02", "1k", 2.7, tint=(0.72, 0.74, 0.76), sat=0.15, value=4.0,
                                 metal=0.4, normal_strength=1.5)
    M["paint_white"] = H.principled("line_paint", (0.62, 0.62, 0.58), rough=0.7)
    M["paint_yellow"] = H.principled("line_yellow", (0.6, 0.42, 0.03), rough=0.7)
    M["galv"] = H.principled("galvanized", (0.5, 0.52, 0.53), rough=0.45, metal=0.85)
    M["bronze"] = H.principled("fixture_bronze", (0.05, 0.045, 0.04), rough=0.5, metal=0.5)
    M["dark_steel"] = H.principled("dark_steel", (0.05, 0.05, 0.05), rough=0.55, metal=0.6)
    M["yellow_steel"] = H.principled("safety_yellow", (0.65, 0.45, 0.02), rough=0.5)
    M["roof"] = H.principled("roof_membrane", (0.18, 0.18, 0.18), rough=0.9)
    M["trim"] = H.principled("bldg_trim", (0.09, 0.09, 0.1), rough=0.5, metal=0.3)
    M["alu"] = H.principled("storefront_alu", (0.35, 0.36, 0.37), rough=0.3, metal=1.0)
    M["glass"] = H.principled("window_glass", (0.01, 0.012, 0.015), rough=0.04, spec=1.0)
    M["tank"] = H.principled("tank_white", (0.72, 0.72, 0.7), rough=0.4, coat=0.3)
    M["sign_white"] = H.principled("sign_white", (0.75, 0.75, 0.72), rough=0.4)
    M["sign_red"] = H.principled("sign_red", (0.45, 0.02, 0.02), rough=0.4)
    M["sign_black"] = H.principled("sign_black", (0.02, 0.02, 0.02), rough=0.4)
    M["sign_back"] = H.principled("sign_back", (0.4, 0.41, 0.42), rough=0.4, metal=0.8)
    M["tree"] = H.principled("tree_foliage", (0.02, 0.035, 0.015), rough=0.9)
    M["bark"] = H.principled("tree_bark", (0.04, 0.03, 0.02), rough=0.9)
    M["dumpster_lid"] = H.principled("dumpster_lid", (0.015, 0.015, 0.015), rough=0.6)
    M["oil"] = oil_stain_material()
    M["led_lens"] = H.emissive("led_lens", (1.0, 0.88, 0.76), 60.0)
    M["hps_lens"] = H.emissive("hps_lens", (1.0, 0.55, 0.2), 60.0)
    M["street_lens"] = H.emissive("street_lens", (1.0, 0.78, 0.55), 60.0)
    M["wallpack_lens"] = H.emissive("wallpack_lens", (1.0, 0.9, 0.8), 40.0)
    M["window_lit"] = lit_window_material()
    M["sign_lit"] = H.emissive("pole_sign_face", (0.95, 0.95, 0.9), 1.6)
    M["sign_lit_red"] = H.emissive("pole_sign_red", (0.9, 0.05, 0.03), 1.2)
    M["door_lit"] = H.emissive("door_glow", (0.9, 0.75, 0.55), 0.12)
    M["wire"] = fence_wire_material()
    return M


def add_crack_sealant(m, scale: float = 0.3, width: float = 0.009) -> None:
    """Asphalt crack-sealant lines: dark, slightly glossy tar along a warped Voronoi network, patchy."""
    nt = m.node_tree
    n, ln = nt.nodes, nt.links
    b = n.get("Principled BSDF")
    base_link = b.inputs["Base Color"].links[0]
    base_out = base_link.from_socket
    rough_out = b.inputs["Roughness"].links[0].from_socket
    tc = n.new("ShaderNodeTexCoord")
    warp = n.new("ShaderNodeTexNoise")
    warp.inputs["Scale"].default_value = 0.6
    warp.inputs["Detail"].default_value = 4.0
    ln.new(tc.outputs["Object"], warp.inputs["Vector"])
    vmix = n.new("ShaderNodeMix")
    vmix.data_type = "VECTOR"
    vmix.inputs["Factor"].default_value = 0.6
    ln.new(tc.outputs["Object"], vmix.inputs[4])
    ln.new(warp.outputs["Color"], vmix.inputs[5])
    vor = n.new("ShaderNodeTexVoronoi")
    vor.feature = "DISTANCE_TO_EDGE"
    vor.inputs["Scale"].default_value = scale
    ln.new(vmix.outputs[1], vor.inputs["Vector"])
    line = n.new("ShaderNodeMapRange")
    line.inputs["From Min"].default_value = width
    line.inputs["From Max"].default_value = width * 0.4
    ln.new(vor.outputs["Distance"], line.inputs["Value"])
    patch = n.new("ShaderNodeTexNoise")
    patch.inputs["Scale"].default_value = 0.15
    ln.new(tc.outputs["Object"], patch.inputs["Vector"])
    pm = n.new("ShaderNodeMapRange")
    pm.inputs["From Min"].default_value = 0.45
    pm.inputs["From Max"].default_value = 0.6
    ln.new(patch.outputs["Fac"], pm.inputs["Value"])
    mask = n.new("ShaderNodeMath")
    mask.operation = "MULTIPLY"
    ln.new(line.outputs["Result"], mask.inputs[0])
    ln.new(pm.outputs["Result"], mask.inputs[1])
    cmix = n.new("ShaderNodeMix")
    cmix.data_type = "RGBA"
    ln.new(mask.outputs[0], cmix.inputs["Factor"])
    ln.new(base_out, cmix.inputs[6])
    cmix.inputs[7].default_value = (0.012, 0.012, 0.012, 1)
    nt.links.remove(base_link)
    ln.new(cmix.outputs[2], b.inputs["Base Color"])
    rmix = n.new("ShaderNodeMix")
    rmix.data_type = "FLOAT"
    ln.new(mask.outputs[0], rmix.inputs["Factor"])
    ln.new(rough_out, rmix.inputs[2])
    rmix.inputs[3].default_value = 0.35
    ln.new(rmix.outputs[0], b.inputs["Roughness"])


def oil_stain_material():
    m = bpy.data.materials.new("oil_stain")
    try:
        m.use_nodes = True
    except Exception:
        pass
    nt = m.node_tree
    n, ln = nt.nodes, nt.links
    b = n.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.012, 0.011, 0.01, 1)
    b.inputs["Roughness"].default_value = 0.62
    b.inputs["Specular IOR Level"].default_value = 0.25
    tc = n.new("ShaderNodeTexCoord")
    grad = n.new("ShaderNodeTexGradient")
    grad.gradient_type = "SPHERICAL"
    ln.new(tc.outputs["Object"], grad.inputs["Vector"])
    noise = n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    noise.inputs["Detail"].default_value = 6.0
    ln.new(tc.outputs["Object"], noise.inputs["Vector"])
    mul = n.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    ln.new(grad.outputs["Fac"], mul.inputs[0])
    ln.new(noise.outputs["Fac"], mul.inputs[1])
    ramp = n.new("ShaderNodeMapRange")
    ramp.inputs["From Min"].default_value = 0.12
    ramp.inputs["From Max"].default_value = 0.3
    ramp.inputs["To Max"].default_value = 0.75
    ln.new(mul.outputs[0], ramp.inputs["Value"])
    ln.new(ramp.outputs["Result"], b.inputs["Alpha"])
    return m


def lit_window_material():
    """Warm interior light seen through horizontal blinds."""
    m = bpy.data.materials.new("window_lit")
    try:
        m.use_nodes = True
    except Exception:
        pass
    nt = m.node_tree
    n, ln = nt.nodes, nt.links
    b = n.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1)
    b.inputs["Roughness"].default_value = 0.05
    tc = n.new("ShaderNodeTexCoord")
    wave = n.new("ShaderNodeTexWave")
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = 0.0
    sep = n.new("ShaderNodeSeparateXYZ")
    ln.new(tc.outputs["Object"], sep.inputs["Vector"])
    mod = n.new("ShaderNodeMath")
    mod.operation = "PINGPONG"
    mod.inputs[1].default_value = 0.04
    ln.new(sep.outputs["Z"], mod.inputs[0])
    gt = n.new("ShaderNodeMath")
    gt.operation = "GREATER_THAN"
    gt.inputs[1].default_value = 0.012
    ln.new(mod.outputs[0], gt.inputs[0])
    mr = n.new("ShaderNodeMapRange")
    mr.inputs["To Min"].default_value = 0.03
    mr.inputs["To Max"].default_value = 0.12
    ln.new(gt.outputs[0], mr.inputs["Value"])
    b.inputs["Emission Color"].default_value = (1.0, 0.68, 0.36, 1)
    ln.new(mr.outputs["Result"], b.inputs["Emission Strength"])
    m["night_emission"] = 1.0
    m["emission_via_link"] = True
    return m


def fence_wire_material():
    """Poly Haven chain-link wire material (CC0), with UVs we generate in metres (UV_PER_M)."""
    blend = CACHE / "modular_chainlink_fence" / "modular_chainlink_fence_1k.blend"
    with bpy.data.libraries.load(str(blend), link=False) as (src, dst):
        dst.materials = [n for n in src.materials if n == "modular_chainlink_fence_wire"]
    m = dst.materials[0]
    m.name = "fence_wire"
    # The PH wire albedo is dark (~0.08); galvanised chain link reads light grey under lot lights. Keep its
    # alpha and normal maps, replace colour/metal/roughness with galvanised values.
    b = m.node_tree.nodes["Principled BSDF"]
    for k, v in (("Base Color", (0.42, 0.43, 0.44, 1)), ("Metallic", 0.6), ("Roughness", 0.5)):
        for link in list(b.inputs[k].links):
            m.node_tree.links.remove(link)
        b.inputs[k].default_value = v
    try:
        m.surface_render_method = "DITHERED"
    except AttributeError:
        pass
    return m


UV_PER_M = 0.197  # measured from the Poly Haven fence panel: 0.17 UV over 0.863 m


# ---- ground ------------------------------------------------------------------------


def plane(name, x0, x1, y0, y1, z, mat, col):
    mb = H.MeshBuilder()
    mb.quad([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], mat)
    return mb.to_object(name, col, smooth_angle=None)


def ground(M, col):
    plane("surround", -250, 250, -250, 250, -0.03, M["gravel"], col)
    plane("lot_surface", -L.HX - 0.5, L.HX + 0.5, -L.HY - 0.3, L.HY + 0.5, 0.0, M["asphalt"], col)
    plane("street", -250, 250, -38.5, -28.0, -0.005, M["street"], col)
    mb = H.MeshBuilder()  # sidewalk with a curb
    mb.box((0, -26.8, -0.0), (250, 1.2, 0.06), M["concrete"])
    mb.box((0, -28.0, -0.02), (250, 0.08, 0.1), M["concrete"])
    mb.box((-7.0, -27.0, 0.005), (4.0, 2.0, 0.06), M["concrete"])  # front gate apron
    mb.to_object("sidewalk", col)
    # street markings: double yellow centre line, white edge lines, a manhole
    mb = H.MeshBuilder()
    for dy in (-0.08, 0.08):
        mb.box((0, -33.25 + dy, 0.0), (250, 0.05, 0.004), M["paint_yellow"])
    mb.box((0, -28.4, 0.0), (250, 0.06, 0.004), M["paint_white"])
    mb.box((0, -38.1, 0.0), (250, 0.06, 0.004), M["paint_white"])
    mb.to_object("street_lines", col, smooth_angle=None)


def stall_paint(M, col):
    """Stall stripes (worn), wheel stops, oil stains, and an aisle arrow."""
    lines = H.MeshBuilder()
    stops = H.MeshBuilder()
    for row, y in L.ROW_Y.items():
        xs = [L.STALL_X[0] - L.STALL_PITCH / 2 + k * L.STALL_PITCH for k in range(len(L.STALL_X) + 1)]
        for x in xs:
            lines.box((x, y, 0.0015), (0.05, L.STALL_DEPTH / 2, 0.0015), M["paint_white"])
        nose = L.ROW_NOSE_YAW[row]
        back = y + (L.STALL_DEPTH / 2 - 0.55) * (1 if nose == 180.0 else -1)
        for x in L.STALL_X:
            stops.box((x, back, 0.065), (0.9, 0.08, 0.065), M["concrete"], bevel=0.02)
    # display line and staff stalls
    for x in (13.0, 17.1, 21.3, 25.5, 29.7, 33.9):
        lines.box((x, -20.6, 0.0015), (0.05, 2.75, 0.0015), M["paint_white"])
    for y in (-15.45, -18.35, -21.25):  # staff stalls west of the office
        lines.box((-36.4, y, 0.0015), (2.75, 0.05, 0.0015), M["paint_white"])
    # painted arrows at the front gate entrance
    lines.box((-7.0, -21.0, 0.0015), (0.08, 1.2, 0.0015), M["paint_white"])
    for s_ in (-1, 1):  # arrow head
        lines.box((-7.0 + s_ * 0.22, -19.95, 0.0015), (0.3, 0.07, 0.0015), M["paint_white"],
                  matrix=Matrix.Translation((-7.0, -19.8, 0)) @ Matrix.Rotation(s_ * 0.75, 4, "Z")
                  @ Matrix.Translation((7.0, 19.8, 0)))
    lines.to_object("stall_lines", col, smooth_angle=None)
    stops.to_object("wheel_stops", col)
    # oil stains: under most stalls (visible in the open one and around tails), plus a few in the aisles
    for i, (spot, (row, x, y, yaw)) in enumerate(sorted(L.SPOTS.items())):
        for k in range(RNG.choice([1, 1, 2])):
            dx, dy = RNG.uniform(-0.5, 0.5), RNG.uniform(-1.0, 1.2) * (1 if yaw == 0 else -1)
            o = plane(f"oil_{spot}_{k}", -1, 1, -1, 1, 0.0, M["oil"], col)
            o.location = (x + dx, y + dy, 0.003 + 0.0005 * k)
            s = RNG.uniform(0.35, 0.8)
            o.scale = (s, s * RNG.uniform(0.7, 1.3), 1)
            o.rotation_euler = (0, 0, RNG.uniform(0, 3.1))
    for x, y in [(-6.0, -8.5), (3.0, 7.6), (-7.2, -22.5), (22.0, -6.5), (20.5, -4.0)]:
        o = plane(f"oil_aisle_{x}_{y}", -1, 1, -1, 1, 0.0, M["oil"], col)
        o.location = (x, y, 0.004)
        o.scale = (1.1, 0.8, 1)


# ---- fence and gates -----------------------------------------------------------------


def fabric(name, a, b, z0, z1, M, col):
    """Chain-link fabric plane from a to b (xy), UVs in metres so the PH wire texture keeps its scale."""
    me = bpy.data.meshes.new(name)
    (x0, y0), (x1, y1) = a, b
    length = math.hypot(x1 - x0, y1 - y0)
    verts = [(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)]
    me.from_pydata(verts, [], [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name="UVMap")
    for li, (u, v) in zip(range(4), [(0, z0), (length, z0), (length, z1), (0, z1)]):
        uv.data[li].uv = (u * UV_PER_M + 0.2, v * UV_PER_M + 0.45)
    me.materials.append(M["wire"])
    o = bpy.data.objects.new(name, me)
    col.objects.link(o)
    return o


def fence_runs():
    """Yield (side, start xy, end xy, outward normal) for each straight fence segment between gaps."""
    sides = {
        "north": ((-L.HX, L.HY), (1, 0), 2 * L.HX, (0, 1)),
        "east": ((L.HX, -L.HY), (0, 1), 2 * L.HY, (1, 0)),
        "south": ((-L.HX, -L.HY), (1, 0), 2 * L.HX, (0, -1)),
        "west": ((-L.HX, L.HY), (0, -1), 2 * L.HY, (-1, 0)),
    }
    for side, (start, d, length, normal) in sides.items():
        cuts = [0.0]
        for g0, g1 in L.FENCE_GAPS.get(side, []):
            cuts += [g0, g1]
        cuts.append(length)
        for s0, s1 in zip(cuts[::2], cuts[1::2]):
            a = (start[0] + d[0] * s0, start[1] + d[1] * s0)
            b = (start[0] + d[0] * s1, start[1] + d[1] * s1)
            yield side, a, b, normal


def barb_arm_strands(mb_posts, wires, p, normal, M, z_top):
    """45-degree arm leaning out of the lot with three strands' attachment points."""
    nx, ny = normal
    arm_len = 0.48
    top = (p[0] + nx * arm_len * 0.707, p[1] + ny * arm_len * 0.707, z_top + arm_len * 0.707)
    mb_posts.tube((p[0], p[1], z_top - 0.02), top, 0.012, M["galv"], segs=6)
    pts = []
    for k in range(3):
        t = (k + 1) / 3.0
        pts.append((p[0] + nx * arm_len * 0.707 * t, p[1] + ny * arm_len * 0.707 * t, z_top + arm_len * 0.707 * t))
    return pts


def fence(M, col):
    posts = H.MeshBuilder()
    rails = H.MeshBuilder()
    strand_pts: dict[tuple, list] = {}
    for side, a, b, normal in fence_runs():
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        dx, dy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        n_posts = max(1, round(length / 3.0))
        strands = [[], [], []]
        for k in range(n_posts + 1):
            s = length * k / n_posts
            p = (a[0] + dx * s, a[1] + dy * s)
            terminal = k in (0, n_posts)
            r = 0.037 if terminal else 0.024
            posts.cylinder((p[0], p[1], (L.FENCE_H + 0.05) / 2), r, L.FENCE_H + 0.05, M["galv"], segs=10)
            posts.cylinder((p[0], p[1], L.FENCE_H + 0.06), r + 0.01, 0.04, M["galv"], segs=10)  # cap
            pts = barb_arm_strands(posts, None, p, normal, M, L.FENCE_H + 0.04)
            for i in range(3):
                strands[i].append(pts[i])
        rails.tube((a[0], a[1], L.FENCE_H - 0.03), (b[0], b[1], L.FENCE_H - 0.03), 0.021, M["galv"], segs=8)
        rails.tube((a[0], a[1], 0.08), (b[0], b[1], 0.08), 0.004, M["galv"], segs=4)
        fabric(f"fabric_{side}_{a[0]:.0f}_{a[1]:.0f}", a, b, 0.05, L.FENCE_H - 0.02, M, col)
        for i, s in enumerate(strands):
            strand_pts[(side, a, i)] = s
    posts.to_object("fence_posts", col)
    rails.to_object("fence_rails", col)
    barb = H.principled("barbed_wire", (0.35, 0.36, 0.37), rough=0.5, metal=0.9)
    for key, pts in strand_pts.items():
        H.curve_wire(f"barb_{key[0]}_{key[1][0]:.0f}_{key[1][1]:.0f}_{key[2]}", pts, 0.004, barb, col)


def gate_leaf(name, length, M, col, along="x", tail=0.0):
    """Gate leaf in its local frame, starting at the origin and running +x (or +y)."""
    mb = H.MeshBuilder()
    h = L.FENCE_H - 0.1
    z0 = 0.08
    total0 = -tail
    if along == "x":
        def P(s, z):
            return (s, 0.0, z)
    else:
        def P(s, z):
            return (0.0, s, z)
    for s in (total0, length):
        mb.tube(P(s, z0), P(s, z0 + h), 0.03, M["galv"])
    for z in (z0, z0 + h, z0 + h / 2):
        mb.tube(P(total0, z), P(length, z), 0.025, M["galv"])
    for s0, s1 in ((0, length / 2), (length / 2, length)):
        mb.tube(P(s0, z0), P(s1, z0 + h), 0.015, M["galv"])
    if tail:
        mb.tube(P(total0, z0), P(0, z0 + h), 0.02, M["galv"])
    # barbed arms on the leaf top
    strands = [[], [], []]
    for k in range(4):
        s = total0 + (length - total0) * k / 3
        p = P(s, 0)
        normal = (0, -1) if along == "x" else (-1, 0)
        pts = barb_arm_strands(mb, None, (p[0], p[1]), normal, M, z0 + h)
        for i in range(3):
            strands[i].append(pts[i])
    o = mb.to_object(name + "_frame", col)
    a = P(total0, 0)
    b = P(length, 0)
    fab = fabric(name + "_fabric", (a[0], a[1]), (b[0], b[1]), z0 + 0.03, z0 + h - 0.03, M, col)
    barb = bpy.data.materials.get("barbed_wire")
    wires = [H.curve_wire(f"{name}_barb_{i}", s, 0.004, barb, col) for i, s in enumerate(strands)]
    return [o, fab, *wires]


def gates(M, col):
    for gid, g in L.GATES.items():
        root = H.empty(gid, col, (g["origin"][0], g["origin"][1], 0.0))
        along = "x" if g["axis"][0] else "y"
        parts = gate_leaf(gid, g["length"], M, col, along=along, tail=2.4 if g["type"] == "slide" else 0.0)
        for p in parts:
            p.parent = root
        root["gate_type"] = g["type"]
    # gate posts (fixed): yellow-painted terminal posts at both sides of each opening, rollers for the slider
    mb = H.MeshBuilder()
    for x in (-10.0, -4.0):
        mb.cylinder((x, -25.0, 1.3), 0.05, 2.6, M["yellow_steel"], segs=12)
    for x in (-11.0, -13.6):
        mb.cylinder((x, -24.62, 1.25), 0.05, 2.5, M["galv"], segs=12)
        mb.box((x, -24.75, 0.12), (0.09, 0.09, 0.06), M["dark_steel"])
        mb.box((x, -24.75, 2.4), (0.09, 0.09, 0.06), M["dark_steel"])
    for y in (-14.0, -8.0):
        mb.cylinder((-40.0, y, 1.3), 0.05, 2.6, M["yellow_steel"], segs=12)
    mb.to_object("gate_posts", col)


# ---- buildings and structures ------------------------------------------------------


def office(M, col):
    o = L.OFFICE
    cx, cy = o["center"]
    hx, hy = o["half"]
    h = o["height"]
    mb = H.MeshBuilder()
    mb.box((cx, cy, h / 2), (hx, hy, h / 2), M["plaster"])
    mb.box((cx, cy, h + o["parapet"] / 2), (hx + 0.05, hy + 0.05, o["parapet"] / 2), M["plaster"])
    mb.box((cx, cy, h + o["parapet"] + 0.03), (hx + 0.1, hy + 0.1, 0.03), M["trim"])
    mb.box((cx, cy, h + 0.05), (hx - 0.1, hy - 0.1, 0.05), M["roof"])
    mb.box((cx, cy, 0.15), (hx + 0.03, hy + 0.03, 0.15), M["concrete"])  # base course
    yN = cy + hy
    for i, (wx, ww) in enumerate(o["windows"]):
        lit = i in o["lit_windows"]
        mb.box((wx, yN + 0.015, 1.65), (ww / 2, 0.015, 0.75), M["window_lit"] if lit else M["glass"])
        for dz in (-0.78, 0.78):
            mb.box((wx, yN + 0.04, 1.65 + dz), (ww / 2 + 0.05, 0.04, 0.04), M["alu"])
        for dx in (-ww / 2 - 0.02, ww / 2 + 0.02):
            mb.box((wx + dx, yN + 0.04, 1.65), (0.04, 0.04, 0.8), M["alu"])
        mb.box((wx, yN + 0.12, 0.86), (ww / 2 + 0.08, 0.1, 0.03), M["concrete"])  # sill
    for wx in (-30.0, -24.0):  # back windows
        mb.box((wx, cy - hy - 0.015, 1.65), (0.9, 0.015, 0.75), M["glass"])
    dx = o["door_x"]
    mb.box((dx, yN + 0.015, 1.1), (0.5, 0.015, 1.1), M["door_lit"])
    mb.box((dx, yN + 0.05, 2.25), (0.6, 0.05, 0.05), M["alu"])
    for s in (-0.55, 0.55):
        mb.box((dx + s, yN + 0.05, 1.1), (0.05, 0.05, 1.15), M["alu"])
    mb.box((dx, yN + 0.9, 2.8), (1.2, 0.9, 0.06), M["trim"])  # door canopy
    mb.box((dx, yN + 1.0, 0.07), (1.4, 1.0, 0.07), M["concrete"])  # stoop
    for x in (cx - hx - 0.06, cx + hx + 0.06):  # downspouts
        mb.box((x, yN - 0.3, h / 2), (0.05, 0.05, h / 2), M["trim"])
    mb.to_object("office", col)
    sign = H.text("office_sign", "OFFICE", 0.55, M["sign_white"], col, (cx + 1.0, yN + 0.06, h + 0.0), (90, 0, 180))
    sign.data.extrude = 0.03
    # rooftop AC from Poly Haven
    ac = [p for p in H.append_objects(CACHE / "exterior_aircon_unit" / "exterior_aircon_unit_1k.blend",
                                      ["exterior_aircon_unit"])]
    for k, p in enumerate(ac):
        col.objects.link(p)
        p.location = (cx - 2.0, cy - 1.0, h + 0.1)
        p.scale = (1.5, 1.5, 1.2)
    can = H.append_objects(CACHE / "metal_trash_can" / "metal_trash_can_1k.blend",
                           ["metal_trash_can", "metal_trash_can_lid", "metal_trash_can_handle_left",
                            "metal_trash_can_handle_right"])
    for p in can:
        col.objects.link(p)
        p.location = (p.location.x - 0.5 + dx + 1.2, p.location.y + yN + 0.6, p.location.z)


def service_bay(M, col):
    b = L.SERVICE_BAY
    cx, cy = b["center"]
    hx, hy = b["half"]
    h = b["height"]
    mb = H.MeshBuilder()
    mb.box((cx, cy, h / 2), (hx, hy, h / 2), M["metal_wall"])
    # low gable roof, ridge along x
    roof = [(-hy - 0.25, h - 0.05), (hy + 0.25, h - 0.05), (0, h + 0.55)]
    mb.profile(roof, cx - hx - 0.25, cx + hx + 0.25, M["metal_wall"])
    mb.box((cx, cy, 0.2), (hx + 0.02, hy + 0.02, 0.2), M["concrete"])
    x_w = cx - hx
    for y in b["doors"]:
        for s in (-1, 1):
            mb.box((x_w - 0.06, y + s * (b["door_w"] / 2 + 0.08), b["door_h"] / 2), (0.06, 0.08, b["door_h"] / 2 + 0.05),
                   M["trim"])
        mb.box((x_w - 0.06, y, b["door_h"] + 0.1), (0.06, b["door_w"] / 2 + 0.16, 0.1), M["trim"])
        mb.box((x_w - 0.3, y, 0.02), (0.3, b["door_w"] / 2 + 0.3, 0.02), M["concrete"])  # door apron
    mb.box((cx + 2.0, cy + hy + 0.02, 1.05), (0.46, 0.02, 1.05), M["dark_steel"])  # man door, north wall
    for x in (cx - hx - 0.06, cx + hx + 0.06):
        mb.box((x, cy - hy + 0.3, h / 2), (0.05, 0.05, h / 2), M["trim"])
    mb.to_object("service_bay", col)
    # roll-up doors: Poly Haven rollershutter_door (CC0), scaled from 1.08 x 2.4 m to the bay openings
    blend = CACHE / "rollershutter_door" / "rollershutter_door_1k.blend"
    for i, y in enumerate(b["doors"]):
        d = H.append_objects(blend, ["rollershutter_door"])[0]
        col.objects.link(d)
        d.name = f"bay_door_{i}"
        d.rotation_euler = (0, 0, math.radians(90))
        d.scale = (b["door_w"] / 1.08, 1.0, b["door_h"] / 2.4)
        d.location = (x_w - 0.02, y, 0.0)
    H.text("bay_sign", "SERVICE", 0.6, M["sign_white"], col, (x_w - 0.04, cy, h - 0.75), (90, 0, -90)).data.extrude = 0.03
    ub = H.append_objects(CACHE / "utility_box_01" / "utility_box_01_1k.blend")
    for p in ub:
        col.objects.link(p)
        p.location = (cx - 2.0, cy + hy + 0.3, 0.0)
    tyre_blend = CACHE / "old_tyre" / "old_tyre_1k.blend"
    for sx, sy in L.TIRE_STACKS:
        for k in range(RNG.choice([3, 4, 5])):
            t = H.append_objects(tyre_blend)[0]
            col.objects.link(t)
            t.rotation_euler = (math.radians(90), 0, RNG.uniform(0, 3))
            t.location = (sx + RNG.uniform(-0.03, 0.03), sy + RNG.uniform(-0.03, 0.03), 0.08 + 0.16 * k)


def fuel_tank(M, col):
    t = L.FUEL_TANK
    cx, cy = t["center"]
    r = t["diameter"] / 2
    ln = t["length"]
    zc = 0.3 + r
    mb = H.MeshBuilder()
    px, py = t["pad_half"]
    mb.box((cx, cy, 0.08), (px, py, 0.08), M["concrete"], bevel=0.02)
    mb.cylinder((cx, cy, zc), r, ln, M["tank"], axis="x", segs=32, bevel=0.12)
    for s in (-1, 1):
        mb.box((cx + s * ln * 0.3, cy, 0.25), (0.12, r * 0.8, 0.17), M["dark_steel"])
    mb.box((cx + 0.6, cy, zc + r + 0.25), (0.22, 0.18, 0.28), M["dark_steel"])  # pump
    mb.tube((cx + 0.6, cy - 0.18, zc + r + 0.4), (cx + 0.9, cy - r - 0.15, 0.9), 0.02, M["dumpster_lid"])
    mb.box((cx - 0.9, cy, zc + r + 0.06), (0.15, 0.15, 0.06), M["dark_steel"])  # vent / fill
    for bx, by in t["bollards"]:
        mb.cylinder((bx, by, 0.55), 0.085, 1.1, M["yellow_steel"], segs=12, bevel=0.03)
    mb.to_object("fuel_tank", col)
    H.text("tank_label", "DIESEL", 0.28, M["sign_red"], col, (cx, cy - r - 0.02, zc), (90, 0, 0)).data.extrude = 0.005
    H.text("tank_label2", "NO SMOKING", 0.12, M["sign_red"], col, (cx, cy - r - 0.03, zc - 0.32), (90, 0, 0))
    drum_blend = CACHE / "barrel_03" / "barrel_03_1k.blend"
    for x, y in L.DRUMS:
        d = H.append_objects(drum_blend)[0]
        col.objects.link(d)
        d.location = (x, y, 0.0)
        d.rotation_euler = (0, 0, RNG.uniform(0, 6))


def dumpsters(M, col):
    for i, (x, y, yaw, color) in enumerate(L.DUMPSTERS):
        mb = H.MeshBuilder()
        body = H.principled(f"dumpster_{i}", color, rough=0.6, metal=0.3)
        prof = [(-0.61, 0.12), (0.61, 0.12), (0.75, 1.2), (-0.61, 1.4)]  # slanted front (+y)
        mb.profile(prof, -0.92, 0.92, body, bevel=0.03)
        mb.box((0, 0.07, 1.33), (0.9, 0.7, 0.03), M["dumpster_lid"], matrix=Matrix.Rotation(-0.15, 4, "X"))
        for s in (-1, 1):
            mb.box((s * 0.96, 0.0, 0.85), (0.05, 0.35, 0.08), M["dark_steel"])  # fork pockets
        for s in (-1, 1):
            for yy in (-0.5, 0.5):
                mb.cylinder((s * 0.75, yy, 0.07), 0.06, 0.05, M["dark_steel"], axis="x")
        o = mb.to_object(f"dumpster_{i}", col)
        o.location = (x, y, 0)
        o.rotation_euler = (0, 0, math.radians(yaw))


# ---- lights, poles, signs, cameras ------------------------------------------------------

LIGHT_POWER = {"led": 1800.0, "hps": 2200.0, "wallpack": 260.0, "street": 1500.0}
LIGHT_COLOR = {"led": (1.0, 0.89, 0.78), "hps": (1.0, 0.56, 0.2), "wallpack": (1.0, 0.92, 0.84),
               "street": (1.0, 0.8, 0.6)}


def spot(name, loc, aim_yaw_deg, tilt_deg, kind, col, size_deg=150.0, blend=0.6, radius=0.18):
    ld = bpy.data.lights.new(name, "SPOT")
    ld.energy = LIGHT_POWER[kind]
    ld.color = LIGHT_COLOR[kind]
    ld.spot_size = math.radians(size_deg)
    ld.spot_blend = blend
    ld.shadow_soft_size = radius
    ld["night_energy"] = ld.energy
    o = bpy.data.objects.new(name, ld)
    o.location = loc
    # spot points down local -z; tilt toward the aim yaw
    yaw = math.radians(aim_yaw_deg)
    aim = Vector((math.cos(yaw) * math.sin(math.radians(tilt_deg)), math.sin(yaw) * math.sin(math.radians(tilt_deg)),
                  -math.cos(math.radians(tilt_deg))))
    o.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
    col.objects.link(o)
    return o


def light_poles(M, lights_col, col):
    mb = H.MeshBuilder()
    for i, (x, y, heads, kind) in enumerate(L.POLE_LIGHTS):
        mb.cylinder((x, y, 0.35), 0.3, 0.7, M["concrete"], segs=16)  # pier base
        mb.box((x, y, (L.POLE_H + 0.7) / 2), (0.075, 0.075, (L.POLE_H - 0.7) / 2), M["bronze"])
        for k, hy in enumerate(heads):
            a = math.radians(hy)
            ax, ay = math.cos(a), math.sin(a)
            hx, hyy = x + ax * 0.75, y + ay * 0.75
            mb.tube((x, y, L.POLE_H - 0.15), (hx, hyy, L.POLE_H - 0.15), 0.04, M["bronze"])
            rot = Matrix.Translation((hx, hyy, L.POLE_H - 0.15)) @ Matrix.Rotation(a, 4, "Z")
            mb.box((0.15, 0, 0), (0.33, 0.21, 0.06), M["bronze"], matrix=rot, bevel=0.02)
            lens = M["hps_lens"] if kind == "hps" else M["led_lens"]
            mb.box((0.15, 0, -0.065), (0.27, 0.16, 0.006), lens, matrix=rot)
            spot(f"pole{i + 1}_{k}", (hx + ax * 0.15, hyy + ay * 0.15, L.POLE_H - 0.25), hy, 22.0, kind, lights_col,
                 size_deg=140.0, blend=0.45, radius=0.2)
    mb.to_object("light_poles", col)
    # wall packs
    mb = H.MeshBuilder()
    for i, (x, y, z, yaw) in enumerate(L.WALL_PACKS):
        a = math.radians(yaw)
        ax, ay = math.cos(a), math.sin(a)
        rot = Matrix.Translation((x + ax * 0.12, y + ay * 0.12, z)) @ Matrix.Rotation(a, 4, "Z")
        mb.box((0, 0, 0), (0.12, 0.17, 0.13), M["bronze"], matrix=rot, bevel=0.02)
        mb.box((0.11, 0, -0.02), (0.02, 0.13, 0.08), M["wallpack_lens"], matrix=rot)
        spot(f"wallpack{i + 1}", (x + ax * 0.3, y + ay * 0.3, z - 0.05), yaw, 55.0, "wallpack", lights_col,
             size_deg=140.0, blend=0.7, radius=0.08)
    mb.to_object("wall_packs", col)
    # street lights (cobra heads over the road)
    mb = H.MeshBuilder()
    for i, (x, y) in enumerate(L.STREET_LIGHTS):
        mb.cylinder((x, y, 4.5), 0.11, 9.0, M["galv"], segs=12, radius2=0.07)
        mb.tube((x, y, 8.8), (x, y + 2.2, 9.1), 0.04, M["galv"])
        mb.box((x, y + 2.6, 9.05), (0.22, 0.45, 0.08), M["galv"], bevel=0.04)
        mb.box((x, y + 2.6, 8.96), (0.16, 0.32, 0.01), M["street_lens"])
        spot(f"street{i + 1}", (x, y + 2.6, 8.9), 90.0, 10.0, "street", lights_col, size_deg=145.0, blend=0.6)
    mb.to_object("street_lights", col)


def pole_sign(M, col):
    s = L.POLE_SIGN
    x, y = s["pos"]
    w, hh = s["size"]
    z = s["height"]
    mb = H.MeshBuilder()
    for dx in (-w / 2 + 0.3, w / 2 - 0.3):
        mb.box((x + dx, y, z / 2), (0.1, 0.1, z / 2), M["dark_steel"])
    mb.box((x, y, z + hh / 2), (w / 2 + 0.08, 0.2, hh / 2 + 0.08), M["dark_steel"])
    for side in (-1, 1):
        mb.box((x, y + side * 0.205, z + hh / 2), (w / 2, 0.005, hh / 2), M["sign_lit"])
    mb.to_object("pole_sign", col)
    for side in (-1, 1):
        for k, line in enumerate(s["lines"]):
            rz = 0 if side < 0 else 180
            t = H.text(f"pole_sign_txt_{side}_{k}", line, 0.26 if k == 0 else 0.22, M["sign_lit_red"], col,
                       (x, y + side * 0.215, z + hh * (0.68 - 0.38 * k)), (90, 0, rz))
            t.data.extrude = 0.0


def fence_signs(M, col):
    """NO TRESPASSING / VIDEO SURVEILLANCE placards, mostly facing out, a few facing in at the gates."""
    spots = [((-25.0, -25.05), (0, -1)), ((10.0, -25.05), (0, -1)), ((30.0, -25.05), (0, -1)),
             ((-40.05, 5.0), (-1, 0)), ((40.05, 10.0), (1, 0)), ((-20.0, 25.05), (0, 1)), ((20.0, 25.05), (0, 1)),
             ((-12.0, -24.95), (0, 1)), ((-40.0 + 0.05, -16.0), (1, 0)), ((40.0 - 0.05, -10.0), (-1, 0))]
    for i, ((x, y), (nx, ny)) in enumerate(spots):
        mb = H.MeshBuilder()
        yaw = math.atan2(ny, nx)
        rot = Matrix.Translation((x, y, 1.55)) @ Matrix.Rotation(yaw - math.pi / 2, 4, "Z")
        mb.box((0, 0.012, 0), (0.3, 0.004, 0.2), M["sign_white"], matrix=rot)
        mb.box((0, 0.0, 0), (0.3, 0.006, 0.2), M["sign_back"], matrix=rot)
        mb.box((0, 0.017, 0.12), (0.28, 0.002, 0.06), M["sign_red"], matrix=rot)
        mb.to_object(f"fence_sign_{i}", col, smooth_angle=None)
        t = H.text(f"fence_sign_txt_{i}", "NO TRESPASSING\nVIDEO SURVEILLANCE", 0.045, M["sign_black"], col,
                   (x + nx * 0.02, y + ny * 0.02, 1.48), (90, 0, math.degrees(math.atan2(nx, -ny))))
        t.data.extrude = 0.0


def camera_poles(M, col):
    mb = H.MeshBuilder()
    for name, (mount, target, lens) in L.CAMERAS.items():
        mx, my, mz = mount
        mb.cylinder((mx, my, (mz + 0.3) / 2), 0.06, mz + 0.3, M["galv"], segs=10)
        mb.box((mx, my, 2.6), (0.12, 0.08, 0.15), M["sign_back"])  # junction box
        d = Vector(target) - Vector(mount)
        d.normalize()
        # bullet camera: arm out of the pole toward the target, housing along the view direction
        arm_end = Vector((mx, my, mz)) + Vector((d.x, d.y, 0)).normalized() * 0.25
        mb.tube((mx, my, mz), arm_end, 0.025, M["sign_white"], segs=8)
        body_c = arm_end + d * 0.12 + Vector((0, 0, 0.06))
        rot = d.to_track_quat("Y", "Z").to_matrix().to_4x4()
        m = Matrix.Translation(body_c) @ rot
        mb.box((0, 0, 0), (0.05, 0.13, 0.05), M["sign_white"], matrix=m, bevel=0.02)
        mb.box((0, 0.02, 0.06), (0.065, 0.16, 0.008), M["sign_white"], matrix=m)  # sun shield
        mb.cylinder((0, 0.131, 0), 0.032, 0.004, M["glass"], axis="y", matrix=m)
    mb.to_object("camera_poles", col)


def cameras(col):
    import camera_model as CM
    for name, (mount, target, lens) in L.CAMERAS.items():
        cd = bpy.data.cameras.new(name)
        hfov, vfov = CM.source_fov(lens)
        cd.sensor_fit = "HORIZONTAL"
        cd.angle = hfov
        cd.clip_start = 0.1
        cd.clip_end = 600
        o = bpy.data.objects.new(name, cd)
        d = Vector(target) - Vector(mount)
        o.location = Vector(mount) + d.normalized() * 0.35
        o.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        o["lens_mm"] = lens
        col.objects.link(o)


# ---- backdrop ---------------------------------------------------------------------------


def backdrop(M, col, lights_col):
    mb = H.MeshBuilder()
    # neighbour warehouse to the north, a low building across the street, a block wall to the west
    mb.box((5.0, 48.0, 4.0), (28.0, 12.0, 4.0), M["concrete"])
    mb.box((5.0, 48.0, 8.05), (28.2, 12.2, 0.05), M["trim"])
    # across the street: a strip of small businesses with storefront glass, roll-up doors and a parapet sign band
    mb.box((10.0, -55.0, 2.6), (22.0, 8.0, 2.6), M["plaster"])
    mb.box((10.0, -46.95, 4.4), (22.0, 0.08, 0.5), M["trim"])
    for k in range(6):
        x = -8.0 + k * 7.2
        mb.box((x, -46.98, 1.4), (1.6, 0.03, 1.1), M["glass"])
        mb.box((x + 2.4, -46.98, 1.5), (0.5, 0.03, 1.2), M["glass"] if k % 2 else M["dark_steel"])
        if k in (1, 4):
            mb.box((x, -46.9, 0.06), (1.8, 0.1, 0.06), M["concrete"])
    mb.box((-16.0, -46.98, 1.8), (1.6, 0.04, 1.8), M["metal_wall"])
    mb.box((36.0, -46.98, 1.8), (1.6, 0.04, 1.8), M["metal_wall"])
    mb.box((-48.0, 0.0, 0.9), (0.15, 40.0, 0.9), M["concrete"])
    mb.box((60.0, -5.0, 3.0), (10.0, 18.0, 3.0), M["metal_wall"])
    for i, x in enumerate((-12.0, 6.0, 24.0)):
        mb.box((x, 35.95, 5.0), (0.15, 0.1, 0.12), M["bronze"])
        mb.box((x, 35.88, 4.95), (0.1, 0.01, 0.07), M["wallpack_lens"])
        spot(f"neighbour_wp{i}", (x, 35.7, 4.85), 270.0, 50.0, "wallpack", lights_col, size_deg=130, blend=0.7)
    mb.to_object("backdrop", col)
    # a few dark trees along the west and north-west
    for i, (x, y, s) in enumerate([(-46.0, 20.0, 1.0), (-45.0, 12.0, 0.8), (-46.5, -3.0, 1.1), (-44.0, 30.0, 1.2),
                                   (-30.0, 31.0, 0.9), (-20.0, 32.0, 1.0), (46.0, 22.0, 1.0)]):
        tb = H.MeshBuilder()
        tb.cylinder((0, 0, 1.2 * s), 0.15 * s, 2.4 * s, M["bark"], segs=8)
        import bmesh
        for k in range(3):
            res = bmesh.ops.create_icosphere(tb.bm, subdivisions=2, radius=1.0)
            m = Matrix.Translation((RNG.uniform(-0.6, 0.6) * s, RNG.uniform(-0.6, 0.6) * s, (3.0 + k * 0.9) * s)) @ \
                Matrix.Diagonal((1.9 * s, 1.7 * s, 1.4 * s, 1))
            bmesh.ops.transform(tb.bm, matrix=m, verts=res["verts"])
            for v in res["verts"]:
                v.co += Vector((RNG.uniform(-0.25, 0.25), RNG.uniform(-0.25, 0.25), RNG.uniform(-0.2, 0.2))) * s
            for f in {f for v in res["verts"] for f in v.link_faces}:
                f.material_index = tb.slot(M["tree"])
        o = tb.to_object(f"tree_{i}", col, smooth_angle=60)
        o.location = (x, y, 0)
    hyd = H.append_objects(CACHE / "fire_hydrant" / "fire_hydrant_1k.blend",
                           ["fire_hydrant", "fire_hydrant_cap_01", "fire_hydrant_cap_02", "fire_hydrant_cap_03",
                            "fire_hydrant_chain"])
    for p in hyd:
        col.objects.link(p)
        p.location = (p.location.x + 0.3 + 2.0, p.location.y - 27.4, 0.0)


def world() -> None:
    w = bpy.data.worlds.new("lot_world")
    bpy.context.scene.world = w
    try:
        w.use_nodes = True
    except Exception:
        pass
    nt = w.node_tree
    bg = nt.nodes.get("Background")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.name = "env"
    env.image = H.image(CACHE / "kloppenheim_02_puresky" / "kloppenheim_02_puresky_2k.hdr")
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.6
    day = H.image(CACHE / "kloofendal_overcast_puresky" / "kloofendal_overcast_puresky_2k.hdr")
    w["night_hdri"] = env.image.name
    w["day_hdri"] = day.name
    day.use_fake_user = True


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else CACHE / "lot.blend"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    root = scene.collection
    statics = H.collection("statics", root)
    dyn = H.collection("dynamic", root)
    lights_col = H.collection("lights", root)
    cams = H.collection("cameras", root)

    M = materials()
    ground(M, statics)
    stall_paint(M, statics)
    fence(M, statics)
    gates(M, dyn)
    office(M, statics)
    service_bay(M, statics)
    fuel_tank(M, statics)
    dumpsters(M, statics)
    light_poles(M, lights_col, statics)
    pole_sign(M, statics)
    fence_signs(M, statics)
    camera_poles(M, statics)
    backdrop(M, statics, lights_col)
    for v in L.vehicles():
        if v["kind"] == "covered_car":
            V.build_covered_car(v, dyn)
        else:
            V.build_vehicle(v, L.VEHICLE_DIMS, L.paint_rgb(v["paint"]), dyn)
    person = P.build_person(dyn)
    person.location = (*L.PERSON_PARK, 0.0)
    cameras(cams)
    world()
    scene["build_hash"] = source_hash()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=False)
    print(f"BUILD_OK {out} objects={len(bpy.data.objects)} hash={scene['build_hash']}")


if __name__ == "__main__":
    main()
