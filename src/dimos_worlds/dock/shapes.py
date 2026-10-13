"""Dock 3-class station + M4D-class quadcopter as triangle meshes, built with manifold3d CSG from spec.json.

Every visible surface comes from here: gen_mjcf.py writes these meshes for MuJoCo (assets/*.obj) and as arrays for
other renderers (assets/parts.npz), and web_glb.py builds the web GLB from them, so the sim and the renders cannot
drift. manifold3d runs anywhere; renderers only read the arrays.

    from dimos_worlds.dock.shapes import build_parts
    parts = build_parts(spec)   # [{"name", "parent", "mat", "verts" (N,3), "faces" (M,3)}]

Frames are geometry.py's (dock, cover_r, cover_l, act_cyl_*, act_rod_*, drone). Shapes follow DJI's dimensioned
drawings where the spec says DWG, DJI's published numbers where it says DJI, and are estimates elsewhere (the spec
labels each). Generic models: no logos, trademarks or third-party artwork on any surface.
"""
from __future__ import annotations

import math

import numpy as np
from manifold3d import CrossSection, JoinType, Manifold, Mesh, OpType

from .geometry import actuator_pins, drone_layout

SEG = 40  # circle segments for small parts


# ------------------------------------------------------------------------------------------------ materials

class _Reg:
    """original-id -> material, so one CSG result can carry several materials (outer shell vs liner)."""

    def __init__(self) -> None:
        self.mat: dict[int, str] = {}
        self.simplify = (0.0, 0)  # (tolerance m, only meshes with more triangles than this): build_parts sets it

    def tag(self, m: Manifold, mat: str) -> Manifold:
        m = m.as_original()
        self.mat[m.original_id()] = mat
        return m


R = _Reg()


def _simplify(m: Manifold) -> Manifold:
    """manifold3d's simplify() on heavy meshes only (web builds); thin details stay as built."""
    tol, n = R.simplify
    return m.simplify(tol) if tol > 0 and m.num_tri() > n else m


def export(m: Manifold, default: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split a manifold's triangles by the material of the primitive they came from."""
    mesh = _simplify(m).to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    f = np.asarray(mesh.tri_verts, dtype=np.int64)
    runs = list(mesh.run_index)
    ids = list(mesh.run_original_id)
    out: dict[str, list[np.ndarray]] = {}
    for k, oid in enumerate(ids):
        a, b = runs[k] // 3, runs[k + 1] // 3
        out.setdefault(R.mat.get(int(oid), default), []).append(f[a:b])
    res = {}
    for mat, chunks in out.items():
        ff = np.concatenate(chunks)
        used, inv = np.unique(ff.ravel(), return_inverse=True)
        res[mat] = (v[used], inv.reshape(-1, 3))
    return res


# ------------------------------------------------------------------------------------------------ helpers

def ccw(pts) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float64)
    area = 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return p if area > 0 else p[::-1].copy()


def poly(pts) -> CrossSection:
    return CrossSection([ccw(pts)])


def rrect(x0, y0, x1, y1, r, seg=24) -> CrossSection:
    if r <= 1e-6:
        return poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return CrossSection.square((x1 - x0 - 2 * r, y1 - y0 - 2 * r)).translate((x0 + r, y0 + r)).offset(
        r, JoinType.Round, 2.0, seg)


def ext_z(sec: CrossSection, z0: float, z1: float) -> Manifold:
    return sec.extrude(z1 - z0).translate((0.0, 0.0, z0))


def ext_y(sec_xz: CrossSection, y0: float, y1: float) -> Manifold:
    """A section drawn in (x, z), extruded across y0..y1."""
    return sec_xz.extrude(y1 - y0).transform(np.array([[1, 0, 0, 0], [0, 0, -1, y1], [0, 1, 0, 0]], dtype=float))


def ext_x(sec_yz: CrossSection, x0: float, x1: float) -> Manifold:
    """A section drawn in (y, z), extruded across x0..x1."""
    return sec_yz.extrude(x1 - x0).transform(np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float))


def box(lo, hi) -> Manifold:
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    return Manifold.cube(tuple(hi - lo)).translate(tuple(lo))


def frame_z(z, origin) -> np.ndarray:
    z = np.asarray(z, float)
    z = z / np.linalg.norm(z)
    t = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(t, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.zeros((3, 4))
    M[:, 0], M[:, 1], M[:, 2], M[:, 3] = x, y, z, origin
    return M


def cyl(a, b, r, r2=None, seg=SEG) -> Manifold:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    return Manifold.cylinder(float(np.linalg.norm(d)), r, r if r2 is None else r2, seg).transform(frame_z(d, a))


def sph(c, r, seg=SEG) -> Manifold:
    return Manifold.sphere(r, seg).translate(tuple(c))


def union(ms) -> Manifold:
    ms = [m for m in ms if m is not None and not m.is_empty()]
    return Manifold.batch_boolean(ms, OpType.Add) if ms else Manifold()


def tube(pts, r, seg=16) -> Manifold:
    """A round tube through a polyline (cylinders with sphere joints)."""
    pts = [np.asarray(p, float) for p in pts]
    return union([sph(p, r, seg) for p in pts] + [cyl(a, b, r, seg=seg) for a, b in zip(pts, pts[1:])])


def strip(pts_yz, x0, x1, t) -> Manifold:
    """A bent plate: polyline in (y, z) thickened by t, extruded across x0..x1."""
    segs = []
    for a, b in zip(pts_yz, pts_yz[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]]) * t / 2
        segs.append(ext_x(poly([a - n, b - n, b + n, a + n]), x0, x1))
    return union(segs)


def skin(base: Manifold, region: Manifold, t=0.0015) -> Manifold:
    """A t-thick layer on the upward-facing surface of `base`, inside `region` (decals that follow a surface)."""
    return (base.translate((0.0, 0.0, t)) ^ region) - base


# ------------------------------------------------------------------------------------------------ cover roof

def roof_base(x, y, s) -> np.ndarray:
    """Outer roof height without the edge rounds (DWG front + side)."""
    c = s["cover"]
    ax = np.abs(x)
    w = 0.02  # ridge crown half-width: the drawing's ridge is flat over X -11..+14 mm
    sx = np.where(ax < w, ax ** 2 / (2 * w) + w / 2, ax)
    y0, k = c["roof_crown_y"]
    crown = k * np.maximum(0.0, np.abs(y) - y0) ** 2
    return c["roof_ridge_z"] - c["roof_slope"] * sx - crown


def roof_outer(x, y, s) -> np.ndarray:
    c = s["cover"]
    z = roof_base(x, y, s)
    a = c["edge_round_y"]
    b = 0.012  # vertical drop of the front/back round: 619 -> 635 mm (DWG side)
    dy = np.clip(c["half_y"] - np.abs(y), 0.0, None)
    u = np.clip((a - dy) / a, 0.0, 1.0)
    z = z - b * (1.0 - np.sqrt(1.0 - u ** 2))
    ax, az = c["end_chamfer"]
    dx = np.clip(c["x_end"] - np.abs(x), 0.0, None)
    return z - az * np.clip(1.0 - dx / ax, 0.0, 1.0)


def roof_inner(x, y, s) -> np.ndarray:
    return roof_base(x, y, s) - s["cover"]["wall_t"]


def _roof_warp(fn, z0, ztop, s):
    def w(v):
        v = np.array(v, dtype=np.float64)
        hi = v[:, 2] > z0 + 1e-9
        r = fn(v[hi, 0], v[hi, 1], s)
        v[hi, 2] = z0 + (v[hi, 2] - z0) / (ztop - z0) * (r - z0)
        return v
    return w


def on_surface(m: Manifold, fn, s, embed=0.0, flip=False) -> Manifold:
    """Drape a flat part built at z in [0, h] onto a roof surface: z -> fn(x, y) - embed + z (downward if flip:
    z -> fn(x, y) + embed - z). The downward case mirrors first, so the mesh keeps its outward winding."""
    if flip:
        m = m.mirror((0.0, 0.0, 1.0))  # z -> -z, orientation kept

    def w(v):
        v = np.array(v, dtype=np.float64)
        base = fn(v[:, 0], v[:, 1], s)
        v[:, 2] = base + (embed if flip else -embed) + v[:, 2]
        return v
    return m.refine_to_length(0.01).warp_batch(w)


def level_on(sec: CrossSection, fn, s, top: float, embed: float) -> Manifold:
    """A level-topped part standing on a roof: a plan section from fn(x, y) - embed up to the plane z = top."""
    m = sec.extrude(0.01).refine_to_length(0.006)

    def w(v):
        v = np.array(v, dtype=np.float64)
        base = fn(v[:, 0], v[:, 1], s) - embed
        v[:, 2] = base + (top - base) * v[:, 2] / 0.01
        return v
    return m.warp_batch(w)


# ------------------------------------------------------------------------------------------------ covers

def _plan(s, inset=0.0) -> CrossSection:
    c = s["cover"]
    xe, hy, ch = c["x_end"] - inset, c["half_y"] - inset, c["vert_chamfer"]
    k = ch - inset * (math.sqrt(2) - 1)  # keeps the chamfer faces parallel when insetting
    return poly([(-xe + k, -hy), (xe - k, -hy), (xe, -hy + k), (xe, hy - k), (xe - k, hy), (-xe + k, hy),
                 (-xe, hy - k), (-xe, -hy + k)])


def _corner_trim(m: Manifold, s, inset=0.0) -> Manifold:
    """The four top-corner facets: a|x| + b|y| + c z <= d (DWG front/side/top)."""
    a, b, cz, d = s["cover"]["corner_plane"]
    n = math.sqrt(a * a + b * b + cz * cz)
    for sx in (-1, 1):
        for sy in (-1, 1):
            m = m.trim_by_plane((-sx * a, -sy * b, -cz), -(d - inset * n) / n)
    return m


def covers(s) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Both cover shells, closed, in the dock frame: {"cover_r": {mat: (v, f)}, "cover_l": ...}."""
    return {side: export(m, "shell") for side, m in cover_solids(s).items()}


def cover_solids(s) -> dict[str, Manifold]:
    """Both cover shells as solids, closed, in the dock frame (material-tagged; export() splits them)."""
    c, bd = s["cover"], s["body"]
    zb, t, hy = c["bottom_z"], c["wall_t"], c["half_y"]
    Z0, ZT = 0.45, 0.80
    outer = ext_z(_plan(s), zb, ZT).refine_to_length(0.008).warp_batch(_roof_warp(roof_outer, Z0, ZT, s))
    outer = R.tag(_corner_trim(outer, s), "shell")
    inner = ext_z(_plan(s, t), zb - 0.2, ZT).refine_to_length(0.008).warp_batch(_roof_warp(roof_inner, Z0, ZT, s))
    inner = R.tag(_corner_trim(inner, s, inset=t), "liner")
    shell = outer - inner

    # openings over the cabinet faces (DWG front; the rear one is assumed the same) and the notch over the RTK fin
    co = bd["cutout"]
    cc = tuple(co["chamfer"])
    sec = _chamfer_rect(co["x"][0], zb - 0.2, co["x"][1], co["top_z"], tl=cc, tr=cc, r=co["r"])
    cut = ext_y(sec, -hy - 0.05, -hy + 0.04) + ext_y(sec, hy - 0.04, hy + 0.05)
    (ax, az), (bx, bz), (cx, cz) = c["notch"]
    cut = cut + ext_y(poly([(ax, az), (bx - 0.006, bz - 0.01), (cx + 0.006, cz - 0.01)]), -hy - 0.05, -hy + 0.04)
    shell = shell - R.tag(cut, "liner")

    # grips: recesses in the four vertical corner facets (DWG front/side)
    g, ch = c["grip"], c["vert_chamfer"]
    for sx in (-1, 1):
        for sy in (-1, 1):
            cxy = np.array([sx * (c["x_end"] - ch / 2), sy * (hy - ch / 2)])
            nrm = np.array([sx, sy]) / math.sqrt(2)
            slot = box((-g["w"] / 2, -0.006, g["z"][0]), (g["w"] / 2, 0.006, g["z"][1]))
            ang = math.degrees(math.atan2(nrm[1], nrm[0])) - 90.0
            slot = slot.rotate((0, 0, ang)).translate((cxy[0] + nrm[0] * 0.001, cxy[1] + nrm[1] * 0.001, 0))
            shell = shell - R.tag(slot, "grip")
    # access hatches on the end walls: a fine groove round a rounded rectangle, a finger slot at the top edge
    # (DJI product images: one mid-height, one over each hinge; the hinge brackets sit behind the lower two)
    ht = c["hatches"]

    def hatch(sx, y0, y1, z0, z1):
        xe = sx * c["x_end"]
        xa, xb = (xe - 0.0012, xe + 0.01) if sx > 0 else (xe - 0.01, xe + 0.0012)
        ring = ext_x(rrect(y0, z0, y1, z1, 0.004), xa, xb) - ext_x(rrect(y0 + 0.0009, z0 + 0.0009, y1 - 0.0009, z1 - 0.0009, 0.0031),
                                                                   xe - 0.02, xe + 0.02)
        ym = (y0 + y1) / 2
        xa2, xb2 = (xe - 0.003, xe + 0.01) if sx > 0 else (xe - 0.01, xe + 0.003)
        return ring + ext_x(rrect(ym - 0.012, z1 - 0.0045, ym + 0.012, z1 - 0.0015, 0.0012), xa2, xb2)

    for sx, key in ((1, "end_pos_x"), (-1, "end_neg_x")):
        shell = shell - R.tag(hatch(sx, *ht[key]["y"], *ht[key]["z"]), "grip")
        for sy in (-1, 1):
            y0, y1 = sorted((sy * ht["hinge"]["y"][0], sy * ht["hinge"]["y"][1]))
            shell = shell - R.tag(hatch(sx, y0, y1, *ht["hinge"]["z"]), "grip")

    # parting split: vertical at x_parting, with the diagonal step and notch edge on the front face (DWG front)
    gap = 0.00025
    xp = c["x_parting"]
    (p0x, p0z), (p1x, p1z) = c["parting_front"]
    right_main = box((xp, -hy + 0.035, -1), (2, hy + 1, 2))
    right_front = ext_y(poly([(p0x, 2.0), (p0x, p0z), (p1x, p1z), (cx, cz), (cx, -1.0), (2.0, -1.0), (2.0, 2.0)]),
                        -hy - 1, -hy + 0.036)
    region = R.tag(right_main + right_front, "seal")  # the parting edges: a dark rubber seal on the real covers
    right = shell ^ region.translate((gap, 0, 0))
    left = shell - region.translate((-gap, 0, 0))
    return {"cover_r": right, "cover_l": left}


def cover_fittings(s) -> dict[str, list[tuple[str, str, Manifold]]]:
    """Parts riding on the covers, in the dock frame (closed): name, material, solid."""
    c, g, rg = s["cover"], s["gauge"], s["rain_gauge"]
    out: dict[str, list] = {"cover_r": [], "cover_l": []}

    def add(side, name, mat, m):
        out[side].append((name, mat, m))

    # ---- wind speed gauge (DWG front/side): level base fairing, mast, collar, hub, three cups; the dock camera and
    #      its light on the mast's -X face (dji_install p8 parts 1-3)
    mx, my = g["mast_xy"]
    mr = g["mast_r"]
    gb = g["base"]
    base_sec = (CrossSection.circle(gb["r"], 64).translate((gb["x"][0], my)) +
                CrossSection.circle(gb["r"], 64).translate((gb["x"][1], my))).hull()
    bev = 0.006
    xc = (gb["x"][0] + gb["x"][1]) / 2
    w_, h_ = gb["x"][1] - gb["x"][0] + 2 * gb["r"], 2 * gb["r"]
    cap = base_sec.translate((-xc, -my)).extrude(bev, 0, 0.0, ((w_ - 2 * bev) / w_, (h_ - 2 * bev) / h_)) \
        .translate((xc, my, gb["top_z"] - bev))
    add("cover_r", "gauge_base", "shell", level_on(base_sec, roof_outer, s, gb["top_z"] - bev, 0.006) + cap)
    (cz0, cz1), (hz0, hz1) = g["collar"]["z"], g["hub"]["z"]
    add("cover_r", "gauge_mast", "shell", cyl((mx, my, gb["top_z"] - 0.004), (mx, my, cz0), mr, seg=48)
        + cyl((mx, my, cz0), (mx, my, cz1), g["collar"]["r"], seg=48))
    hr, ct = g["hub"]["r"], g["hub"]["cap_top"]
    cap = Manifold.sphere(1.0, 48).scale((hr, hr, ct - hz1)).translate((mx, my, hz1)).trim_by_plane((0, 0, 1), hz1)
    add("cover_r", "gauge_hub", "shell", cyl((mx, my, hz0 - 0.001), (mx, my, hz1), hr, seg=48) + cap)
    cu = g["cups"]
    parts = []
    hub = np.array([mx, my, cu["z"]])
    for k in range(3):
        a_ = math.radians(cu["phase_deg"] + 120 * k)
        d = np.array([math.cos(a_), math.sin(a_), 0.0])
        tip = hub + d * cu["arm_r"]
        parts.append(cyl(hub, tip, 0.0025, seg=12))
        tang = np.array([-d[1], d[0], 0.0])  # cups open tangentially
        cup = sph(tip, cu["cup_r"], 32) - sph(tip + tang * 0.003, cu["cup_r"] - 0.0015, 32)
        parts.append(cup.trim_by_plane(tuple(-tang), float(-np.dot(tang, tip))))
    add("cover_r", "gauge_cups", "shell", union(parts))
    xf = mx - mr
    # flush with the mast's front (DWG front shows nothing standing proud of it)
    add("cover_r", "extcam_lens", "lens", cyl((xf + 0.006, my, g["ext_cam_z"]), (xf - 0.0005, my, g["ext_cam_z"]), g["ext_cam_r"], seg=32))
    add("cover_r", "extcam_light", "light", cyl((xf + 0.006, my, g["ext_light_z"]), (xf - 0.0005, my, g["ext_light_z"]),
                                                g["ext_light_r"], seg=24))

    # ---- rainfall gauge (dji_install p8 part 4, dji_maint p14 part 2): a level disc proud of the sloping roof,
    #      white ring with six screws round a metal sensing plate
    rx, ry = rg["xy"]
    top = rg["top_z"]
    ring = level_on(CrossSection.circle(rg["ring_r"], 72).translate((rx, ry)), roof_outer, s, top - 0.004, 0.006)
    ring = ring + Manifold.cylinder(0.004, rg["ring_r"], rg["ring_r"] - 0.004, 72).translate((rx, ry, top - 0.004))
    ring = ring - Manifold.cylinder(0.01, rg["disc_r"] + 0.002, rg["disc_r"] + 0.002, 64).translate((rx, ry, top - 0.003))
    add("cover_r", "rain_ring", "shell", ring)
    add("cover_r", "rain_disc", "metal", Manifold.cylinder(0.0035, rg["disc_r"], rg["disc_r"] - 0.002, 64).translate((rx, ry, top - 0.005)))
    rs = rg["screw_r"]
    add("cover_r", "rain_screws", "metal", union([Manifold.cylinder(0.0015, 0.0032, 0.0028, 12).translate(
        (rx + rs * math.cos(2 * math.pi * k / rg["screws"] + math.pi / 2),
         ry + rs * math.sin(2 * math.pi * k / rg["screws"] + math.pi / 2), top - 0.0005)) for k in range(rg["screws"])]))

    # ---- markings + internal camera on the inner face of the gauge cover's roof (a wall once open; dji_maint
    #      p14). Wall coordinates: u = y, v = distance from the parting edge along the roof.
    mk = s["cover_markers"]
    xp = c["x_parting"]

    def wall(m: Manifold, embed=0.0003) -> Manifold:
        return on_surface(m, roof_inner, s, embed=embed, flip=True)

    def flat(pts_uv, h=0.0012):
        return poly([(xp + v, u) for u, v in pts_uv]).extrude(h)

    hv, hh, hw, bar = mk["H_v"], mk["H_h"], mk["H_w"], mk["H_bar"]
    H = union([flat([(u0, hv - hh / 2), (u0 + bar, hv - hh / 2), (u0 + bar, hv + hh / 2), (u0, hv + hh / 2)])
               for u0 in (-hw / 2, hw / 2 - bar)] +
              [flat([(-hw / 2, hv - bar / 2), (hw / 2, hv - bar / 2), (hw / 2, hv + bar / 2), (-hw / 2, hv + bar / 2)])])
    add("cover_r", "marker_H", "marker", wall(H))
    (tu0, tu1), (tv0, tv1) = mk["tri_small_u"], mk["tri_small_v"]
    add("cover_r", "marker_tri_small", "marker", wall(flat([(tu0, tv1), (tu1, tv1), ((tu0 + tu1) / 2, tv0)])))
    (bu0, bu1), (bv0, bv1) = mk["tri_big_u"], mk["tri_big_v"]
    big = [flat([(sy * bu0, bv0), (sy * bu0, bv1), (sy * bu1, (bv0 + bv1) / 2)]) for sy in (-1, 1)]
    add("cover_r", "marker_tri_big", "marker", wall(union(big)))
    # the module hangs from the roof's inner face down to the lens, whose glass sits 0.5 mm behind the camera's
    # optical centre (spec pos_closed), so a renderer's camera is never inside the mesh
    cam = s["cameras"]["dock_internal"]["pos_closed"]
    z_in = float(roof_inner(np.array([cam[0]]), np.array([0.0]), s)[0])
    z_glass = cam[2] + 0.0005
    depth = z_in - (z_glass + 0.004)  # housing face 4 mm above the glass; the lens barrel stands proud of it
    if depth < 0.003:
        raise ValueError("dock_internal camera is too close to the roof's inner face for its housing")
    housing = rrect(cam[0] - 0.022, -0.036, cam[0] + 0.020, 0.036, 0.008).extrude(depth + 0.004)
    add("cover_r", "intcam_housing", "camera", wall(housing, embed=0.004))
    add("cover_r", "intcam_lens", "lens", cyl((cam[0], 0.0, z_glass + 0.0045), (cam[0], 0.0, z_glass), 0.011, seg=32))
    add("cover_r", "intcam_light", "light",
        cyl((cam[0] + 0.024, 0.0, z_in - depth + 0.001), (cam[0] + 0.024, 0.0, z_in - depth - 0.002), 0.0055, seg=20))

    # left cover: white target plate with three dark squares on a 45-degree gusset in the rear corner by the
    # parting edge (dji_install p8 / dji_um p25 top view, DJI product images). Size and place EST.
    pl = mk["plate"]
    pc = np.array(pl["center_closed"])
    nrm = np.array([0.7071, -0.7071, 0.0])  # faces the parting plane and the front; once open: up and front
    u_ax = np.array([0.0, 0.0, 1.0])
    w_ax = np.cross(u_ax, nrm)
    M = np.zeros((3, 4))
    M[:, 0], M[:, 1], M[:, 2], M[:, 3] = w_ax, u_ax, nrm, pc
    pw, ph = pl["size"]
    add("cover_l", "marker_plate", "marker", box((-pw / 2, -ph / 2, -0.004), (pw / 2, ph / 2, 0.0)).transform(M))
    squares = [box((-pw / 2 + 0.008, v0, 0.0), (-pw / 2 + 0.028, v0 + 0.020, 0.0008)) for v0 in (-0.050, -0.010, 0.030)]
    add("cover_l", "marker_plate_squares", "camera", union(squares).transform(M))
    a_pt, b_pt = pc + w_ax * pw / 2, pc - w_ax * pw / 2
    add("cover_l", "marker_gusset", "liner",
        ext_z(poly([a_pt[:2], b_pt[:2], (b_pt[0], a_pt[1])]), pc[2] - ph / 2, pc[2] + ph / 2).translate(tuple(-nrm * 0.004)))

    # actuator clevis just inside each cover's end wall (clear of the tray edge at x 0.2904)
    for side, sx in (("cover_r", 1), ("cover_l", -1)):
        _, (_, _, pz) = actuator_pins(s, side[-1])
        add(side, "clevis", "frame", box((sx * 0.2925, -0.026, pz - 0.028), (sx * (c["x_end"] - c["wall_t"] + 0.002), 0.026, pz + 0.028)))
    return out


# ------------------------------------------------------------------------------------------------ dock body

def _chamfer_rect(x0, z0, x1, z1, tl=(0.0, 0.0), tr=(0.0, 0.0), br=(0.0, 0.0), bl=(0.0, 0.0), r=0.0) -> CrossSection:
    """Rectangle in a plane with optional corner chamfers (dx, dz) per corner, corners rounded by r."""
    pts = []
    pts += [(x0, z0 + bl[1]), (x0 + bl[0], z0)] if bl[0] else [(x0, z0)]
    pts += [(x1 - br[0], z0), (x1, z0 + br[1])] if br[0] else [(x1, z0)]
    pts += [(x1, z1 - tr[1]), (x1 - tr[0], z1)] if tr[0] else [(x1, z1)]
    pts += [(x0 + tl[0], z1), (x0, z1 - tl[1])] if tl[0] else [(x0, z1)]
    sec = poly(pts)
    if r > 0:
        sec = sec.offset(-r, JoinType.Round, 2.0, 16).offset(r, JoinType.Round, 2.0, 16)
    return sec


def stadium(cx, cz, w, h) -> CrossSection:
    r = min(w, h) / 2
    return rrect(cx - w / 2, cz - h / 2, cx + w / 2, cz + h / 2, r * 0.999, seg=16)


def base_and_body(s) -> list[tuple[str, str, object]]:
    b, bd, p, fp, c = s["base"], s["body"], s["pad"], s["front_panel"], s["cover"]
    out: list[tuple[str, str, object]] = []

    def add(n, mat, m):
        out.append((n, mat, m))

    # ---- mounting base brackets (dji_install p8 part 14): four feet, slots and bolt holes (DWG front + side)
    fz0, fz1 = b["frame_z"]
    x0, x1 = b["bracket_x"]
    feet, recesses = [], []
    for sx in (-1, 1):
        xa, xb = sorted((sx * x0, sx * x1))
        xo = sx * x1  # outer end face
        for ya, yb in (b["bracket_y_front"], b["bracket_y_rear"]):
            sy = 1 if yb > 0 else -1
            feet.append(ext_z(rrect(xa, ya, xb, yb, 0.004), 0.0, fz0))
            for yc in (0.161, 0.197, 0.233):  # three vertical stiffener slots on the end face (DWG side)
                recesses.append(box((xo - sx * 0.004, sy * yc - 0.006, 0.014), (xo + sx * 0.002, sy * yc + 0.006, 0.058)))
            recesses.append(ext_x(stadium(sy * 0.260, 0.022, 0.020, 0.012), min(xo - sx * 0.005, xo + sx * 0.002),
                                  max(xo - sx * 0.005, xo + sx * 0.002)))
            yo = yb if sy > 0 else ya  # outer face toward the front/back: bolt slot near the inner end (DWG front)
            recesses.append(ext_y(stadium(sx * 0.152, 0.021, 0.018, 0.010), min(yo - sy * 0.005, yo + sy * 0.002),
                                  max(yo - sy * 0.005, yo + sy * 0.002)))
    for mat, vf in export(R.tag(union(feet), "bracket") - R.tag(union(recesses), "cabinet_in"), "bracket").items():
        add(f"brackets_{mat}", mat, vf)
    (fx0, fx1), (fy0, fy1) = b["frame_x"], b["frame_y"]
    add("frame", "frame", ext_z(rrect(fx0, fy0, fx1, fy1, 0.012), fz0, fz1))
    bolts = []
    for sx, xf in ((-1, fx0), (1, fx1)):  # five bolts along each end of the band (DWG side)
        for y in (0.1597, 0.1007, 0.0406, -0.0791, -0.1587):
            bolts.append(cyl((xf - sx * 0.001, y, 0.080), (xf + sx * 0.0025, y, 0.080), 0.0045, seg=12))
    add("frame_bolts", "metal", union(bolts))
    bp = b["bottom_plate"]
    hx, hy = bp["half"]
    sp = bp["spout"]
    plate = ext_z(rrect(-hx, -hy, hx, hy, 0.010), bp["z"][0], fz0 + 0.001)
    spout = box((sp["x"][0], sp["y"][0], sp["z"][0] + 0.010), (sp["x"][1], sp["y"][1], sp["z"][1] + 0.001)) + \
        cyl((0, sp["y"][0], sp["z"][0] + 0.010), (0, sp["y"][1], sp["z"][0] + 0.010), (sp["x"][1] - sp["x"][0]) / 2, seg=24)
    add("bottom_plate", "bracket", plate + spout)  # drain pipe hole underneath (dji_install p8 part 21)

    # ---- lower cabinet: port panel, e-stop, E-Port plate, port bay with connectors, electrical cabinet door,
    #      actuator bays in both ends
    ex, yf, yb, top = bd["end_x"], bd["front_y"], bd["back_y"], bd["top_z"]
    (dx0, dz0), (dx1, dz1) = fp["door"]
    door = _chamfer_rect(dx0, dz0, dx1, dz1, tr=tuple(fp["door_chamfer_top"]), br=tuple(fp["door_chamfer_bottom"]), r=0.010)
    win = door.offset(-fp["door_window_inset"], JoinType.Round, 2.0, 12)
    cab = R.tag(box((-ex, yf, fz1), (ex, yb, top)), "panel")
    ixp, iyp = bd["tray_inner_half"]
    cab = cab - R.tag(ext_z(rrect(-ixp, -iyp, ixp, iyp, 0.022), p["floor_z"], 1.0), "pad")  # the pad pocket
    cab = cab - R.tag(ext_y(win, yf - 0.01, yf + 0.045), "cabinet_in")  # the electrical bay behind the door's window
    (bx0, bz0), (bx1, bz1) = fp["port_bay"]
    cab = cab - R.tag(ext_y(rrect(bx0 + 0.003, bz0 - 0.01, bx1 - 0.003, bz1 - 0.013, 0.014), yf - 0.01, yf + 0.018), "panel_dark")
    eb = bd["end_bay"]
    for sx in (-1, 1):
        cab = cab - R.tag(box((sx * (ex - eb["depth"]), eb["y"][0], eb["z"][0]), (sx * (ex + 0.01), eb["y"][1], eb["z"][1])),
                          "cabinet_in")
    for mat, vf in export(cab, "panel").items():
        add(f"cabinet_{mat}", mat, vf)
    (qx0, qz0), (qx1, qz1) = fp["port_panel"]
    ch = fp["port_panel_chamfer"]
    panel = ext_y(_chamfer_rect(qx0, qz0, qx1, qz1, tl=(ch, ch), r=0.008), yf - 0.0015, yf + 0.001)
    panel = panel - ext_y(rrect(bx0, bz0 - 0.01, bx1, bz1, 0.015), yf - 0.01, yf + 0.01)
    add("port_panel", "panel", panel)
    add("end_drums", "frame", union([cyl((sx * (ex - 0.004), eb["drum_y"], eb["drum_z"]), (sx * (ex + 0.020), eb["drum_y"], eb["drum_z"]),
                                         eb["drum_r"], seg=40) for sx in (-1, 1)]))
    # e-stop (dji_install p8 part 15): a yellow housing with rounded corners, a raised yellow collar round the red
    # mushroom button (DJI product images)
    ex_, ez = fp["estop"]
    bw, bh_ = fp["estop_base"]
    kx, kz = fp["estop_button"]
    er = fp["estop_r"]
    add("estop_base", "yellow", ext_y(rrect(ex_ - bw / 2, ez - bh_ / 2, ex_ + bw / 2, ez + bh_ / 2, 0.012), yf - 0.010, yf - 0.001)
        + cyl((kx, yf - 0.0095, kz), (kx, yf - 0.016, kz), fp["estop_collar_r"], fp["estop_collar_r"] - 0.003, seg=48))
    add("estop_button", "red", cyl((kx, yf - 0.0155, kz), (kx, yf - 0.023, kz), er - 0.004, seg=48)
        + cyl((kx, yf - 0.023, kz), (kx, yf - 0.029, kz), er, er - 0.005, seg=48))
    # E-Port plate: four screws, two square sockets and an oval (dji_install p8 part 20)
    (px0, pz0), (px1, pz1) = fp["eport"]
    add("eport_plate", "metal", ext_y(rrect(px0, pz0, px1, pz1, 0.008), yf - 0.004, yf))
    pzc = (pz0 + pz1) / 2
    bits = [ext_y(rrect(x - 0.007, pzc - 0.009, x + 0.007, pzc + 0.009, 0.002), yf - 0.007, yf - 0.003) for x in fp["eport_sockets_x"]]
    bits.append(ext_y(stadium(fp["eport_oval_x"], pzc, 0.012, 0.024), yf - 0.007, yf - 0.003))
    bits += [cyl((x, yf - 0.003, z), (x, yf - 0.0055, z), 0.004, seg=12) for x in (px0 + 0.008, px1 - 0.008)
             for z in (pz0 + 0.0075, pz1 - 0.0065)]
    add("eport_bits", "camera", union(bits))
    # cable glands at the top of the port bay pointing down (AC in, LAN in, PoE out), each with a grey latch, and a
    # cable from each curving down and toward +X over the panel's bottom lip into the base (DJI product images)
    gl, cb = fp["glands"], fp["cables"]
    yc, (gz0, gz1), hw = gl["y"], gl["z"], gl["w"] / 2
    glands, latches, cables = [], [], []
    for x, (xe, ze) in zip(fp["ports_x"], cb["end"]):
        glands.append(ext_y(rrect(x - hw, gz0 + 0.014, x + hw, gz1 + 0.002, 0.004), yc - hw, yc + hw))
        glands.append(cyl((x, yc, gz0 + 0.015), (x, yc, gz0), hw * 0.78, cb["r"] + 0.002, seg=28))  # strain relief
        lw = gl["latch_w"] / 2
        latches.append(ext_y(rrect(x - lw, gl["latch_z"][0], x + lw, gl["latch_z"][1], 0.002), yc - hw - 0.0025, yc - hw + 0.001))
        # cubic Bezier: straight down out of the gland, then bending toward +X to the bottom lip, then back into
        # the base frame under the cabinet
        p0, p1 = np.array([x, yc, gz0 + 0.002]), np.array([x, yc, gz0 - 0.040])
        p2, p3 = np.array([xe - 0.012, yc - 0.002, ze + 0.020]), np.array([xe, yc - 0.003, ze])
        path = [(1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
                for t in np.linspace(0.0, 1.0, 18)]
        d_end = np.array([0.35, 0.65, -0.68])
        path += [p3 + d_end / np.linalg.norm(d_end) * k for k in (0.012, 0.024, 0.036)]
        cables.append(tube(path, cb["r"], seg=14))
    add("connectors", "black", union(glands))
    add("connector_latches", "grip", union(latches))
    add("cables", "black", union(cables))
    # printed port names above the glands, thin separators between them, and the rating plate (decals, 0.3 mm)
    lb = fp["labels"]
    (lz0, lz1), lwid = lb["z"], lb["w"]
    ya, yb_ = yf - 0.0021, yf - 0.0015  # on the port panel's face
    decals = [ext_y(rrect(x - lwid / 2, lz0 + 0.0075, x + lwid / 2, lz1, 0.0005), ya, yb_) for x in fp["ports_x"]]
    decals += [ext_y(rrect(x - lwid * 0.3, lz0, x + lwid * 0.3, lz0 + 0.0045, 0.0005), ya, yb_) for x in fp["ports_x"]]
    decals += [box((x - 0.0004, ya, lz0 - 0.002), (x + 0.0004, yb_, lz1 + 0.004)) for x in lb["separators_x"]]
    (rx0, rz0), (rx1, rz1) = lb["rating_plate"]
    ya, yb_ = yf + 0.0174, yf + 0.018  # on the port bay's back face
    plate_ = ext_y(rrect(rx0, rz0, rx1, rz1, 0.002), ya, yb_) - \
        ext_y(rrect(rx0 + 0.0012, rz0 + 0.0012, rx1 - 0.0012, rz1 - 0.0012, 0.0015), yf + 0.01, yf + 0.03)
    rows_ = [ext_y(rrect(rx0 + 0.005, z, rx1 - 0.005 - 0.018 * (k % 2), z + 0.0022, 0.0004), ya, yb_)
             for k, z in enumerate(np.linspace(rz0 + 0.006, rz1 - 0.009, 5))]
    add("labels", "label", union(decals + [plate_] + rows_))
    # electrical cabinet door (dji_install p8 part 12; DJI product images): a smoked window in a narrow dark frame with
    # chamfered right corners and two bolts; behind the pane a row of DIN-rail breakers and a grey box
    add("door", "cabinet", ext_y(door, yf - 0.006, yf + 0.002) - ext_y(win, yf - 0.02, yf + 0.02))
    add("door_glass", "smoked_glass", ext_y(win.offset(0.002, JoinType.Round, 2.0, 12), yf - 0.0045, yf - 0.0015))
    br = fp["breakers"]
    (bx0, bx1), (bz0, bz1), nb = br["x"], br["z"], br["n"]
    wb = (bx1 - bx0) / nb
    mods = [ext_y(rrect(bx0 + k * wb + 0.0005, bz0, bx0 + (k + 1) * wb - 0.0005, bz1, 0.0015), yf + 0.012, yf + 0.046)
            for k in range(nb)]
    add("door_breakers", "breaker", union(mods))
    add("door_breaker_toggles", "black", union([box((bx0 + (k + 0.3) * wb, yf + 0.006, (bz0 + bz1) / 2 - 0.004),
                                                    (bx0 + (k + 0.7) * wb, yf + 0.012, (bz0 + bz1) / 2 + 0.006)) for k in range(nb)]))
    add("door_rail", "metal", box((bx0 - 0.012, yf + 0.040, (bz0 + bz1) / 2 - 0.0175), (bx1 + 0.012, yf + 0.046, (bz0 + bz1) / 2 + 0.0175)))
    (gx0, gz0), (gx1, gz1) = fp["door_box"]
    add("door_box", "panel_dark", ext_y(rrect(gx0, gz0, gx1, gz1, 0.004), yf + 0.015, yf + 0.046)
        - ext_y(rrect(gx0 + 0.012, gz0 + 0.020, gx1 - 0.012, gz0 + 0.034, 0.003), yf + 0.010, yf + 0.018))
    add("door_bolts", "metal", union([cyl((x, yf - 0.006, z), (x, yf - 0.009, z), 0.0045, seg=16) for x, z in fp["door_bolts"]]))

    # ---- tray (the charcoal landing-pad housing) on the cabinet: a vertical wall, a chamfer up and in to the rim,
    #      black rubber lips on the front and back rim, pocket down to the pad floor (DWG top view; DJI product
    #      images)
    tx, ty = bd["tray_half"]
    ix, iy = bd["tray_inner_half"]
    fz, rz, wz, ci = p["floor_z"], p["rim_z"], bd["tray_wall_z"], bd["tray_chamfer_in"]

    def ring(hx_, hy_, r_, z_):
        q = np.asarray(rrect(-hx_, -hy_, hx_, hy_, r_).to_polygons()[0], float)
        return np.column_stack([q, np.full(len(q), z_)])

    tray = ext_z(rrect(-tx, -ty, tx, ty, 0.010), top - 0.017, wz) + \
        Manifold.hull_points(np.vstack([ring(tx, ty, 0.010, wz - 0.001), ring(tx - ci, ty - ci, 0.006, rz)]))
    tray = tray + ext_z(rrect(-ix - 0.008, -iy - 0.008, ix + 0.008, iy + 0.008, 0.026), fz - 0.012, top - 0.016)  # tub
    tray = tray - R.tag(ext_z(rrect(-ix, -iy, ix, iy, 0.022), fz, 1.0), "pad")
    # the +-X ends stop at the vertical wall: no chamfer or rim there, so the pad blocks' end faces stand clear and
    # the channel between them is open along X (DJI product video from the dock camera across the pad; the parked
    # aircraft's camera sees over the end into the open cover)
    tray = tray - box((ix, -1.0, wz), (1.0, 1.0, 1.0)) - box((-1.0, -1.0, wz), (-ix, 1.0, 1.0))
    tray = tray - ext_x(poly([(-ty - 0.01, top + 0.010), (-ty + 0.012, top - 0.020), (-ty - 0.01, top - 0.020)]), -1, 1)
    for mat, vf in export(tray, "housing").items():
        add(f"tray_{mat}", mat, vf)
    lx, lz1, yl = bd["tray_lip_half_x"], bd["tray_lip_top_z"], ty - ci
    lips = []
    for sy in (-1, 1):
        lips.append(ext_y(rrect(-lx, rz, lx, lz1, 0.004), sy * yl - 0.004, sy * yl + 0.004))
        lips += [cyl((sx * lx, sy * yl - 0.0055, (rz + lz1) / 2), (sx * lx, sy * yl + 0.0055, (rz + lz1) / 2), 0.0062, seg=20)
                 for sx in (-1, 1)]
    add("tray_lips", "black", union(lips))

    # ---- RTK module (dji_install p8 part 18) on its bent bracket; the fin that fills the covers' front notch
    rt = s["rtk"]
    fa = rt["fin_apex_z"]
    add("rtk_fin", "housing", ext_y(poly([(0.0, fa), (-0.028, top + 0.003), (0.028, top + 0.003)]), yf - 0.013, yf - 0.0015))
    cx, cy = rt["puck_center"]
    z0, z1 = rt["puck_z"]
    r = rt["puck_r"]
    prof = np.array([(0.0, z0), (r, z0), (r, z1 - 0.010), (r - 0.003, z1 - 0.005), (r - 0.012, z1 - 0.0012),
                     (r - 0.030, z1), (0.0, z1)])
    add("rtk_puck", "shell", CrossSection([prof]).revolve(96).translate((cx, cy, 0.0)))
    bo = rt["bolts"]
    fl0, fl1 = bo["z"]
    add("rtk_flange", "camera", Manifold.cylinder(fl1 - fl0, bo["pitch_r"] + 0.0015, bo["pitch_r"] + 0.0015, 96)
        .translate((cx, cy, fl0)))
    add("rtk_bolts", "metal", union([cyl((cx + bo["pitch_r"] * math.cos(a), cy + bo["pitch_r"] * math.sin(a), fl1 - 0.001),
                                         (cx + bo["pitch_r"] * math.cos(a), cy + bo["pitch_r"] * math.sin(a), fl1 + 0.003),
                                         bo["r"], seg=6) for a in np.radians(np.arange(0, 360, 360 / bo["n"]))]))
    pz0, pz1 = rt["plate_z"]
    hx = rt["bracket_half_x"]
    plate = Manifold.cylinder(pz1 - pz0, r, r, 96).translate((cx, cy, pz0))
    add("rtk_bracket", "panel_dark", plate + ext_x(poly(rt["bracket_yz"]), -hx, hx))
    mt = rt["mount"]
    add("rtk_mount", "panel_dark", box((mt["x"][0], mt["y"][0], pz0 - 0.010), (mt["x"][1], mt["y"][1], pz0 + 0.001))
        + box((-0.044, yf - 0.004, mt["z"][0]), (-0.020, yf + 0.001, pz0)))

    # ---- rear: the cabinet back panel standing out into the covers' rear cut-out (plain: no source shows its face)
    rp = s["rear_panel"]
    rsec = _chamfer_rect(rp["x"][0], rp["z"][0], rp["x"][1], rp["z"][1], r=rp["r"])
    add("rear_panel", "panel", ext_y(rsec, yb - 0.004, rp["y_out"]))

    # ---- hinge knuckles on brackets from the cabinet's end faces (inside the covers' end-wall notches)
    kn = []
    piv_x, piv_z = c["pivot_x"], c["pivot_z"]
    for sx in (-1, 1):
        for sy in (-1, 1):
            ya, yb_ = (0.197, 0.243) if sy > 0 else (-0.243, -0.197)
            kn.append(cyl((sx * piv_x, ya, piv_z), (sx * piv_x, yb_, piv_z), 0.010, seg=24))
            kn.append(box((sx * (ex - 0.002), ya - 0.006, fz1), (sx * piv_x, yb_ + 0.006, piv_z)))
    add("hinges", "bracket", union(kn))
    # actuator body pins: two cheeks either side of each cylinder's cross pin, inside the end bays
    acts = s["actuator"]
    bx, _, bz = acts["body_pivot"]
    cheeks = []
    for sx in (-1, 1):
        for y0, y1 in ((0.024, 0.030), (-0.030, -0.024)):
            cheeks.append(box((sx * (ex - eb["depth"] + 0.001), y0, bz - 0.022), (sx * (bx + 0.016), y1, bz + 0.022)))
        cheeks.append(cyl((sx * bx, -0.030, bz), (sx * bx, 0.030, bz), 0.006, seg=16))
    add("act_mounts", "frame", union(cheeks))
    return out


def block_top(x, ay, s) -> np.ndarray:
    """Top of a pad block at (x, |y|), vectorised (DWG top view open + the crest heights measured on DJI imagery).

    A scoop: from the inner edge line a concave face rises to the arch line, a land (the crest) runs from the arch
    to the lens line in the middle and to the outline diagonal toward the ends, then the outer face falls to the
    edge height (corner facets, outer edge). The crest is level for |x| <= crest_flat_x and falls to the edge height
    by crest_end_x."""
    p = s["pad"]
    fz, tz, ez = p["floor_z"], p["block_top_z"], p["block_edge_z"]
    yi, xf, xe = p["block_inner_y"], p["crest_flat_x"], p["crest_end_x"]
    ax = np.abs(np.asarray(x, float))
    ay = np.asarray(ay, float)
    u = np.clip((ax - xf) / (xe - xf), 0.0, 1.0)
    crest = ez + (tz - ez) * (1.0 - u ** 2)
    A = np.asarray(p["arch"], float)
    ya = np.interp(ax, A[:, 0], A[:, 1])
    (c0x, c0y), (c1x, c1y) = p["block_corner"]
    diag = c1y + (ax - c1x) * (c0y - c1y) / (c0x - c1x)
    yl = np.minimum(p["land_outer_y"], diag)
    w = np.clip((ay - yi) / np.maximum(ya - yi, 1e-3), 0.0, None)
    face = fz + (crest - fz) * w ** p["face_pow"]
    k = 0.010  # soft minimum of face and land: rounds the crest edge by k / 2
    top = 0.5 * (face + crest - np.sqrt((face - crest) ** 2 + k * k))
    v = np.clip((ay - yl) / p["outer_fall"], 0.0, 1.0)
    top = top - (np.maximum(crest, ez) - ez) * v * v * (3.0 - 2.0 * v)
    return np.maximum(top, fz)


def pad_blocks(s) -> dict[str, Manifold]:
    """The two pad blocks either side of the channel: a heightfield (block_top) over the block plan, from just under
    the floor. The plan's outer corners follow the tray pocket's rounded corners."""
    p, bd = s["pad"], s["body"]
    fz = p["floor_z"]
    ix, iy = bd["tray_inner_half"]
    yi = p["block_inner_y"]
    zb, zt = fz - 0.004, 1.0
    out = {}
    for tag, sy in (("f", -1), ("r", 1)):
        y0, y1 = sorted((sy * yi, sy * (iy - 0.0005)))
        plan = rrect(-ix + 0.0005, y0, ix - 0.0005, y1, 0.004) ^ rrect(-ix + 0.0005, -iy + 0.0005, ix - 0.0005, iy - 0.0005, 0.022)
        m = ext_z(plan, zb, zt).refine_to_length(0.004)

        def warp(v):
            v = np.array(v, dtype=np.float64)
            top = block_top(v[:, 0], np.abs(v[:, 1]), s)
            v[:, 2] = zb + (v[:, 2] - zb) / (zt - zb) * (top - zb)
            return v
        out[tag] = m.warp_batch(warp)
    return out


def pad_collision(s) -> dict[str, Manifold]:
    """Convex pieces of the blocks for contact (MuJoCo collides a mesh through its hull): each block cut in three
    along X, so the hulls follow the crest falling toward the ends. The hulls fill the scoop to its chord, well under
    the aircraft's arms."""
    out = {}
    cuts = (-1.0, -s["pad"]["crest_flat_x"], s["pad"]["crest_flat_x"], 1.0)
    for tag, m in pad_blocks(s).items():
        for k, (a, b) in enumerate(zip(cuts, cuts[1:])):
            out[f"pad_block_{tag}{k}"] = m ^ box((a, -1.0, -1.0), (b, 1.0, 2.0))
    return out


def pad(s) -> list[tuple[str, str, object]]:
    """Landing pad inside the tray: blocks, corner light panels, orientation marker, vents (DWG top view)."""
    p = s["pad"]
    fz = p["floor_z"]
    out: list[tuple[str, str, object]] = []
    blocks = pad_blocks(s)
    for tag, m in blocks.items():
        out.append((f"pad_block_{tag}", "pad_block", m))
    # positioning-marker light panels on the corner facets (dji_maint 5.5 "light panels of the positioning markers")
    panels = []
    for sx in (-1, 1):
        for sy, tag in ((-1, "f"), (1, "r")):
            q = [(sx * a, sy * b) for a, b in p["light_panel"]]
            region = ext_z(poly(q).offset(-0.004, JoinType.Miter), fz - 0.01, 1.0)
            panels.append(skin(blocks[tag], region, 0.0015))
    out.append(("light_panels", "light_panel", union(panels)))
    out.append(("pad_stops", "black", union([box((sx * 0.262 - 0.008, sy * 0.050 - 0.008, fz), (sx * 0.262 + 0.008, sy * 0.050 + 0.008, fz + 0.014))
                                             for sx in (-1, 1) for sy in (-1, 1)])))
    # aircraft orientation marker: triangle toward the nose (+X) and an H lying across (DWG top view, part 8). A
    # decal (0.3 mm): the parked aircraft's belly pads sit right over it
    mk = [poly([(-0.043, -0.012), (-0.043, 0.012), (-0.022, 0.0)]).extrude(0.0003),
          box((-0.020, 0.020, 0), (0.030, 0.030, 0.0003)), box((-0.020, -0.030, 0), (0.030, -0.020, 0.0003)),
          box((0.000, -0.020, 0), (0.010, 0.020, 0.0003))]
    out.append(("marker_orient", "marker", union(mk).translate((0, 0, fz - 0.0001))))
    out.append(("pad_vents", "camera", box((-0.200, -0.028, fz), (-0.120, 0.028, fz + 0.0015)) +
                union([box((0.090 + 0.008 * k, -0.020, fz), (0.094 + 0.008 * k, 0.020, fz + 0.0016)) for k in range(8)])))
    return out


def actuator_meshes(s) -> list[tuple[str, str, str, Manifold]]:
    """Linear actuators in their own frames: cylinder from the body pin along +Z, rod ending on the cover pin."""
    a = s["actuator"]
    out = []
    for side in ("r", "l"):
        body = cyl((0, 0, -0.018), (0, 0, a["cyl_len"]), a["r_cyl"], seg=32) + cyl((0, -0.022, 0), (0, 0.022, 0), 0.012, seg=20)
        rod = cyl((0, 0, -a["rod_len"]), (0, 0, 0), a["r_rod"], seg=24) + cyl((0, -0.016, 0), (0, 0.016, 0), 0.010, seg=20)
        out.append((f"act_cyl_{side}", f"act_cyl_{side}", "actuator", body))
        out.append((f"act_rod_{side}", f"act_rod_{side}", "metal", rod))
    return out


# ------------------------------------------------------------------------------------------------ drone

def _closed_mesh(V: np.ndarray, F: np.ndarray) -> Manifold:
    """Manifold from a closed, consistently wound triangle mesh (flipped if it encloses negative volume)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        F = F[:, ::-1]
    m = Manifold(Mesh(vert_properties=np.ascontiguousarray(V, dtype=np.float32),
                      tri_verts=np.ascontiguousarray(F, dtype=np.uint32)))
    if m.is_empty():
        raise ValueError(f"loft is not a valid manifold: {m.status()}")
    return m


def loft(rings: list[np.ndarray]) -> Manifold:
    """A closed solid through rings of equal point count (same start and direction), capped at both ends."""
    n, m = len(rings[0]), len(rings)
    V = np.concatenate([np.asarray(r, float) for r in rings] + [np.mean(rings[0], 0)[None], np.mean(rings[-1], 0)[None]])
    F = []
    for k in range(m - 1):
        a, b = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            F += [(a + i, b + i, b + j), (a + i, b + j, a + j)]
    c0, c1, last = m * n, m * n + 1, (m - 1) * n
    for i in range(n):
        j = (i + 1) % n
        F += [(c0, i, j), (c1, last + j, last + i)]
    return _closed_mesh(V, np.array(F, dtype=np.int64))


def _superellipse(n_pts: int, a: float, b: float, e: float) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.stack([a * np.sign(c) * np.abs(c) ** (2 / e), b * np.sign(s) * np.abs(s) ** (2 / e)], 1)


def _profile(keys, xs) -> np.ndarray:
    """Key rows (x, v1, v2, ...) interpolated at xs, one smoothing pass (end rows kept)."""
    K = np.asarray(keys, float)
    out = np.stack([np.interp(xs, K[:, 0], K[:, c]) for c in range(1, K.shape[1])], 1)
    sm = out.copy()
    sm[1:-1] = (out[:-2] + 2 * out[1:-1] + out[2:]) / 4
    return sm


def tube_loft(path, sizes, n_pts: int = 24, e: float = 2.4) -> Manifold:
    """Loft of superellipse sections (half width, half height) along a straight or gently bent polyline."""
    P = [np.asarray(p, float) for p in path]
    rings = []
    for k, (p, (w, h)) in enumerate(zip(P, sizes)):
        t = P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)]
        t /= np.linalg.norm(t)
        side = np.cross([0, 0, 1.0], t)
        side /= np.linalg.norm(side)
        up = np.cross(t, side)
        sec = _superellipse(n_pts, w, h, e)
        rings.append(p + sec[:, :1] * side + sec[:, 1:] * up)
    return loft(rings)


def _blade(hinge, u, z0, s, pitch: float = 0.0) -> Manifold:
    """One folded propeller blade lying along u from its hinge: swept (scimitar) planform, thin section, the chord
    twisted by the blade pitch (signed pitch length in m: angle atan(pitch / (2 pi r)), capped at 35 deg). The two
    blades of a folded prop carry opposite signs, so seen end-on they open into a narrow V."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    n = np.array([-u[1], u[0]])
    R = s["drone"]["prop_d"] / 2
    # r from the motor axis, chord, chord-centre offset toward the leading edge, thickness (EST from DJI images)
    st = [(0.012, 0.016, 0.000, 0.0040), (0.025, 0.026, 0.002, 0.0030), (0.045, 0.030, 0.004, 0.0025),
          (0.070, 0.029, 0.005, 0.0021), (0.095, 0.026, 0.005, 0.0019), (0.120, 0.021, 0.004, 0.0017),
          (0.140, 0.016, 0.002, 0.0015), (R - 0.008, 0.010, -0.001, 0.0013), (R, 0.004, -0.003, 0.0010)]
    rings = []
    n3, z3 = np.array([n[0], n[1], 0.0]), np.array([0.0, 0.0, 1.0])
    for r, c, o, t in st:
        th = math.copysign(min(math.atan(abs(pitch) / (2 * math.pi * r)), math.radians(35.0)), pitch)
        a, b = n3 * math.cos(th) + z3 * math.sin(th), -n3 * math.sin(th) + z3 * math.cos(th)
        cen = np.array([*(np.asarray(hinge, float) + u * (r - 0.012) + n * o), z0 + t / 2])
        sec = _superellipse(12, c / 2, t / 2, 2.0)
        rings.append(cen + sec[:, :1] * a + sec[:, 1:] * b)
    return loft(rings)


def drone(s) -> list[tuple[str, str, Manifold]]:
    """M4D-class quadcopter. DJI: L/W/H with and without props, motor spacings, 2611 motors (prop D derived).
    MEAS (two DJI product images, bundle-adjusted on DJI's motor axes; spec.json drone.shape): body plan and
    top height, nose and vision sensors, GNSS dome, beacon, arm exits, directions and heights, fins, pods, gimbal head
    and lens layout. EST: belly, gimbal depth, blade planform (geometry.drone_layout holds the shared numbers)."""
    d = s["drone"]
    L = drone_layout(s)
    out: list[tuple[str, str, Manifold]] = []
    # fuselage: an upper shell (the plan the top photo sees; the nose overhangs the gimbal) over a narrower lower
    # body whose front rises toward the gimbal, both superellipse sections through (x, half width, z bottom, z top)
    def section_loft(keys, n):
        xs = np.linspace(keys[0][0], keys[-1][0], n)
        rings = []
        for x, (hw, zb, zt) in zip(xs, _profile(keys, xs)):
            sec = _superellipse(56, hw, (zt - zb) / 2, 3.2)
            rings.append(np.column_stack([np.full(len(sec), x), sec[:, 0], sec[:, 1] + (zb + zt) / 2]))
        return loft(rings)

    parts = [section_loft(L["body_keys"], 64), section_loft(L["lower_keys"], 40)]
    # dark block under the gimbal, its front face sloping up to the gimbal's back
    ch = L["chin"]
    (cx0, cx1), (cz0, cz1), chw, cfb = ch["x"], ch["z"], ch["half_w"], ch["front_bottom_x"]
    n_ = np.array([-(cz1 - cz0), 0.0, cx1 - cfb])
    n_ /= np.linalg.norm(n_)
    off = float(n_ @ np.array([cfb, 0, cz0]))
    chin = ext_x(rrect(-chw, cz0, chw, cz1 + 0.01, 0.006), cx0, cx1).trim_by_plane(tuple(n_), off)
    slots = [box((cx0 - 0.05, yk - 0.0007, cz0 + 0.004), (cx1 + 0.05, yk + 0.0007, cz1 - 0.004)).trim_by_plane(tuple(-n_), -(off + 0.0015))
             for yk in np.linspace(-chw + 0.005, chw - 0.005, 5)]
    out.append(("drone_chin", "drone_dark", chin - union(slots)))
    # arms: straight from the body sides up into the motor pods, thick at the root (fillet)
    fins = []
    aw, ah = L["arm_w"] / 2, L["arm_h"] / 2
    ar = L["arm_root"]
    for k, ((ex, ey, ez), (mx, my), fy) in enumerate(zip(L["arm_exit"], L["motors"], L["fin_y"])):
        E = np.array([ex, ey, ez])
        T = np.array([mx, my, L["arm_z_pod"]])
        root = ar["front"] if k < 2 else ar["rear"]
        ts = list(ar["t"]) + [0.5, 0.9, 1.0]
        sizes = [(w, h) for w, h, _ in root] + [(aw, ah)] * 3
        dz = [z for _, _, z in root] + [0.0] * 3
        parts.append(tube_loft([E + t * (T - E) + np.array([0, 0, z]) for t, z in zip(ts, dz)], sizes))
        # antenna fin on top of the arm: a tapered tab, rounded on top
        tf = (fy - abs(ey)) / (abs(my) - abs(ey))
        F = E + tf * (T - E)
        u = (T - E)[:2] / np.linalg.norm((T - E)[:2])
        nv = np.array([-u[1], u[0]])
        pts = []
        top = [(0.0042 * math.cos(a), ah + L["fin_h"] - 0.0042 + 0.0042 * math.sin(a)) for a in np.linspace(0, math.pi, 9)]
        for a, b, zz in [(a, b, 0.0) for a in (-0.0080, 0.0080) for b in (-0.0028, 0.0028)] + \
                        [(a, b, zz) for a, zz in top for b in (-0.0017, 0.0017)]:
            q = F[:2] + u * a + nv * b
            pts.append((q[0], q[1], F[2] + zz))
        fins.append(Manifold.hull_points(np.array(pts)))
    # vision sensor bumps on the nose corners; each lens is a cap of a slightly larger bump (proud by ~0.4 mm)
    lx, ly, lz = L["eye_lens"]
    eyes = []
    for sy in (-1, 1):
        ax = np.array([0.85, sy * 0.30, 0.42])
        ax /= np.linalg.norm(ax)
        c = np.array([lx, sy * ly, lz]) - ax * 0.0104
        parts.append(Manifold.sphere(1.0, 40).scale((0.0105, 0.0105, 0.0095)).translate(tuple(c)))
        cap = Manifold.sphere(1.0, 40).scale((0.0109, 0.0109, 0.0099)).translate(tuple(c))
        eyes.append(cap ^ cyl(c, c + ax * 0.02, 0.0066, seg=32))
    out.append(("drone_body", "drone", union(parts + fins)))
    out.append(("drone_eyes", "lens", union(eyes)))
    # top: GNSS/RTK antenna dome, beacon, dongle-bay cover, E-Port cap
    gn = L["gnss"]
    gx, gr, gt, gf_, grb = gn["x"], gn["r"], gn["top"], gn["fillet"], gn["r_base"]
    z0g = 0.128
    prof = [(0.0, z0g), (grb + 0.0012, z0g), (grb + 0.0012, z0g + 0.004), (grb, z0g + 0.0075)]
    prof += [(gr - gf_ + gf_ * math.cos(a), gt - gf_ + gf_ * math.sin(a)) for a in np.linspace(0.0, math.pi / 2, 10)]
    prof += [(0.0, gt)]
    out.append(("drone_gnss", "drone", poly(prof).revolve(56).translate((gx, 0, 0))))
    out.append(("drone_beacon", "light", cyl((L["beacon_x"], 0, 0.128), (L["beacon_x"], 0, 0.1385), 0.0058, seg=24)))
    (px0, py0), (px1, py1) = L["top_plate"]
    cx_, cl, cw = L["eport_cap"]
    out.append(("drone_top", "grip", ext_z(rrect(px0, py0, px1, py1, 0.003), 0.1300, 0.1374)
                + ext_z(rrect(cx_ - cl / 2, -cw / 2, cx_ + cl / 2, cw / 2, cw / 2 - 0.0002), 0.1300, 0.1385)))
    # belly: the internal charging modules (ribbed pads) are the lowest surface; they carry the aircraft on the pad
    bl = L["belly"]
    pads = []
    for xa, xb in bl["charge_pads"]:
        pads.append(ext_z(rrect(xa, -bl["half_w"], xb, bl["half_w"], 0.004), 0.0006, bl["z"] + 0.004))
        pads += [ext_z(rrect(xa + 0.004, -bl["half_w"] + 0.004 + k * 0.012, xb - 0.004, -bl["half_w"] + 0.010 + k * 0.012, 0.001),
                       0.0, 0.0008) for k in range(int((2 * bl["half_w"] - 0.006) // 0.012))]
    out.append(("drone_belly", "drone_dark", union(pads)))
    # gimbal camera: head under the nose, ribbed pitch drums on both sides, lens cluster on the face
    g = L["gimbal"]
    gf, gb, (gz0, gz1), hw = g["face_x"], g["back_x"], g["z"], g["half_w"]
    cr = g["corner_r"]
    head = ext_x(rrect(-hw, gz0, hw, gz1, cr), gb, gf - 0.0015) + \
        ext_x(rrect(-hw + 0.0015, gz0 + 0.0015, hw - 0.0015, gz1 - 0.0015, cr - 0.0015), gf - 0.0015, gf)
    dr = g["drum"]
    drums = []
    for sy in (-1, 1):
        y0, y1 = sy * (hw - 0.001), sy * dr["y_out"]
        drums.append(cyl((dr["x"], y0, dr["z"]), (dr["x"], y1, dr["z"]), dr["r"] - 0.0008, seg=48))
        for k in range(4):
            ya = y0 + (y1 - y0) * (0.12 + 0.22 * k)
            drums.append(cyl((dr["x"], ya, dr["z"]), (dr["x"], ya + sy * 0.0014, dr["z"]), dr["r"], seg=48))
    collar = cyl((gb + 0.020, 0, gz1 - 0.002), (gb + 0.020, 0, 0.125), 0.016, seg=40)
    out.append(("drone_gimbal", "drone_dark", union([head, collar] + drums)))
    w = g["wide"]
    lens = [cyl((gf - 0.001, w["y"], w["z"]), (gf + 0.0015, w["y"], w["z"]), w["r"], seg=48)]
    for key in ("medium_tele", "tele"):
        (ya, za), (yb, zb) = g[key]
        lens.append(ext_x(rrect(ya, za, yb, zb, 0.0018), gf - 0.001, gf + 0.0010))
    out.append(("drone_lenses", "lens", union(lens)))
    (ya, za), (yb, zb) = g["lrf"]
    out.append(("drone_lrf", "metal", ext_x(rrect(ya, za, yb, zb, 0.0015), gf - 0.001, gf + 0.0008)))
    # motor pods (rounded bottoms, front LEDs), 2611 bells, prop mounts and folded propellers. Parked, the two blades
    # of each prop lie along X pointing inboard, a narrow V (DJI product images and video): that is how the
    # props stay inside the closed covers.
    pods, bells, hubs, blades, leds = [], [], [], [], []
    zp0, zp1 = L["z_pod"]
    zb0, zb1 = L["z_bell"]
    pr, br = d["pod_d_x"] / 2, d["bell_d"] / 2
    rb = 0.006  # rounded pod bottom
    for (mx, my), fold in zip(L["motors"], L["prop_fold"]):
        pods.append(cyl((mx, my, zp0 + rb), (mx, my, zp1), pr, seg=48) +
                    Manifold.sphere(1.0, 40).scale((pr, pr, rb)).translate((mx, my, zp0 + rb)).trim_by_plane((0, 0, -1), -(zp0 + rb)))
        bells.append(cyl((mx, my, zb0), (mx, my, zb1 - 0.002), br, seg=48) + cyl((mx, my, zb1 - 0.002), (mx, my, zb1), br, br - 0.0015, seg=48))
        hubs.append(cyl((mx, my, zb1), (mx, my, zb1 + 0.006), 0.0105, seg=32) +
                    cyl((mx, my, zb1 + 0.006), (mx, my, L["z_hub_top"]), 0.0065, 0.0050, seg=24))
        if mx > 0:
            o = np.array([mx, my]) + np.array([0.6, math.copysign(0.8, my)]) * (pr - 0.0006)
            leds.append(cyl((o[0], o[1], zp0 + rb + 0.002), (o[0], o[1], zp0 + rb + 0.005), 0.0028, seg=16))
        u0 = np.array([math.cos(math.radians(fold)), math.sin(math.radians(fold))])
        nv = np.array([-u0[1], u0[0]])
        for k, (side, spread) in enumerate(((1, 4.0), (-1, -4.0))):
            a = math.radians(spread)
            u = np.array([u0[0] * math.cos(a) - u0[1] * math.sin(a), u0[0] * math.sin(a) + u0[1] * math.cos(a)])
            hinge = np.array([mx, my]) + nv * side * 0.0085 + u0 * 0.012
            # pitch 0.16 m EST (a 13-inch prop with a 6.4-inch pitch, if DJI's "1364" reads like that: UNVERIFIED)
            blades.append(_blade(hinge, u, L["prop_z"] + 0.0028 * k, s, pitch=0.16 * side))
    out.append(("drone_pods", "drone", union(pods)))
    out.append(("drone_motors", "motor", union(bells)))
    out.append(("drone_props", "prop", union(hubs + blades)))
    out.append(("drone_leds", "light", union(leds)))
    return out


def drone_inertia(s) -> dict:
    """Mass properties from DJI's 1850 g, split over the main parts (battery 640 g is DJI; the split is EST)."""
    d = s["drone"]
    L = drone_layout(s)
    bl, bw, bh = d["battery"]["lwh"]
    parts = [(d["battery"]["mass_kg"], (-0.020, 0.0, 0.060), (bl, bh, bw)),  # long side along x, 96 mm upright
             (0.560, (0.030, 0.0, 0.075), (0.240, 0.100, 0.130)),
             (0.200, (0.142, 0.0, 0.084), (0.048, 0.084, 0.073))]
    zm = (L["z_pod"][0] + L["z_bell"][1]) / 2
    for (ex, ey, ez), (mx, my) in zip(L["arm_exit"], L["motors"]):
        parts.append((0.101, (mx, my, zm), (0.036, 0.036, 0.04)))
        parts.append((0.012, ((ex + mx) / 2, (ey + my) / 2, (ez + L["arm_z_pod"]) / 2), (0.17, 0.018, 0.016)))
    scale = d["mass_kg"] / sum(p[0] for p in parts)
    parts = [(pm * scale, c, sz) for pm, c, sz in parts]
    M = sum(p[0] for p in parts)
    com = sum(np.array(c) * pm for pm, c, _ in parts) / M
    inertia = np.zeros((3, 3))
    for pm, c, (a, b, cc) in parts:
        r = np.array(c) - com
        inertia += pm / 12.0 * np.diag([b * b + cc * cc, a * a + cc * cc, a * a + b * b]) + pm * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return {"mass": M, "com": com.tolist(), "inertia": inertia}


# ------------------------------------------------------------------------------------------------ assembly

def _mesh(m) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(m, tuple):
        return m
    mesh = _simplify(m).to_mesh()
    return np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3], np.asarray(mesh.tri_verts, dtype=np.int64)


def build_parts(spec: dict, simplify_tol: float = 0.0, simplify_min_tris: int = 4000) -> list[dict]:
    """All visual meshes: dicts with name, parent, mat, verts (in the parent frame), faces. simplify_tol > 0 runs
    manifold3d's simplify() (surfaces move less than the tolerance) on meshes over simplify_min_tris triangles."""
    R.simplify = (simplify_tol, simplify_min_tris)
    try:
        return _build_parts(spec)
    finally:
        R.simplify = (0.0, 0)


def _build_parts(spec: dict) -> list[dict]:
    R.mat.clear()
    c = spec["cover"]
    piv = {"cover_r": np.array([c["pivot_x"], 0.0, c["pivot_z"]]), "cover_l": np.array([-c["pivot_x"], 0.0, c["pivot_z"]])}
    parts: list[dict] = []

    def put(name, parent, mat, m, offset=None):
        v, f = _mesh(m)
        if len(f) == 0:
            return
        if offset is not None:
            v = v - offset
        parts.append({"name": name, "parent": parent, "mat": mat, "verts": v, "faces": f})

    for name, mat, m in base_and_body(spec) + pad(spec):
        put(name, "dock", mat, m)
    for side, mats in covers(spec).items():
        for mat, vf in mats.items():
            put(f"{side}_shell_{mat}", side, mat, vf, piv[side])
    for side, items in cover_fittings(spec).items():
        for name, mat, m in items:
            put(f"{side}_{name}", side, mat, m, piv[side])
    for name, parent, mat, m in actuator_meshes(spec):
        put(name, parent, mat, m)
    for name, mat, m in drone(spec):
        put(name, "drone", mat, m)
    return parts
