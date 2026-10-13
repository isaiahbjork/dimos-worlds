"""Dock 3-class station + M4D-class quadcopter: frames, cover kinematics, cameras, collision proxies, materials.

Pure Python (a renderer's embedded interpreter can import it too). The visible surfaces are triangle meshes from
shapes.py (manifold3d); gen_mjcf.py writes them to assets/ for MuJoCo and other renderers. This module holds what
the builders and the runtime need without manifold3d.

Frames (see spec.json):
  dock     origin on the ground under the dock centre, +X toward the gauge cover, -Y front, +Z up
  cover_r  origin on the right hinge pivot, axes = dock axes when the cover is closed; opens +angle about +Y
  cover_l  origin on the left hinge pivot; mirror of cover_r, opens +angle about -Y
  act_cyl_* / act_rod_*  actuator cylinder (origin on its body pin) and rod (origin on its cover pin); local +Z
           points from the body pin to the cover pin
  drone    origin on the pad contact plane under the motor-rectangle centre, nose +X

A collision proxy: {"name", "parent", "kind" (box | cyl | capsule | hull), "size", "pos", "quat"}; box size =
half extents, cyl size = (radius, half length) along local Z, capsule "from"/"to" + radius, hull = a convex
mesh asset named in "mesh". quat is (w, x, y, z).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

SPEC_PATH = Path(__file__).resolve().with_name("spec.json")
Q0 = (1.0, 0.0, 0.0, 0.0)


def load_spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


# ---------------------------------------------------------------- small vector helpers

def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _scale(a, k):
    return tuple(x * k for x in a)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = math.sqrt(sum(c * c for c in a))
    return tuple(c / n for c in a)


def rot_y(v, deg):
    """Rotate (x, y, z) about +Y by deg (right-handed): +X goes toward -Z."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    x, y, z = v
    return (x * c + z * s, y, -x * s + z * c)


def q_axis(axis, deg):
    a = math.radians(deg) / 2
    n = _norm(axis)
    return (math.cos(a), *(math.sin(a) * c for c in n))


# ---------------------------------------------------------------- cover kinematics

def pivot(spec: dict, side: str = "r") -> tuple[float, float, float]:
    c = spec["cover"]
    return ((c["pivot_x"] if side == "r" else -c["pivot_x"]), 0.0, c["pivot_z"])


def cover_world(spec: dict, p_local, angle_deg: float, side: str = "r"):
    """Cover-local point (relative to its pivot) -> dock frame at a cover angle (0 closed, open_deg open)."""
    return _add(pivot(spec, side), rot_y(p_local, angle_deg if side == "r" else -angle_deg))


def cover_dir(d_local, angle_deg: float, side: str = "r"):
    return rot_y(d_local, angle_deg if side == "r" else -angle_deg)


def closed_to_local(spec: dict, p_closed, side: str = "r"):
    """A dock-frame point on the closed cover -> the cover's local frame."""
    return _sub(p_closed, pivot(spec, side))


def actuator_pins(spec: dict, side: str = "r"):
    """(body pin, cover pin when closed) in the dock frame."""
    a = spec["actuator"]
    sx = 1 if side == "r" else -1
    bp = (sx * a["body_pivot"][0], a["body_pivot"][1], a["body_pivot"][2])
    cp = (sx * a["cover_pin"][0], a["cover_pin"][1], a["cover_pin"][2])
    return bp, cp


def actuator_pose(spec: dict, angle_deg: float, side: str = "r"):
    """Body pin, cover pin and unit axis (body pin -> cover pin) at a cover angle, dock frame."""
    bp, cp = actuator_pins(spec, side)
    cpw = cover_world(spec, closed_to_local(spec, cp, side), angle_deg, side)
    return bp, cpw, _norm(_sub(cpw, bp))


# ---------------------------------------------------------------- cameras

def camera_axes(look, up=(0.0, 0.0, 1.0)):
    """MuJoCo xyaxes for a camera looking along `look` (camera looks down its -Z)."""
    z = _norm(_scale(look, -1.0))
    x = _cross(up, z)
    if math.sqrt(sum(c * c for c in x)) < 1e-6:
        x = (1.0, 0.0, 0.0)
    x = _norm(x)
    y = _cross(z, x)
    return x, y


def dfov_to_fovy(dfov_deg: float, w: int, h: int) -> float:
    """Rectilinear lens: DJI quotes diagonal FOV; MuJoCo wants the vertical one."""
    t = math.tan(math.radians(dfov_deg) / 2)
    return math.degrees(2 * math.atan(t * h / math.hypot(w, h)))


def dfov_to_hfov(dfov_deg: float, w: int, h: int) -> float:
    t = math.tan(math.radians(dfov_deg) / 2)
    return math.degrees(2 * math.atan(t * w / math.hypot(w, h)))


def drone_layout(spec: dict) -> dict:
    """Aircraft layout from DJI's numbers (motor spacings, height) plus estimates for the shape (spec labels)."""
    d = spec["drone"]
    fr = d["motor_fr"] / 2
    motors = [(fr, d["motor_lr_front"] / 2), (fr, -d["motor_lr_front"] / 2),
              (-fr, d["motor_lr_rear"] / 2), (-fr, -d["motor_lr_rear"] / 2)]
    H = d["lwh"][2]
    sh = d["shape"]
    (fx, fy, fz), (rx, ry, rz) = sh["arm_exit_front"], sh["arm_exit_rear"]
    fold = sh["prop_fold_deg"]
    g = sh["gimbal"]
    keys = sh["body"]["keys"]
    return {"motors": motors, "z_top": H, "z_pod": tuple(sh["z_pod"]), "z_bell": (sh["z_pod"][1], H),
            "z_hub_top": d["lwh_props"][2], "prop_z": H + 0.0015,
            # parked blades lie along X pointing inboard (front props aft, rear props forward), leaning toward the
            # centreline; motor order: front-left, front-right, rear-left, rear-right (deg from +X)
            "prop_fold": [180.0 + fold["front"], 180.0 - fold["front"], -fold["rear"], fold["rear"]],
            "arm_exit": [(fx, fy, fz), (fx, -fy, fz), (rx, ry, rz), (rx, -ry, rz)], "arm_z_pod": sh["arm_z_pod"],
            "arm_w": sh["arm_w"], "arm_h": sh["arm_h"], "fin_y": [sh["fin_y"][0]] * 2 + [sh["fin_y"][1]] * 2,
            "fin_h": sh["fin_h"], "body_keys": keys, "body_x": (keys[0][0], keys[-1][0]),
            "lower_keys": sh["lower"]["keys"], "chin": sh["chin"], "arm_root": sh["arm_root"],
            "gnss": sh["gnss"], "beacon_x": sh["beacon_x"], "eye_lens": tuple(sh["eye_lens"]),
            "top_plate": sh["top_plate"], "eport_cap": sh["eport_cap"], "belly": sh["belly"], "gimbal": g,
            "gimbal_front": g["face_x"], "gimbal_z": tuple(g["z"]),
            "wide_cam": (g["face_x"] + 0.0015, g["wide"]["y"], g["wide"]["z"]),
            "feet": [(x, sy * y) for x, y in sh["feet"] for sy in (1, -1)]}


def cameras(spec: dict) -> list[dict]:
    """All cameras in their parent's local frame: name, parent, pos, look, up, lens fields."""
    out = []
    for name in ("dock_internal", "dock_external"):
        c = spec["cameras"][name]
        out.append({**c, "name": name, "pos": closed_to_local(spec, c["pos_closed"], c["parent"][-1]),
                    "look": tuple(c["look_closed"]), "up": tuple(c["up_closed"])})
    w = spec["drone"]["cameras"]["wide"]
    L = drone_layout(spec)
    out.append({"name": "drone_wide", "parent": "drone", "pos": L["wide_cam"], "look": (1.0, 0.0, 0.0),
                "up": (0.0, 0.0, 1.0), "width": w["w"], "height": w["h"], "dfov_deg": w["dfov_deg"],
                "lens": "rectilinear"})
    return out


def camera_world(spec: dict, cam: dict, cover_deg: float, drone_pos=(0.0, 0.0, 0.0)):
    """World position, look and up of a camera at a cover angle (drone at drone_pos, nose +X)."""
    if cam["parent"] == "drone":
        return _add(drone_pos, cam["pos"]), cam["look"], cam["up"]
    side = cam["parent"][-1]
    return (cover_world(spec, cam["pos"], cover_deg, side), cover_dir(cam["look"], cover_deg, side),
            cover_dir(cam["up"], cover_deg, side))


# ---------------------------------------------------------------- collision proxies

def _box(name, parent, lo, hi, quat=Q0):
    c = _scale(_add(lo, hi), 0.5)
    h = _scale(_sub(hi, lo), 0.5)
    return {"name": name, "parent": parent, "kind": "box", "size": tuple(abs(v) for v in h), "pos": c, "quat": quat}


def collision(spec: dict) -> list[dict]:
    """Simple solids for contact. The drone touches only the pad plate when it lands centred."""
    b, bd, p, c, rt = spec["base"], spec["body"], spec["pad"], spec["cover"], spec["rtk"]
    out = []
    x0, x1 = b["bracket_x"]
    i = 0
    for sx in (-1, 1):
        for ya, yb in (b["bracket_y_front"], b["bracket_y_rear"]):
            out.append(_box(f"base_bracket_{i}", "dock", (min(sx * x0, sx * x1), ya, 0.0),
                            (max(sx * x0, sx * x1), yb, b["frame_z"][0])))
            i += 1
    (fx0, fx1), (fy0, fy1) = b["frame_x"], b["frame_y"]
    out.append(_box("base_frame", "dock", (fx0, fy0, b["frame_z"][0]), (fx1, fy1, b["frame_z"][1])))
    ix, iy = bd["tray_inner_half"]
    tx, ty = bd["tray_half"]
    fz, rz = p["floor_z"], p["rim_z"]
    zb = min(bd["top_z"] - 0.017, fz - 0.020)  # the pad pocket goes down into the cabinet
    out.append(_box("cabinet", "dock", (-bd["end_x"], bd["front_y"], b["frame_z"][1]), (bd["end_x"], bd["back_y"], zb)))
    out.append(_box("pad_plate", "dock", (-ix, -iy, zb), (ix, iy, fz)))
    out.append(_box("tray_rim_front", "dock", (-tx, -ty, zb), (tx, -iy, rz)))
    out.append(_box("tray_rim_back", "dock", (-tx, iy, zb), (tx, ty, rz)))
    wz = bd["tray_wall_z"]  # the +-X ends stop at the housing wall (no rim there)
    out.append(_box("tray_rim_left", "dock", (-tx, -iy, zb), (-ix, iy, wz)))
    out.append(_box("tray_rim_right", "dock", (ix, -iy, zb), (tx, iy, wz)))
    for tag in ("f", "r"):
        for k in range(3):  # shapes.pad_collision: each block in three convex pieces along X
            out.append({"name": f"pad_block_{tag}{k}", "parent": "dock", "kind": "hull", "mesh": f"pad_block_{tag}{k}",
                        "pos": (0.0, 0.0, 0.0), "quat": Q0})
    cx, cy = rt["puck_center"]
    z0, z1 = rt["puck_z"]
    out.append({"name": "rtk_puck", "parent": "dock", "kind": "cyl", "size": (rt["puck_r"], (z1 - z0) / 2),
                "pos": (cx, cy, (z0 + z1) / 2), "quat": Q0})
    # covers: roof slab(s), end wall, front and back walls, in cover-local frames. The parting line is off
    # centre (x_parting), so the right cover carries the ridge and is longer than the left one.
    t, hy, xe, zt, k = c["wall_t"], c["half_y"], c["x_end"], c["roof_ridge_z"], c["roof_slope"]
    xp = c["x_parting"]
    ang = math.degrees(math.atan(k))
    for side in ("r", "l"):
        sx = 1 if side == "r" else -1
        pv = pivot(spec, side)

        def L(pt, pv=pv):
            return _sub(pt, pv)

        a, b = (0.0, xe) if side == "r" else (abs(xp), xe)  # |x| span of the sloped slab
        mid = (a + b) / 2
        out.append({"name": f"cover_{side}_roof", "parent": f"cover_{side}", "kind": "box",
                    "size": ((b - a) / 2 / math.cos(math.radians(ang)), hy - 0.01, t / 2),
                    "pos": L((sx * mid, 0.0, zt - k * mid - t / 2 - 0.004)), "quat": q_axis((0, 1, 0), ang * sx)})
        if side == "r":
            out.append(_box("cover_r_roof_crest", "cover_r", L((xp, -hy + 0.01, zt - 0.006 - t)), L((0.0, hy - 0.01, zt - 0.006))))
        out.append(_box(f"cover_{side}_end", f"cover_{side}", L((min(sx * (xe - t), sx * xe), -hy, c["bottom_z"] + 0.03)),
                        L((max(sx * (xe - t), sx * xe), hy, zt - k * xe - 0.03))))
        x_in = xp if side == "r" else xp - 0.002
        for wy, tag in ((-1, "front"), (1, "back")):
            out.append(_box(f"cover_{side}_{tag}", f"cover_{side}",
                            L((min(x_in, sx * (xe - 0.02)), min(wy * hy, wy * (hy - t)), bd["cutout"]["top_z"] + 0.005)),
                            L((max(x_in, sx * (xe - 0.02)), max(wy * hy, wy * (hy - t)), zt - 0.06))))
    # drone
    L = drone_layout(spec)
    bx0, bx1 = L["body_x"]
    zb = L["belly"]["z"]
    lk = L["lower_keys"]
    lw = max(k[1] for k in lk)
    out.append(_box("drone_body", "drone", (lk[0][0], -lw, zb), (0.060, lw, 0.080)))  # lower body (front rises, see shapes)
    out.append(_box("drone_body_top", "drone", (bx0, -0.050, 0.080), (bx1 - 0.004, 0.050, 0.136)))
    g = L["gimbal"]
    dy = g["drum"]["y_out"]
    out.append(_box("drone_gimbal", "drone", (g["back_x"], -dy, g["z"][0]), (g["face_x"], dy, g["z"][1])))
    for k, ((ex, ey, ez), (mx, my)) in enumerate(zip(L["arm_exit"], L["motors"])):
        out.append({"name": f"drone_arm_{k}", "parent": "drone", "kind": "capsule", "from": (ex, ey, ez),
                    "to": (mx, my, L["arm_z_pod"]), "size": (L["arm_w"] / 2,), "pos": (0, 0, 0), "quat": Q0})
        z0, z1 = L["z_pod"][0], L["z_top"]
        out.append({"name": f"drone_motor_{k}", "parent": "drone", "kind": "cyl",
                    "size": (spec["drone"]["pod_d_x"] / 2, (z1 - z0) / 2), "pos": (mx, my, (z0 + z1) / 2), "quat": Q0})
    for k, (fx_, fy_) in enumerate(L["feet"]):
        out.append(_box(f"drone_foot_{k}", "drone", (fx_ - 0.009, fy_ - 0.007, 0.0), (fx_ + 0.009, fy_ + 0.007, zb + 0.005)))
    return out


# ---------------------------------------------------------------- materials (no logos, no trademarks)

MATERIALS = {  # name: (srgb rgba, roughness, metallic)
    "shell": ((0.91, 0.92, 0.92, 1), 0.42, 0.0),
    "liner": ((0.15, 0.16, 0.17, 1), 0.70, 0.0),
    "seal": ((0.05, 0.05, 0.055, 1), 0.75, 0.0),
    "grip": ((0.72, 0.73, 0.74, 1), 0.55, 0.0),
    "pad": ((0.12, 0.13, 0.14, 1), 0.70, 0.0),
    "housing": ((0.19, 0.21, 0.23, 1), 0.55, 0.0),
    "pad_block": ((0.21, 0.23, 0.25, 1), 0.55, 0.0),
    "light_panel": ((0.86, 0.87, 0.88, 1), 0.35, 0.0),
    "panel": ((0.86, 0.87, 0.87, 1), 0.50, 0.0),
    "panel_dark": ((0.74, 0.75, 0.76, 1), 0.55, 0.0),
    "cabinet_in": ((0.16, 0.16, 0.17, 1), 0.65, 0.0),
    "breaker": ((0.88, 0.88, 0.86, 1), 0.50, 0.0),
    "label": ((0.42, 0.43, 0.44, 1), 0.60, 0.0),
    "frame": ((0.80, 0.81, 0.82, 1), 0.55, 0.1),
    "bracket": ((0.24, 0.24, 0.25, 1), 0.60, 0.2),
    "black": ((0.05, 0.05, 0.05, 1), 0.55, 0.0),
    "marker": ((0.96, 0.96, 0.96, 1), 0.40, 0.0),
    "yellow": ((0.95, 0.74, 0.04, 1), 0.45, 0.0),
    "red": ((0.82, 0.07, 0.05, 1), 0.35, 0.0),
    "cabinet": ((0.12, 0.12, 0.13, 1), 0.55, 0.0),
    "smoked_glass": ((0.10, 0.10, 0.11, 0.55), 0.08, 0.0),
    "metal": ((0.70, 0.71, 0.73, 1), 0.30, 0.85),
    "actuator": ((0.15, 0.15, 0.16, 1), 0.45, 0.3),
    "camera": ((0.13, 0.13, 0.14, 1), 0.50, 0.0),
    "lens": ((0.02, 0.02, 0.03, 1), 0.05, 0.0),
    "light": ((0.97, 0.97, 0.92, 1), 0.10, 0.0),
    "drone": ((0.70, 0.71, 0.72, 1), 0.55, 0.0),
    "drone_dark": ((0.30, 0.31, 0.32, 1), 0.55, 0.0),
    "motor": ((0.12, 0.12, 0.13, 1), 0.45, 0.4),
    "prop": ((0.05, 0.05, 0.055, 1), 0.45, 0.0),
}
