"""The warehouse as a DimOS scene package: a robot-agnostic MuJoCo world built from layout.json.

    python -m dimos_worlds.warehouse.scene        # (re)cooks src/dimos_worlds/warehouse/scene/

The package directory holds:
  warehouse.xml     MJCF: floor, walls, pallet racking A-D with the tote pick bays, conveyor, pick table, pallets,
                    the arm pedestal, 17 totes, lights and the six fixed cameras. Everything is static (no joints), so
                    a robot composed into it keeps qpos[7:] = its own joints (the DimOS walking policies read that).
  places.json       named places (points at each work spot, the arm cell as a stay-out), in world x/y
  scene.meta.json   DimOS scene-package metadata (dimos.simulation.scene_assets.spec), written by DimOS's own
                    ScenePackage.write_metadata so the format matches what DimOS loads

Frame: the layout frame is the world frame (x east, y north, z up, metres; origin = SW interior corner).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from dimos_worlds.warehouse.cell import (
    ARM_BASE,
    ARM_PEDESTAL_HALF,
    CELL_SIZE,
    LAYOUT,
    LOCATIONS,
    TOTE_SIZE,
    initial_poses,
)

PACKAGE_DIR = Path(__file__).resolve().with_name("scene")
SCENE_XML = PACKAGE_DIR / "warehouse.xml"
PLACES_JSON = PACKAGE_DIR / "places.json"
META_JSON = PACKAGE_DIR / "scene.meta.json"

WALL_HEIGHT_M = 3.0  # cut-away walls: high enough for any robot's sensors, low enough to see in from above
TOTE_RGBA = "0.16 0.38 0.78 1"


def _f(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".") if v != 0 else "0"


def _vec(vs) -> str:  # noqa: ANN001
    return " ".join(_f(float(v)) for v in vs)


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def build_root() -> ET.Element:
    """The scene as an MJCF element tree (no robot)."""
    root = ET.Element("mujoco", model="warehouse")
    ET.SubElement(root, "compiler", angle="radian")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", diffuse="0.5 0.5 0.5", ambient="0.3 0.3 0.3", specular="0 0 0")
    ET.SubElement(visual, "quality", shadowsize="4096")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="wh_floor", type="2d", builtin="checker", rgb1="0.62 0.62 0.6",
                  rgb2="0.58 0.58 0.56", width="512", height="512")
    ET.SubElement(asset, "material", name="wh_floor", texture="wh_floor", texrepeat="60 40", reflectance="0.05")
    ET.SubElement(asset, "texture", name="wh_sky", type="skybox", builtin="gradient", rgb1="0.86 0.88 0.9",
                  rgb2="0.55 0.58 0.62", width="256", height="256")
    for name, rgba in (("wh_steel", "0.25 0.35 0.55 1"), ("wh_beam", "0.95 0.55 0.1 1"),
                       ("wh_wood", "0.72 0.56 0.36 1"), ("wh_belt", "0.15 0.15 0.15 1"),
                       ("wh_frame", "0.55 0.55 0.58 1"), ("wh_tote", TOTE_RGBA), ("wh_table", "0.8 0.8 0.78 1"),
                       ("wh_mark", "0.95 0.8 0.1 1"), ("wh_load", "0.62 0.48 0.32 1"),
                       ("wh_wall", "0.82 0.82 0.8 1")):
        ET.SubElement(asset, "material", name=name, rgba=rgba)

    wb = ET.SubElement(root, "worldbody")
    W, D = CELL_SIZE
    ET.SubElement(wb, "light", name="wh_sun", pos=f"{W / 2} {D / 2} 8", dir="0 0 -1", directional="true",
                  diffuse="0.7 0.7 0.7")
    ET.SubElement(wb, "light", name="wh_fill", pos=f"{W / 2} -4 6", dir="0 0.6 -0.8", diffuse="0.3 0.3 0.3")
    ET.SubElement(wb, "geom", name="floor", type="plane", size=f"{W / 2 + 2} {D / 2 + 2} 0.05",
                  pos=f"{W / 2} {D / 2} 0", material="wh_floor")
    ET.SubElement(wb, "camera", name="overview", pos=f"{W / 2} -6 18", xyaxes="1 0 0 0 0.85 0.53", fovy="60")
    for cam_id, cam in LAYOUT["cameras"].items():  # the fixed cameras
        mount, target = np.array(cam["mount"], float), np.array(cam["target"], float)
        fwd = (target - mount) / np.linalg.norm(target - mount)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        # MuJoCo fovy is vertical; at 16:9 a horizontal fov h has vertical 2 atan(tan(h/2) * 9/16).
        fovy = math.degrees(2 * math.atan(math.tan(math.radians(cam["hfov_deg"]) / 2) * 9 / 16))
        ET.SubElement(wb, "camera", name=cam_id, pos=_vec(mount), xyaxes=_vec((*right, *up)), fovy=f"{fovy:.1f}")

    def box(name: str, center, half, material: str) -> None:  # noqa: ANN001
        ET.SubElement(wb, "geom", name=name, type="box", pos=_vec(center), size=_vec(half), material=material)

    # Walls (cut away at WALL_HEIGHT_M). Doors are drawn closed: the roll-up door at night, the others shut.
    t = float(LAYOUT["building"]["wall_thickness"])
    h = WALL_HEIGHT_M / 2
    box("wall_s", (W / 2, -t / 2, h), (W / 2 + t, t / 2, h), "wh_wall")
    box("wall_n", (W / 2, D + t / 2, h), (W / 2 + t, t / 2, h), "wh_wall")
    box("wall_w", (-t / 2, D / 2, h), (t / 2, D / 2, h), "wh_wall")
    box("wall_e", (W + t / 2, D / 2, h), (t / 2, D / 2, h), "wh_wall")

    # Pallet racking: frames at every bay line, beams per level, a loaded pallet in most bins. The pick bay's
    # ground level has the tote decks instead (the shelf locations, below).
    for sid, sh in LAYOUT["shelves"].items():
        x0, y0, x1, y1 = sh["rect"]
        bay = sh["bay_length"]
        n = int(round((x1 - x0) / bay))
        pick_bay = sh["pick_bay"]["bay"]
        for k in range(n + 1):
            x = x0 + k * bay
            for j, y in enumerate((y0 + 0.04, y1 - 0.04)):
                box(f"rack{sid}_up{k}_{j}", (x, y, sh["frame_height"] / 2), (0.04, 0.04, sh["frame_height"] / 2),
                    "wh_steel")
        for lvl, bz in enumerate(sh["beam_z"]):
            for b in range(n):
                if b + 1 == pick_bay and lvl == 0:
                    continue
                cx = x0 + (b + 0.5) * bay
                for j, y in enumerate((y0 + 0.04, y1 - 0.04)):
                    box(f"rack{sid}_beam{lvl}_{b}_{j}", (cx, y, bz - 0.06), (bay / 2 - 0.04, 0.025, 0.06), "wh_beam")
        for i, b in enumerate(sh["bins"]):
            bx_, by_, bz_ = b["center"]
            hh = 0.45 + 0.35 * ((i * 7) % 5) / 4  # loads of a few heights, the same every build
            for j, dx in enumerate((-0.64, 0.64)):
                if (i + j) % 4 == 3:
                    continue  # some bins half empty
                box(f"rack{sid}_pal{i}_{j}", (bx_ + dx, by_, bz_ + 0.07), (0.6, 0.5, 0.07), "wh_wood")
                box(f"rack{sid}_load{i}_{j}", (bx_ + dx, by_, bz_ + 0.14 + hh / 2), (0.55, 0.45, hh / 2), "wh_load")
    for p in LAYOUT["pallets"]:  # decoration pallets with a load (the place pallets are drawn below)
        if p.get("place") or not p.get("load_height"):
            continue
        (cx, cy), (sx, sy, sz) = p["center"], p["size"]
        box(f"{p['id']}_deck", (cx, cy, sz / 2), (sx / 2, sy / 2, sz / 2), "wh_wood")
        box(f"{p['id']}_load", (cx, cy, sz + p["load_height"] / 2),
            (sx / 2 - 0.05, sy / 2 - 0.05, p["load_height"] / 2), "wh_load")
    for bol in LAYOUT["bollards"]:
        ET.SubElement(wb, "geom", name=bol["id"], type="cylinder", size=_vec((bol["r"], bol["height"] / 2)),
                      pos=_vec((bol["x"], bol["y"], bol["height"] / 2)), material="wh_mark")

    for loc in LOCATIONS:
        cx, cy = loc.center
        sx, sy = loc.size
        if loc.kind == "shelf":
            front = -1 if loc.approach_yaw > 0 else 1  # the aisle side of the rack
            for z in sorted({s.z for s in loc.slots}):
                box(f"{loc.id}_deck{z:.2f}", (cx, cy, z - 0.015), (sx / 2, sy / 2, 0.015), "wh_wood")
                for j, y in enumerate((cy - sy / 2 + 0.04, cy + sy / 2 - 0.04)):
                    box(f"{loc.id}_beam{z:.2f}_{j}", (cx, y, z - 0.06), (sx / 2, 0.025, 0.045), "wh_beam")
            box(f"{loc.id}_label", (cx, cy + front * (sy / 2 + 0.005), 0.30), (0.2, 0.004, 0.04), "wh_mark")
        elif loc.kind == "table":
            box(f"{loc.id}_top", (cx, cy, loc.top_z - 0.02), (sx / 2, sy / 2, 0.02), "wh_table")
            for i, (dx, dy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
                box(f"{loc.id}_leg{i}", (cx + dx * (sx / 2 - 0.04), cy + dy * (sy / 2 - 0.04), (loc.top_z - 0.04) / 2),
                    (0.025, 0.025, (loc.top_z - 0.04) / 2), "wh_frame")
        elif loc.kind == "conveyor":
            box(f"{loc.id}_belt", (cx, cy, loc.top_z - 0.03), (sx / 2, sy / 2, 0.03), "wh_belt")
            box(f"{loc.id}_frame", (cx, cy, (loc.top_z - 0.06) / 2), (sx / 2, sy / 2 - 0.05, (loc.top_z - 0.06) / 2),
                "wh_frame")
            box(f"{loc.id}_stop", (cx + sx / 2 + 0.02, cy, loc.top_z + 0.03), (0.02, sy / 2, 0.03), "wh_mark")
        elif loc.kind == "pallet":
            box(f"{loc.id}_deck", (cx, cy, loc.top_z - 0.01), (sx / 2, sy / 2, 0.01), "wh_wood")
            for i, dy in enumerate((-1, 0, 1)):
                box(f"{loc.id}_runner{i}", (cx, cy + dy * (sy / 2 - 0.05), (loc.top_z - 0.02) / 2),
                    (sx / 2, 0.05, (loc.top_z - 0.02) / 2), "wh_wood")
    bx, by, bz = ARM_BASE
    box("arm_pedestal", (bx, by, bz / 2), (*ARM_PEDESTAL_HALF, bz / 2), "wh_frame")

    # Totes, static: an open box (floor + four walls) at each start pose.
    hx, hy, hz = (v / 2 for v in TOTE_SIZE)
    wall = 0.008
    for tote_id, (x, y, z, yaw) in initial_poses().items():
        body = ET.SubElement(wb, "body", name=tote_id, pos=_vec((x, y, z)), quat=_vec(_yaw_quat(yaw)))
        ET.SubElement(body, "geom", type="box", pos=_vec((0, 0, -hz + wall)), size=_vec((hx, hy, wall)),
                      material="wh_tote")
        for px_, py_, sx_, sy_ in ((hx - wall, 0, wall, hy), (-hx + wall, 0, wall, hy),
                                   (0, hy - wall, hx, wall), (0, -hy + wall, hx, wall)):
            ET.SubElement(body, "geom", type="box", pos=_vec((px_, py_, 0)), size=_vec((sx_, sy_, hz)),
                          material="wh_tote")
    return root


def scene_xml() -> str:
    """The scene MJCF as a string (built from layout.json, identical to the cooked warehouse.xml)."""
    root = build_root()
    ET.indent(root)
    return ET.tostring(root, encoding="unicode") + "\n"


def places() -> list[dict]:
    """Named places in the dimos places format: a point at every work spot, plus the arm cell as a stay-out."""
    out: list[dict] = []
    for loc in LOCATIONS:
        x, y, _ = loc.stand_point(*loc.center) if loc.kind not in ("dock",) else (*loc.center, 0.0)
        out.append({"id": loc.id, "name": loc.name, "kind": "point", "x": round(x, 3), "y": round(y, 3)})
    x0, y0, x1, y1 = LAYOUT["markings"]["arm_cell_keepout"]
    out.append({"id": "arm-cell", "name": "Arm cell", "kind": "stay",
                "points": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}]})
    return out


def cook(package_dir: Path = PACKAGE_DIR) -> Path:
    """Write warehouse.xml, places.json and scene.meta.json. Returns the metadata path."""
    from dimos.simulation.scene_assets.spec import SceneMeshAlignment, ScenePackage

    package_dir.mkdir(parents=True, exist_ok=True)
    xml_path = package_dir / SCENE_XML.name
    xml_path.write_text(scene_xml())
    (package_dir / PLACES_JSON.name).write_text(json.dumps({"places": places()}, indent=2) + "\n")
    W, D = CELL_SIZE
    package = ScenePackage(
        package_dir=package_dir,
        source_path=Path("dimos_worlds/warehouse/layout.json"),
        alignment=SceneMeshAlignment(y_up=False),
        mujoco_scene_path=xml_path,
        metadata_path=package_dir / META_JSON.name,
        stats={
            "name": "warehouse",
            "extent_m": [W, D],
            "places": len(LOCATIONS),
            "places_file": PLACES_JSON.name,
            "cameras": sorted(LAYOUT["cameras"]),
            "totes": len(initial_poses()),
            "racks": sorted(LAYOUT["shelves"]),
        },
    )
    return package.write_metadata()


def main() -> None:
    print(f"wrote {cook()}")


if __name__ == "__main__":
    main()
