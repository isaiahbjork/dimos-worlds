"""Generate layout.json, the single source of truth for the warehouse scene.

    python -m dimos_worlds.warehouse.gen_layout      # rewrites src/dimos_worlds/warehouse/layout.json

Metres. Origin = south-west interior corner (floor level). x east, y north, z up. The building is 30 x 20 m.
Rectangles are [x0, y0, x1, y1]. Rack bins are centres of the pallet opening (x, y, z of the pallet deck top).
Dimensions are catalogue-style typical values (UNVERIFIED for any real site).

Who reads it:
  * cell.py   places, slots, walking routes (A* on the floor plan), reach checks
  * scene.py  the MuJoCo scene package (racks, decks, conveyor, table, pallets, pedestal, totes, cameras, lights)
  * render/   the optional Blender scene

The arm cell is sized for a KUKA LBR iiwa 14 R820 (reach 820 mm, KUKA data sheet; MuJoCo Menagerie model): the
pedestal is 0.50 m high; the conveyor end (0.55 m north), the pick table (0.62 m east) and the half pallet (0.56 m
south) are inside its reach with the tool pointing down (cell.arm_can_reach). Conveyor and table tops at 0.75 m are in
the usual 0.6-0.9 m working-height range (UNVERIFIED per site).
"""
import json
import math
from pathlib import Path

W, D, H = 30.0, 20.0, 9.0
BAY, DEPTH, N_BAYS, X0 = 2.7, 1.1, 7, 2.0
BEAM_Z = [0.25, 1.85, 3.45]  # top of the beam / pallet deck at each level
FRAME_H = 5.0
rows = {"A": 16.6, "B": 13.0, "C": 8.6, "D": 5.0}  # y0 of each row (depth DEPTH northwards)
PICK_SIDE = {"A": "south", "B": "north", "C": "south", "D": "north"}

TOTE = (0.30, 0.20, 0.12)  # x, y, z full extents (a small KLT-style tote)
# Pick bay: bay 1 of every row has two wire decks for totes instead of a ground-level pallet. Three totes per deck,
# at the front edge of the rack so a humanoid standing in the aisle reaches them.
PICK_BAY = 1
DECKS = (("u", 0.85), ("l", 0.40))
DECK_DX = (-0.75, 0.0, 0.75)
FRONT_INSET = 0.13  # tote centre behind the rack's aisle edge

# ---- the arm cell (see the docstring)
ARM = (26.0, 12.0)  # pedestal centre
PEDESTAL_H = 0.50
CONV_TOP = 0.75
TABLE_TOP = 0.75
PALLET2_TOP = 0.14  # half pallet 800 x 600 (deck top)
CONV_Y = ARM[1] + 0.55
CONV_X0 = 22.8
CONV_PICK_X = ARM[0]  # the tote at the end stops here, straight north of the base
CONV_PITCH = 0.40
CONV_W = 0.40
TABLE = (ARM[0] + 0.40, ARM[1] - 0.40, ARM[0] + 0.85, ARM[1] + 0.40)
TABLE_SLOT_X = ARM[0] + 0.62
P2 = (ARM[0], ARM[1] - 0.56)  # half pallet centre

HUMANOID_HOME = (1.0, 11.0, 0.0)  # in the charging spot, facing east
QUADRUPED_HOME = (3.0, 11.4, 0.0)  # the cross aisle, west end, facing east


def r3(v):
    return [round(float(a), 3) for a in v]


def build() -> dict:
    shelves = {}
    places = []
    for sid, y0 in rows.items():
        bins = []
        for bay in range(N_BAYS):
            for lvl in range(len(BEAM_Z)):
                if bay + 1 == PICK_BAY and lvl == 0:
                    continue  # the pick bay's ground level holds the tote decks
                bins.append({"id": f"{sid}-{bay + 1:02d}-{lvl + 1}", "bay": bay + 1, "level": lvl + 1,
                             "center": [round(X0 + BAY * (bay + 0.5), 3), round(y0 + DEPTH / 2, 3), BEAM_Z[lvl]],
                             "size": [BAY - 0.2, DEPTH - 0.1, 1.35]})
        south = PICK_SIDE[sid] == "south"
        cx = round(X0 + BAY * (PICK_BAY - 0.5), 3)
        cy = round(y0 + DEPTH / 2, 3)
        front_y = y0 + FRONT_INSET if south else y0 + DEPTH - FRONT_INSET
        pid = f"shelf-{sid.lower()}"
        shelves[sid] = {"rect": [X0, y0, round(X0 + BAY * N_BAYS, 3), round(y0 + DEPTH, 3)], "bay_length": BAY,
                        "depth": DEPTH, "frame_height": FRAME_H, "beam_z": BEAM_Z, "pick_side": PICK_SIDE[sid],
                        "pick_bay": {"bay": PICK_BAY, "place_id": pid, "deck_z": [z for _, z in DECKS]},
                        "bins": bins}
        places.append({
            "id": pid, "name": f"Shelf {sid}", "kind": "shelf", "center": [cx, cy], "size": [BAY - 0.2, DEPTH],
            "top_z": min(z for _, z in DECKS), "approach_yaw_deg": 90 if south else -90,
            "slots": [{"id": f"{pid}-{lv}{i + 1}", "xyz": r3((cx + dx, front_y, z))}
                      for lv, z in DECKS for i, dx in enumerate(DECK_DX)],
        })

    places += [
        {"id": "floor", "name": "Floor by the shelves", "kind": "floor", "center": [5.5, 15.35], "size": [7.0, 2.5],
         "top_z": 0.0, "approach_yaw_deg": 90, "slots": []},
        {"id": "conveyor", "name": "Conveyor", "kind": "conveyor",
         "center": r3(((CONV_X0 + CONV_PICK_X + 0.2) / 2, CONV_Y)), "size": r3((CONV_PICK_X + 0.2 - CONV_X0, CONV_W)),
         "top_z": CONV_TOP, "approach_yaw_deg": -90,
         "slots": [{"id": f"conveyor-{i + 1}", "xyz": r3((CONV_PICK_X - i * CONV_PITCH, CONV_Y, CONV_TOP))}
                   for i in range(8)]},
        {"id": "pick-table", "name": "Pick table", "kind": "table",
         "center": r3(((TABLE[0] + TABLE[2]) / 2, (TABLE[1] + TABLE[3]) / 2)),
         "size": r3((TABLE[2] - TABLE[0], TABLE[3] - TABLE[1])), "top_z": TABLE_TOP, "approach_yaw_deg": 180,
         "slots": [{"id": f"pick-table-{i + 1}", "xyz": r3((TABLE_SLOT_X, ARM[1] + dy, TABLE_TOP))}
                   for i, dy in enumerate((-0.16, 0.16))]},
        {"id": "pallet-1", "name": "Pallet 1", "kind": "pallet", "center": [23.4, 3.6], "size": [1.2, 1.0],
         "top_z": 0.144, "approach_yaw_deg": -90,  # outbound staging by the dock; front row reachable from the aisle
         "slots": [{"id": f"pallet-1-a{i + 1}", "xyz": r3((23.4 + dx, 3.6 + 0.30, 0.144))}
                   for i, dx in enumerate((-0.35, 0.0, 0.35))]},
        {"id": "pallet-2", "name": "Pallet 2", "kind": "pallet", "center": r3(P2), "size": [0.8, 0.6],
         "top_z": PALLET2_TOP, "approach_yaw_deg": 90,  # the arm's half pallet, two layers of four
         "slots": [{"id": f"pallet-2-{layer}{i + 1}", "xyz": r3((P2[0] + dx, P2[1] + dy, PALLET2_TOP + li * TOTE[2]))}
                   for li, layer in enumerate("ab")
                   for i, (dx, dy) in enumerate(((-0.16, 0.13), (0.16, 0.13), (-0.16, -0.13), (0.16, -0.13)))]},
        {"id": "charging-spot", "name": "Charging spot", "kind": "dock", "center": [1.0, 11.0], "size": [1.2, 1.2],
         "top_z": 0.0, "approach_yaw_deg": 0, "slots": []},
    ]

    # Where the totes start: a slot id, or [x, y, z of the tote's base, yaw] for loose ones.
    totes_initial = [
        ["tote-01", "shelf-a-u1"], ["tote-02", "shelf-a-u3"],
        ["tote-03", "shelf-b-u2"],
        ["tote-04", "shelf-c-u1"], ["tote-05", "shelf-c-u2"], ["tote-06", "shelf-c-l3"],
        ["tote-07", [4.6, 15.1, 0.0, 0.3]], ["tote-08", [6.4, 15.5, 0.0, -0.4]],  # left on the floor in aisle A-B
        *[[f"tote-{9 + i:02d}", f"conveyor-{i + 1}"] for i in range(7)],
        ["tote-16", "shelf-d-l1"], ["tote-17", "shelf-d-l2"],
    ]

    keepout = r3((ARM[0] - 0.95, ARM[1] - 0.95, ARM[0] + 0.95, ARM[1] + 0.95))
    return {
        "version": 4,
        "units": "m",
        "frame": "origin = SW interior corner; x east, y north, z up",
        "building": {"size": [W, D, H], "wall_thickness": 0.2, "eave_height": 8.5,
                     "floor": "polished concrete slab, painted markings"},
        "shelves": shelves,
        "places": places,
        "totes": {"size": list(TOTE), "mass_kg": 0.8, "initial": totes_initial},
        "pallets": [  # every pallet on the floor; `place` = the place it is (else decoration with a load)
            {"id": "pallet-1", "place": "pallet-1", "center": [23.4, 3.6], "size": [1.2, 1.0, 0.144], "yaw_deg": 0,
             "load_height": 0.0},
            {"id": "pallet-2", "place": "pallet-2", "center": r3(P2), "size": [0.8, 0.6, PALLET2_TOP], "yaw_deg": 0,
             "load_height": 0.0},
            {"id": "pallet-3", "place": None, "center": [29.2, 3.6], "size": [1.2, 1.0, 0.144], "yaw_deg": 0,
             "load_height": 1.3},
            {"id": "pallet-4", "place": None, "center": [27.4, 3.6], "size": [1.2, 1.0, 0.144], "yaw_deg": 0,
             "load_height": 1.0},
        ],
        "conveyor": {"id": "conveyor",
                     "rect": r3((CONV_X0, CONV_Y - CONV_W / 2, CONV_PICK_X + 0.2, CONV_Y + CONV_W / 2)),
                     "top_z": CONV_TOP, "flow": "+x", "pick_x": CONV_PICK_X, "pitch": CONV_PITCH, "speed_ms": 0.25,
                     "type": "roller, side rails, end stop at pick_x"},
        "pick_table": {"id": "pick-table", "rect": r3(TABLE), "top_z": TABLE_TOP},
        "arm_cell": {"pedestal_center": list(ARM), "pedestal_size": [0.4, 0.4], "pedestal_height": PEDESTAL_H,
                     "robot": "kuka_iiwa_14 (MuJoCo Menagerie, BSD-3-Clause)", "base_yaw_deg": 0,
                     "reach_m": 0.80, "tool_m": 0.12, "wrist_to_tip_m": 0.246, "clearance_m": 0.15,
                     "transit_offset": [0.40, 0.0, 0.62],
                     "reach_targets": {"conveyor": r3((CONV_PICK_X, CONV_Y, CONV_TOP + TOTE[2])),
                                       "pick_table": r3((TABLE_SLOT_X, ARM[1], TABLE_TOP + TOTE[2])),
                                       "pallet-2": r3((P2[0], P2[1], PALLET2_TOP + TOTE[2]))}},
        "humanoid": {"robot": "unitree_g1 (MuJoCo Menagerie, BSD-3-Clause)", "home": list(HUMANOID_HOME),
                     "edge_gap_m": 0.18, "floor_standoff_m": 0.30, "reach_m": 0.62, "reach_z": [0.0, 1.05],
                     "body_radius_m": 0.35},
        "quadruped": {"robot": "unitree_go1 body (DimOS's Go2 sim body)", "home": list(QUADRUPED_HOME),
                      "body_radius_m": 0.35},
        "charging": {"id": "charging-spot", "rect": [0.4, 10.4, 1.6, 11.6], "center": [1.0, 11.0], "dock_wall": "west"},
        "aisles": [
            {"id": "aisle-AB", "rect": [X0, 14.1, 20.9, 16.6], "width": 2.5},
            {"id": "aisle-CD", "rect": [X0, 6.1, 20.9, 8.6], "width": 2.5},
            {"id": "aisle-cross", "rect": [X0, 9.7, 22.5, 13.0], "width": 3.3},
            {"id": "aisle-north", "rect": [X0, 17.7, 29.4, 19.6], "width": 1.9},
            {"id": "aisle-south", "rect": [X0, 0.6, 21.5, 5.0], "width": 4.4},
            {"id": "aisle-east", "rect": [20.9, 0.6, 22.5, 19.6], "width": 1.6},
            {"id": "aisle-west", "rect": [0.2, 0.6, X0, 19.6], "width": 1.8},
        ],
        # Bollards guarding the roll-up door's frame.
        "bollards": [{"id": f"bollard-{i + 1}", "x": x, "y": 0.9, "r": 0.1, "height": 1.0}
                     for i, x in enumerate((24.0 - 0.6, 28.5 + 0.6))],
        "doors": [
            {"id": "rollup-1", "type": "roll_up", "wall": "south", "rect": [24.0, 0.0, 28.5, 0.0], "height": 4.5,
             "open_fraction": {"day": 0.7, "night": 0.0}},
            {"id": "door-personnel", "type": "personnel", "wall": "west", "rect": [0.0, 18.0, 0.0, 18.9],
             "height": 2.1},
            {"id": "door-exit", "type": "emergency_exit", "wall": "east", "rect": [30.0, 6.0, 30.0, 6.9],
             "height": 2.1},
        ],
        "cameras": {  # mount [x, y, z], target [x, y, z], lens mm and its horizontal field of view, places in view
            "cam1": {"name": "Cam 1", "mount": [0.35, 0.35, 6.0], "target": [14.0, 8.0, 0.8], "lens_mm": 2.8,
                     "hfov_deg": 100, "note": "SW corner, whole floor",
                     "places": ["shelf-c", "shelf-d", "charging-spot"]},
            "cam2": {"name": "Cam 2", "mount": [29.65, 19.65, 6.0], "target": [16.0, 8.0, 0.8], "lens_mm": 2.8,
                     "hfov_deg": 100, "note": "NE corner, racks, cross aisle, conveyor",
                     "places": ["shelf-b", "conveyor", "pick-table"]},
            "cam3": {"name": "Cam 3", "mount": [0.35, 19.65, 6.0], "target": [8.0, 13.0, 0.8], "lens_mm": 2.8,
                     "hfov_deg": 100, "note": "NW corner, aisles A-B and C-D",
                     "places": ["shelf-a", "shelf-b", "floor"]},
            "cam4": {"name": "Cam 4", "mount": [29.7, 9.5, 4.5], "target": [26.0, 12.0, 0.8], "lens_mm": 4.0,
                     "hfov_deg": 80, "note": "east wall: arm cell, conveyor end, pick table, pallet 2",
                     "places": ["conveyor", "pick-table", "pallet-2"]},
            "cam5": {"name": "Cam 5", "mount": [29.65, 0.35, 5.0], "target": [24.0, 4.5, 0.6], "lens_mm": 2.8,
                     "hfov_deg": 100, "note": "SE corner, the dock and outbound pallets", "places": ["pallet-1"]},
            "cam6": {"name": "Cam 6", "mount": [0.3, 15.35, 3.5], "target": [10.0, 15.35, 0.5], "lens_mm": 2.8,
                     "hfov_deg": 100, "note": "west wall, down aisle A-B: both pick faces and the floor",
                     "places": ["shelf-a", "shelf-b", "floor"]},
        },
        "lights": {"high_bay_grid": {"x": [4.5, 10.0, 15.5, 21.0, 26.5], "y": [3.5, 10.5, 16.5], "z": 7.6},
                   "modes": {"day": "all on, 100 %, daylight through the open roll-up door",
                             "night": "every other fixture at 8 %, exit signs on"}},
        "markings": {"aisle_edge": "yellow, 100 mm", "pedestrian_walkway": "green", "pallet_bays": "white boxes",
                     "arm_cell_keepout": keepout,
                     "arm_cell_note": "iiwa 14 is a collaborative arm; the hatched band marks its reach"},
    }


def main() -> None:
    out = Path(__file__).with_name("layout.json")
    out.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
