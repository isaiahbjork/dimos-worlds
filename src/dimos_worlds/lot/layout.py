"""The lot, as data. One source of truth for the MuJoCo collision model (gen_lot.py -> lot.xml)
and the optional Blender render scene (render/build_scene.py -> assets/cache/lot.blend).

Pure Python, no dependencies: Blender's bundled Python imports it too.

Coordinates: metres, z up, x east, y north, origin at the lot centre. The fenced lot spans
x in [-40, 40], y in [-25, 25]. The street runs along the south fence. Headings (`yaw`) are
degrees counter-clockwise about +z, applied to a model whose nose points +y (north).

Real-world scale references (typical values, not a survey of a specific site):
  * parking stall stripes 5.5 m long; stalls here are 3.6 m wide (a vehicle storage lot, spaced for doors)
  * chain-link fence 2.4 m (8 ft) fabric + 3 strands of barbed wire on 45 degree arms
  * lot light poles 7.6 m (25 ft) to the fixture, LED shoebox heads, one older HPS head left
  * cameras 4.2 to 5.0 m up on poles, 1/2.8" sensors with 2.8 mm or 4 mm lenses (camera_model.py)
"""

from __future__ import annotations

import math
import random

HX, HY = 40.0, 25.0  # half extents of the fenced lot
FENCE_H = 2.44  # fabric height (8 ft)
BARB_H = 0.35  # barbed-wire arm rise above the fabric

# ---- Rows and spots (ids are part of the events API: spot-1 .. spot-24) ------

ROW_Y = {"row-a": 12.0, "row-b": 4.0, "row-c": -4.0, "row-d": -12.0}
ROW_NOSE_YAW = {"row-a": 180.0, "row-b": 0.0, "row-c": 180.0, "row-d": 0.0}  # noses face the aisle
STALL_X = [-9.0, -5.4, -1.8, 1.8, 5.4, 9.0]
STALL_PITCH = 3.6
STALL_DEPTH = 5.5
SPOTS: dict[int, tuple[str, float, float, float]] = {}  # spot -> (row, x, y, yaw)
for _r, (_row, _y) in enumerate(ROW_Y.items()):
    for _i, _x in enumerate(STALL_X):
        SPOTS[_r * 6 + _i + 1] = (_row, _x, _y, ROW_NOSE_YAW[_row])
EMPTY_SPOTS = {16}  # baseline: spot 16 is open, so vehicle_moved:row-c:14:16 has somewhere to go

# ---- Vehicles ---------------------------------------------------------------
# kind -> (length, width, height) in metres. Typical class sizes (e.g. a full-size crew-cab pickup is
# about 5.9 x 2.0 x 1.95 m; a mid-size sedan 4.85 x 1.85 x 1.45 m).
VEHICLE_DIMS = {
    "sedan": (4.85, 1.85, 1.45),
    "hatch": (4.35, 1.80, 1.47),
    "suv": (4.95, 1.98, 1.80),
    "suv_compact": (4.55, 1.85, 1.66),
    "pickup": (5.90, 2.03, 1.95),
    "minivan": (5.15, 2.00, 1.78),
    "box_truck": (7.60, 2.45, 3.40),
    "skid_steer": (3.45, 1.83, 2.07),
    "trailer": (4.90, 2.05, 0.75),
    "covered_car": (4.40, 1.80, 1.30),  # Poly Haven covered_car, measured
}

# Paint colours (linear-ish sRGB base colours) with rough US fleet shares (white, black, grey and
# silver dominate). Shares are a design choice for variety, UNVERIFIED as statistics.
PAINT = {
    "white": ((0.80, 0.80, 0.78), 25),
    "black": ((0.012, 0.012, 0.014), 18),
    "grey": ((0.16, 0.17, 0.18), 17),
    "silver": ((0.45, 0.47, 0.49), 12),
    "blue": ((0.03, 0.07, 0.20), 9),
    "red": ((0.35, 0.02, 0.02), 9),
    "brown": ((0.17, 0.11, 0.07), 4),
    "green": ((0.04, 0.10, 0.06), 3),
    "tan": ((0.42, 0.36, 0.26), 3),
}


def _row_vehicles() -> list[dict]:
    rng = random.Random(20261007)
    names, weights = zip(*[(k, w) for k, (_, w) in PAINT.items()])
    out = []
    for spot in sorted(SPOTS):
        if spot in EMPTY_SPOTS:
            continue
        row, x, y, yaw = SPOTS[spot]
        if spot == 14:  # a white pickup; the vehicle_moved:row-c:14:16 example moves it
            kind, paint = "pickup", "white"
        else:
            kind = rng.choices(["pickup", "suv", "sedan", "suv_compact", "hatch", "minivan"],
                               weights=[6, 4, 3, 3, 2, 1])[0]
            paint = rng.choices(names, weights=weights)[0]
        out.append({"name": f"veh-{spot}", "kind": kind, "paint": paint, "x": x, "y": y, "yaw": yaw,
                    "spot": spot, "movable": True, "seed": spot})
    return out


# Vehicles outside the numbered rows: front display line, staff cars, service customers, equipment.
# Static in the events API (not movable), but still solid in the physics model.
EXTRA_VEHICLES = [
    # display line along the south fence, noses to the street
    {"name": "veh-x-disp1", "kind": "pickup", "paint": "red", "x": 15.0, "y": -20.6, "yaw": 180.0},
    {"name": "veh-x-disp2", "kind": "suv", "paint": "black", "x": 19.2, "y": -20.6, "yaw": 180.0},
    {"name": "veh-x-disp3", "kind": "pickup", "paint": "silver", "x": 23.4, "y": -20.6, "yaw": 180.0},
    {"name": "veh-x-disp4", "kind": "pickup", "paint": "blue", "x": 27.6, "y": -20.6, "yaw": 180.0},
    {"name": "veh-x-disp5", "kind": "suv_compact", "paint": "white", "x": 31.8, "y": -20.6, "yaw": 180.0},
    # staff cars in front of the office
    {"name": "veh-x-staff1", "kind": "sedan", "paint": "grey", "x": -36.4, "y": -16.9, "yaw": 90.0},
    {"name": "veh-x-staff2", "kind": "hatch", "paint": "blue", "x": -36.4, "y": -19.8, "yaw": 90.0},
    # service customers waiting south of the bay (the dark south-east corner)
    {"name": "veh-x-svc1", "kind": "minivan", "paint": "silver", "x": 28.0, "y": -15.2, "yaw": 90.0},
    {"name": "veh-x-svc2", "kind": "sedan", "paint": "white", "x": 34.5, "y": -15.2, "yaw": 90.0},
    # equipment and the box truck along the north fence
    {"name": "veh-x-skid", "kind": "skid_steer", "paint": "equip_yellow", "x": 5.0, "y": 19.8, "yaw": 180.0},
    {"name": "veh-x-trailer", "kind": "trailer", "paint": "grey", "x": 10.5, "y": 19.8, "yaw": 0.0},
    {"name": "veh-x-box", "kind": "box_truck", "paint": "white", "x": 27.0, "y": 20.2, "yaw": 90.0},
    {"name": "veh-x-covered", "kind": "covered_car", "paint": "white", "x": 30.5, "y": 5.0, "yaw": 0.0},
    {"name": "veh-x-pick6", "kind": "pickup", "paint": "black", "x": 17.2, "y": 19.6, "yaw": 0.0},
]
EQUIP_PAINT = {"equip_yellow": (0.62, 0.38, 0.02)}


def vehicles() -> list[dict]:
    extra = [dict(v, movable=False, spot=None, seed=1000 + i) for i, v in enumerate(EXTRA_VEHICLES)]
    return _row_vehicles() + extra


def paint_rgb(name: str) -> tuple[float, float, float]:
    if name in PAINT:
        return PAINT[name][0]
    return EQUIP_PAINT[name]


# ---- Fence and gates --------------------------------------------------------

# Gaps as (side, s0, s1), s measured along the side: south runs west->east (s = x + 40),
# west runs north->south (s = 25 - y).
FENCE_GAPS = {
    "south": [(30.0, 36.0)],  # front gate opening, x in [-10, -4]
    "west": [(33.0, 39.0)],  # gate 2 opening, y in [-14, -8]
}

GATES = {
    # Front gate: cantilever sliding gate. Closed it spans x -10..-4 on the fence line; it opens by sliding
    # 6.2 m west, inside the fence. Joint value = slide along +x in metres (range -6.2..0, closed = 0).
    "front-gate": {"type": "slide", "origin": (-10.0, -24.85), "length": 6.0, "axis": (1.0, 0.0),
                   "range": (-6.2, 0.0), "joint": "front-gate-slide"},
    # Gate 2: single swing leaf, hinge at the south post, leaf along +y; swings into the lot.
    # Joint value = hinge angle in degrees (range -85..0, closed = 0).
    "gate-2": {"type": "hinge", "origin": (-40.0, -14.0), "length": 6.0, "axis": (0.0, 1.0),
               "range": (-85.0, 0.0), "joint": "gate-2-hinge"},
}

# ---- Buildings and structures -----------------------------------------------

OFFICE = {"center": (-26.0, -16.7), "half": (7.0, 3.9), "height": 4.0, "parapet": 0.45,
          # north face (lot side) at y = -12.8; windows as (x centre, width); door at x = -21.6
          "windows": [(-31.2, 1.8), (-28.4, 1.8), (-25.6, 1.8), (-23.6, 1.2)], "lit_windows": [1, 3],
          "door_x": -21.6}
SERVICE_BAY = {"center": (29.0, -6.0), "half": (5.0, 5.0), "height": 5.6,
               # roll-up doors on the west face (x = 24) at these y centres, 3.0 m wide, 3.7 m tall
               "doors": [-8.4, -3.6], "door_w": 3.0, "door_h": 3.7}
FUEL_TANK = {"center": (-30.0, -1.6), "length": 3.2, "diameter": 1.4, "yaw": 0.0, "pad_half": (2.6, 2.0),
             "bollards": [(-32.8, -3.8), (-27.2, -3.8), (-32.8, 0.6), (-27.2, 0.6)]}
DUMPSTERS = [  # (x, y, yaw, colour) front-load 4 cubic yard bins, about 1.83 x 1.22 x 1.4 m
    (-36.4, -22.6, 0.0, (0.05, 0.16, 0.08)),
    (-34.2, -22.6, 0.0, (0.03, 0.06, 0.16)),
    (36.8, -21.6, 90.0, (0.05, 0.16, 0.08)),
]
TIRE_STACKS = [(35.6, -9.6), (36.3, -10.4)]
DRUMS = [(-33.6, 0.2), (-33.6, -0.6), (-34.3, -0.2)]

# ---- Lights -----------------------------------------------------------------
# Pole lights: (x, y, [head yaw degrees...], kind). Head yaw is the direction the head throws light.
# kind: "led" = 4000 K LED shoebox, about 150 W / 20 000 lm; "hps" = 250 W high-pressure sodium, about
# 2 100 K, 28 000 lm. Typical catalogue figures, UNVERIFIED for a specific product.
POLE_H = 7.6
POLE_LIGHTS = [
    (-14.0, 8.0, [90.0, 270.0], "led"),
    (14.0, 8.0, [90.0, 270.0], "led"),
    (-14.0, -8.0, [90.0, 270.0], "led"),
    (14.0, -8.0, [90.0, 270.0], "led"),
    (-4.0, 21.5, [270.0], "led"),  # north fence
    (18.0, 21.5, [270.0], "led"),
    (21.0, -16.5, [90.0, 0.0], "led"),
    (-2.0, -19.5, [90.0], "led"),  # south yard, lights rows C/D tails and the front gate approach
    (-38.0, -4.0, [285.0], "hps"),  # old sodium head on the west fence, aimed at gate 2
]
# Wall packs: (x, y, z, facing yaw)
WALL_PACKS = [
    (-21.6, -12.75, 3.1, 90.0),  # over the office door
    (-33.05, -16.0, 3.2, 180.0),  # office west wall
    (-33.05, -13.3, 3.4, 150.0),  # office west wall, north end, aimed at gate 2
    (23.95, -6.0, 4.9, 180.0),  # service bay, between the roll-up doors
    (29.0, -0.95, 4.9, 90.0),  # service bay north wall
    (34.05, -6.0, 4.9, 0.0),  # service bay east wall, lights the east fence lane
]
# Street lights outside the south fence: (x, y) cobra heads 9 m up, 3000 K, throw north over the road.
STREET_LIGHTS = [(-30.0, -36.5), (5.0, -36.5), (40.0, -36.5)]
POLE_SIGN = {"pos": (-16.5, -27.6), "height": 5.2, "size": (3.2, 1.2),
             "lines": ["VEHICLE STORAGE", "NO TRESPASSING"]}

# ---- Cameras ----------------------------------------------------------------
# name: (mount xyz, target xyz, lens mm). Mounts are camera poles (or the building) 4.2-5 m up.
# The mapping place -> camera lives in places.json. camera_model.py has the sensor and lens model.
CAMERAS = {
    "cam1": ((-17.0, -20.4, 4.4), (-8.0, -25.0, 0.6), 2.8),  # front gate, south fence
    "cam2": ((-28.0, 6.0, 5.0), (-32.0, -11.5, 0.4), 4.0),  # gate 2, fuel tank, office windows
    "cam3": ((22.0, 14.0, 5.0), (31.0, -8.0, 0.4), 4.0),  # service bay doors, east fence
    "cam4": ((12.0, -19.5, 5.0), (-0.9, -11.8, 0.3), 4.0),  # rows C and D, from the south-east
    "cam5": ((0.0, 21.8, 5.0), (0.0, 8.9, 0.3), 4.0),  # rows A and B, looking south down the x = 0 stall line
    "cam6": ((-24.0, 17.0, 4.2), (-6.0, 24.0, 0.6), 4.0),  # north fence
}

# ---- People -----------------------------------------------------------------
PERSON_PARK = (55.0, -45.0)  # outside the fence, out of every camera's view
PERSON_HEIGHT = 1.78
# Pose per place (pose name, heading yaw degrees: the way the person faces). Poses: walk, stand, crouch.
# Crouching where the person would plausibly be at a vehicle; walking along fences and gates.
PERSON_POSES = {
    "front-gate": ("walk", 0.0),  # walking in through the gate
    "gate-2": ("walk", 270.0),  # entering from the west, heading east
    "service-bay-doors": ("stand", 270.0),  # facing the bay doors (east)
    "row-a": ("stand", 90.0),  # between two vehicles, facing west
    "row-b": ("crouch", 90.0),  # at the driver side of the vehicle in spot 9
    "row-c": ("stand", 90.0),  # at the driver door of the vehicle in spot 15
    "row-d": ("walk", 90.0),  # walking west along the south drive lane
    "fuel-tank": ("crouch", 45.0),  # at the tank, facing north-west
    "office-windows": ("stand", 180.0),  # facing the office windows (south)
    "north-fence": ("walk", 270.0),  # walking east along the fence
    "east-fence": ("walk", 0.0),  # walking north along the fence
    "south-fence": ("walk", 90.0),  # walking west along the fence
}
# Heading convention: yaw rotates a model facing +y (north); 90 faces west, 180 south, 270 east.
# Top of head above the ground for each pose (for pixel-height checks).
POSE_TOP_M = {"walk": 1.76, "stand": 1.78, "crouch": 1.15}


def gate_leaf_xy(gid: str, value: float) -> list[tuple[float, float]]:
    """End points of a gate leaf for a joint value (slide metres or hinge degrees)."""
    g = GATES[gid]
    ox, oy = g["origin"]
    ax, ay = g["axis"]
    if g["type"] == "slide":
        x0, y0 = ox + ax * value, oy + ay * value
        return [(x0, y0), (x0 + ax * g["length"], y0 + ay * g["length"])]
    a = math.radians(value)
    dx, dy = ax * math.cos(a) - ay * math.sin(a), ax * math.sin(a) + ay * math.cos(a)
    return [(ox, oy), (ox + dx * g["length"], oy + dy * g["length"])]
