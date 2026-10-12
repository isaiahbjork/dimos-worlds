"""Generate layout.json: the Demo plant (process area for scheduled inspection rounds), one file for every reader.

    python3 gen_layout.py            # writes layout.json next to this file (stdlib only)

Readers: gen_mjcf.py (MuJoCo world), go2_scene.py (the Go2's prior map and places), render/build_scene.py (Blender
scene), plant_state.py (instrument truth and events), gauge_loop.py (where the dog stands for each gauge).

Frame: metres, origin at the SW corner of the concrete slab, x east, y north, z up. Slab 56 x 36 m.

What is on it (all dimensions are typical catalogue-style figures, UNVERIFIED for any real plant):
  * control room (12 x 8 m, 3.6 m high) with a door opening on the south wall and the instrument-air panel inside
  * pump bay: three end-suction pumps P-101..P-103 with motors M-101..M-103 on plinths, suction header behind them,
    discharge risers up to the pipe rack, each with a discharge pressure gauge, a gate valve and a drip tray
  * pipe rack: 34 m east-west, two tiers at 4.5 and 5.1 m, columns every 6 m; stair tower at its east end (stay out)
  * tank farm: two vertical tanks T-201, T-202 in a 1.0 m concrete bund; outlet line with temperature gauge and gate
    valve (T-201), drain line with a ball valve (T-202); a stile over the bund wall (stairs: stay out)
  * air compressor C-301 package: motor M-301, compressor block, air receiver with pressure gauge PG-301 and an
    isolation valve
  * painted walkways joining all of it; the Go2 stays on flat ground and never takes stairs
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "layout.json"
VERSION = 1

SLAB = (0.0, 0.0, 56.0, 36.0)
SITE = {"id": "plant", "name": "Demo plant", "place": "Sim", "tz": "America/Phoenix"}

solids: list[dict] = []
pipes: list[dict] = []
instruments: list[dict] = []
places: list[dict] = []


def box(sid: str, center: tuple[float, float, float], size: tuple[float, float, float], *, group: str,
        material: str, collide: bool = True, **extra) -> dict:
    """Axis-aligned box; center (x, y, z), full size (sx, sy, sz)."""
    s = {"id": sid, "shape": "box", "center": [round(v, 3) for v in center], "size": [round(v, 3) for v in size],
         "group": group, "material": material, "collide": collide, **extra}
    solids.append(s)
    return s


def cyl(sid: str, a: tuple[float, float, float], b: tuple[float, float, float], r: float, *, group: str,
        material: str, collide: bool = True, **extra) -> dict:
    """Cylinder from point a to point b, radius r."""
    s = {"id": sid, "shape": "cyl", "from": [round(v, 3) for v in a], "to": [round(v, 3) for v in b],
         "radius": r, "group": group, "material": material, "collide": collide, **extra}
    solids.append(s)
    return s


def pipe(pid: str, points: list[tuple[float, float, float]], r: float, *, material: str = "pipe_grey",
         collide: bool | None = None) -> None:
    """A pipe run (polyline). Collides when any part is below 3 m (overhead pipes do not block a dog)."""
    low = min(p[2] for p in points) - r < 3.0
    pipes.append({"id": pid, "points": [[round(v, 3) for v in p] for p in points], "radius": r,
                  "material": material, "collide": low if collide is None else collide})


def stand_for(target: tuple[float, float, float], facing: tuple[float, float], dist: float) -> list[float]:
    """Where the dog stands to look at `target` whose face points along `facing` (unit xy): dist m out, facing back."""
    fx, fy = facing
    x, y = target[0] + fx * dist, target[1] + fy * dist
    yaw = math.degrees(math.atan2(-fy, -fx))
    return [round(x, 3), round(y, 3), round(yaw, 1)]


# ---------------------------------------------------------------- ground and walkways
x0, y0, x1, y1 = SLAB
walkways = [
    {"id": "walk-south", "name": "South walkway", "poly": [[3, 10.5], [55, 10.5], [55, 12.5], [3, 12.5]]},
    {"id": "walk-west", "name": "West walkway", "poly": [[15, 3], [17, 3], [17, 24.6], [15, 24.6]]},
    {"id": "walk-north", "name": "North walkway", "poly": [[3, 22.6], [55, 22.6], [55, 24.6], [3, 24.6]]},
    {"id": "walk-east", "name": "East walkway", "poly": [[53, 10.5], [55, 10.5], [55, 24.6], [53, 24.6]]},
    {"id": "walk-cr", "name": "Control room path", "poly": [[7.2, 24.6], [8.8, 24.6], [8.8, 26.0], [7.2, 26.0]]},
    {"id": "walk-comp", "name": "Compressor spur", "poly": [[43.5, 4.2], [51.5, 4.2], [51.5, 10.5], [43.5, 10.5]]},
    {"id": "walk-pumps", "name": "Pump bay apron", "poly": [[19, 12.5], [34, 12.5], [34, 13.6], [19, 13.6]]},
]

# ---------------------------------------------------------------- control room (indoor checkpoint)
CR = (2.0, 26.0, 14.0, 34.0)
CR_H = 3.6
WALL_T = 0.25
DOOR_X = (7.3, 8.7)
cx0, cy0, cx1, cy1 = CR
box("cr-wall-n", ((cx0 + cx1) / 2, cy1 - WALL_T / 2, CR_H / 2), (cx1 - cx0, WALL_T, CR_H), group="building",
    material="cr_wall")
box("cr-wall-w", (cx0 + WALL_T / 2, (cy0 + cy1) / 2, CR_H / 2), (WALL_T, cy1 - cy0, CR_H), group="building",
    material="cr_wall")
box("cr-wall-e", (cx1 - WALL_T / 2, (cy0 + cy1) / 2, CR_H / 2), (WALL_T, cy1 - cy0, CR_H), group="building",
    material="cr_wall")
box("cr-wall-s-w", ((cx0 + DOOR_X[0]) / 2, cy0 + WALL_T / 2, CR_H / 2), (DOOR_X[0] - cx0, WALL_T, CR_H),
    group="building", material="cr_wall")
box("cr-wall-s-e", ((DOOR_X[1] + cx1) / 2, cy0 + WALL_T / 2, CR_H / 2), (cx1 - DOOR_X[1], WALL_T, CR_H),
    group="building", material="cr_wall")
box("cr-lintel", ((DOOR_X[0] + DOOR_X[1]) / 2, cy0 + WALL_T / 2, 2.4 + (CR_H - 2.4) / 2),
    (DOOR_X[1] - DOOR_X[0], WALL_T, CR_H - 2.4), group="building", material="cr_wall")
box("cr-roof", ((cx0 + cx1) / 2, (cy0 + cy1) / 2, CR_H + 0.1), (cx1 - cx0 + 0.4, cy1 - cy0 + 0.4, 0.2),
    group="roof", material="roof_metal", collide=False)
# inside: instrument-air panel on the north wall, operator desk with consoles, a cabinet row on the east wall
box("cr-panel", (8.0, cy1 - WALL_T - 0.25, 1.0), (2.4, 0.5, 2.0), group="equipment", material="panel_grey")
box("cr-desk", (5.0, 29.5, 0.38), (3.2, 1.0, 0.76), group="equipment", material="desk")
box("cr-console", (5.0, 29.8, 0.98), (2.6, 0.3, 0.45), group="equipment", material="console")
box("cr-cabinets", (cx1 - WALL_T - 0.35, 30.0, 1.0), (0.6, 4.0, 2.0), group="equipment", material="panel_grey")

# ---------------------------------------------------------------- pump bay
PUMP_X = {"101": 22.0, "102": 27.0, "103": 32.0}
PUMP_Y = 14.5
pump_stations = {}
for n, px in PUMP_X.items():
    box(f"plinth-{n}", (px, PUMP_Y, 0.1), (2.6, 1.0, 0.2), group="equipment", material="concrete_plinth")
    box(f"baseplate-{n}", (px, PUMP_Y, 0.25), (2.2, 0.6, 0.1), group="equipment", material="steel_dark")
    cyl(f"M-{n}", (px - 1.15, PUMP_Y, 0.58), (px - 0.35, PUMP_Y, 0.58), 0.27, group="equipment",
        material="motor_blue", thermal="motor", tag=f"M-{n}")
    cyl(f"M-{n}-fan", (px - 1.25, PUMP_Y, 0.58), (px - 1.15, PUMP_Y, 0.58), 0.25, group="equipment",
        material="steel_dark")
    box(f"coupling-{n}", (px - 0.15, PUMP_Y, 0.58), (0.4, 0.3, 0.3), group="equipment", material="guard_yellow")
    cyl(f"P-{n}", (px + 0.2, PUMP_Y - 0.17, 0.55), (px + 0.2, PUMP_Y + 0.17, 0.55), 0.3, group="equipment",
        material="pump_blue", thermal="pump", tag=f"P-{n}")
    # suction from the header (north), discharge riser up to the rack
    pipe(f"suction-{n}", [(px + 0.2, PUMP_Y + 0.17, 0.55), (px + 0.2, 17.0, 0.55)], 0.08)
    riser_x = px + 0.55
    pipe(f"discharge-{n}", [(px + 0.2, PUMP_Y - 0.1, 0.85), (riser_x, PUMP_Y - 0.1, 0.85),
                            (riser_x, PUMP_Y - 0.1, 3.6), (riser_x, 20.0, 3.6), (riser_x, 20.0, 4.6)], 0.065)
    gauge_pos = (riser_x, PUMP_Y - 0.1 - 0.065 - 0.11, 1.15)
    instruments.append({
        "tag": f"PG-{n}", "kind": "gauge", "place": f"pump-{n}", "measure": "pressure", "unit": "bar",
        "label": f"P-{n} discharge pressure", "range": [0, 10], "normal": [4.0, 7.0], "nominal": 5.6,
        "dial_m": 0.16, "pos": [round(v, 3) for v in gauge_pos], "facing": [0, -1],
        "stand": stand_for(gauge_pos, (0, -1), 1.05),
    })
    valve_pos = (riser_x, PUMP_Y - 0.1 - 0.065, 1.75)
    instruments.append({
        "tag": f"XV-{n}", "kind": "valve", "style": "gate", "place": f"pump-{n}", "label": f"P-{n} discharge valve",
        "normal_state": "open", "pos": [round(v, 3) for v in valve_pos], "facing": [0, -1],
        "stand": stand_for(valve_pos, (0, -1), 1.35),
    })
    motor_aim = (px - 0.75, PUMP_Y - 0.27, 0.58)
    instruments.append({
        "tag": f"M-{n}", "kind": "motor", "place": f"pump-{n}", "label": f"P-{n} motor",
        "nominal_c": 52.0, "warn_c": 75.0, "alarm_c": 85.0, "pos": [round(v, 3) for v in motor_aim],
        "facing": [0, -1], "stand": stand_for(motor_aim, (0, -1), 1.45),
    })
    tray_pos = (px, PUMP_Y - 0.62, 0.0)
    instruments.append({
        "tag": f"DT-{n}", "kind": "drip_tray", "place": f"pump-{n}", "label": f"P-{n} seal and drip tray",
        "pos": [round(v, 3) for v in tray_pos], "facing": [0, -1],
        "stand": stand_for((px, PUMP_Y - 0.5, 0.0), (0, -1), 1.5),
    })
    pump_stations[n] = px

# suction header on low supports behind the pumps; it goes underground at both ends
pipe("suction-header", [(19.5, 17.0, -0.3), (19.5, 17.0, 0.55), (46.0, 17.0, 0.55), (46.0, 17.0, -0.3)], 0.11)
for i, sx in enumerate(range(20, 47, 3)):
    box(f"header-support-{i}", (sx + 0.5, 17.0, 0.21), (0.3, 0.5, 0.42), group="equipment", material="concrete_plinth")

# pump shelter roof (open sides) on four columns
for i, (sx, sy) in enumerate(((19.4, 13.2), (34.6, 13.2), (19.4, 16.2), (34.6, 16.2))):
    box(f"shelter-col-{i}", (sx, sy, 2.0), (0.2, 0.2, 4.0), group="structure", material="steel_painted")
box("shelter-roof", (27.0, 14.7, 4.05), (15.8, 3.6, 0.1), group="roof", material="roof_metal", collide=False)

# ---------------------------------------------------------------- pipe rack
RACK_X = [18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 52.0]
RACK_Y = (19.2, 20.8)
for i, rx in enumerate(RACK_X):
    for j, ry in enumerate(RACK_Y):
        box(f"rack-col-{i}-{j}", (rx, ry, 2.6), (0.3, 0.3, 5.2), group="structure", material="steel_rack")
    box(f"rack-beam-lo-{i}", (rx, 20.0, 4.45), (0.25, 1.9, 0.2), group="structure", material="steel_rack",
        collide=False)
    box(f"rack-beam-hi-{i}", (rx, 20.0, 5.05), (0.25, 1.9, 0.2), group="structure", material="steel_rack",
        collide=False)
for k, (dy, r, mat) in enumerate(((-0.65, 0.16, "pipe_grey"), (-0.25, 0.11, "pipe_green"), (0.15, 0.13, "pipe_grey"),
                                  (0.55, 0.08, "pipe_yellow"))):
    pipe(f"rack-lo-{k}", [(17.0, 20.0 + dy, 4.55 + r), (53.0, 20.0 + dy, 4.55 + r)], r, material=mat)
for k, (dy, r, mat) in enumerate(((-0.5, 0.09, "pipe_blue"), (0.0, 0.14, "pipe_grey"), (0.5, 0.06, "pipe_red"))):
    pipe(f"rack-hi-{k}", [(17.0, 20.0 + dy, 5.15 + r), (53.0, 20.0 + dy, 5.15 + r)], r, material=mat)
box("rack-tray", (35.0, 20.75, 5.6), (36.0, 0.4, 0.08), group="structure", material="steel_galv", collide=False)
# stair tower to the rack walkway at the east end: STAIRS, the dog does not go here
STAIR_RACK = (49.6, 17.6, 51.4, 19.0)
for s in range(9):
    h = 0.2 * (s + 1)
    box(f"rack-stair-{s}", (STAIR_RACK[0] + 0.9, STAIR_RACK[1] + 0.15 + s * 0.15, h / 2), (1.8, 0.3, h),
        group="stairs", material="grating")
box("rack-landing", (50.5, 19.4, 1.85), (1.8, 0.8, 0.1), group="stairs", material="grating", collide=False)
box("rack-stair-rail-w", (STAIR_RACK[0], 18.3, 1.0), (0.06, 1.6, 2.0), group="stairs", material="guard_yellow")
box("rack-stair-rail-e", (STAIR_RACK[2], 18.3, 1.0), (0.06, 1.6, 2.0), group="stairs", material="guard_yellow")

# ---------------------------------------------------------------- tank farm
BUND = (35.0, 26.0, 53.0, 35.2)
BUND_H, BUND_T = 1.0, 0.3
bx0, by0, bx1, by1 = BUND
box("bund-s-w", ((bx0 + 43.3) / 2, by0 + BUND_T / 2, BUND_H / 2), (43.3 - bx0, BUND_T, BUND_H), group="bund",
    material="concrete_wall")
box("bund-s-e", ((44.7 + bx1) / 2, by0 + BUND_T / 2, BUND_H / 2), (bx1 - 44.7, BUND_T, BUND_H), group="bund",
    material="concrete_wall")
box("bund-s-stile", (44.0, by0 + BUND_T / 2, BUND_H / 2), (1.4, BUND_T, BUND_H), group="bund",
    material="concrete_wall")
box("bund-n", ((bx0 + bx1) / 2, by1 - BUND_T / 2, BUND_H / 2), (bx1 - bx0, BUND_T, BUND_H), group="bund",
    material="concrete_wall")
box("bund-w", (bx0 + BUND_T / 2, (by0 + by1) / 2, BUND_H / 2), (BUND_T, by1 - by0, BUND_H), group="bund",
    material="concrete_wall")
box("bund-e", (bx1 - BUND_T / 2, (by0 + by1) / 2, BUND_H / 2), (BUND_T, by1 - by0, BUND_H), group="bund",
    material="concrete_wall")
# stile: steps up and over the bund wall (STAIRS: stay out)
for s in range(4):
    h = 0.25 * (s + 1)
    box(f"stile-step-{s}", (44.0, by0 - 0.95 + s * 0.25, h / 2), (1.2, 0.25, h), group="stairs", material="grating")
TANKS = {"201": (40.0, 30.6), "202": (48.0, 30.6)}
TANK_R, TANK_H = 3.2, 9.0
for n, (tx, ty) in TANKS.items():
    cyl(f"T-{n}", (tx, ty, 0.0), (tx, ty, TANK_H), TANK_R, group="tank", material="tank_green", tag=f"T-{n}",
        thermal="tank")
    cyl(f"T-{n}-roof", (tx, ty, TANK_H), (tx, ty, TANK_H + 0.35), TANK_R * 0.98, group="tank", material="tank_roof",
        collide=False)
# T-201 outlet: south through the bund to a tee, gauge and valve outside the bund, then underground
pipe("t201-outlet", [(40.0, 30.6 - TANK_R, 0.6), (40.0, 24.95, 0.6), (40.0, 24.95, -0.3)], 0.08)
pipe("t201-tg-stub", [(40.0, 25.2, 0.6), (40.0, 25.2, 1.08)], 0.03, material="steel_galv")
tg201 = (40.0, 25.2 - 0.03 - 0.1, 1.16)
instruments.append({
    "tag": "TG-201", "kind": "gauge", "place": "tank-201-outlet", "measure": "temperature", "unit": "°C",
    "label": "T-201 outlet temperature", "range": [0, 120], "normal": [15.0, 55.0], "nominal": 31.0,
    "dial_m": 0.16, "pos": list(tg201), "facing": [0, -1], "stand": stand_for(tg201, (0, -1), 1.05),
})
xv201 = (40.0, 25.75, 0.6)
instruments.append({
    "tag": "XV-201", "kind": "valve", "style": "gate_h", "place": "tank-201-outlet", "label": "T-201 outlet valve",
    "normal_state": "open", "pos": list(xv201), "facing": [0, -1], "stand": stand_for((40.0, 25.6, 1.1), (0, -1), 1.6),
})
# T-202 drain: low line through the bund, ball valve, into a grated drain pit
pipe("t202-drain", [(48.0, 30.6 - TANK_R, 0.35), (48.0, 25.0, 0.35), (48.0, 25.0, -0.3)], 0.05)
box("drain-pit", (48.0, 24.95, 0.01), (0.7, 0.5, 0.02), group="equipment", material="grating", collide=False)
hv202 = (48.0, 25.55, 0.35)
instruments.append({
    "tag": "HV-202", "kind": "valve", "style": "ball", "place": "tank-202-drain", "label": "T-202 drain valve",
    "normal_state": "closed", "pos": list(hv202), "facing": [0, -1], "stand": stand_for((48.0, 25.55, 0.4), (0, -1), 1.5),
})
pipe("t202-outlet", [(46.6, 30.6 - TANK_R + 0.25, 0.6), (46.6, 24.95, 0.6), (46.6, 24.95, -0.3)], 0.08)
pipe("t202-tg-stub", [(46.6, 25.2, 0.6), (46.6, 25.2, 1.08)], 0.03, material="steel_galv")
tg202 = (46.6, 25.2 - 0.03 - 0.1, 1.16)
instruments.append({
    "tag": "TG-202", "kind": "gauge", "place": "tank-202-drain", "measure": "temperature", "unit": "°C",
    "label": "T-202 outlet temperature", "range": [0, 120], "normal": [15.0, 55.0], "nominal": 29.0,
    "dial_m": 0.16, "pos": list(tg202), "facing": [0, -1], "stand": stand_for(tg202, (0, -1), 1.05),
})

# ---------------------------------------------------------------- air compressor package
box("comp-skid", (47.0, 7.0, 0.1), (5.6, 2.0, 0.2), group="equipment", material="steel_dark")
cyl("M-301", (44.6, 7.0, 0.62), (45.6, 7.0, 0.62), 0.32, group="equipment", material="motor_blue", thermal="motor",
    tag="M-301")
cyl("M-301-fan", (44.45, 7.0, 0.62), (44.6, 7.0, 0.62), 0.3, group="equipment", material="steel_dark")
box("comp-belt-guard", (46.0, 6.45, 0.75), (0.9, 0.12, 0.8), group="equipment", material="guard_yellow")
box("C-301", (47.2, 7.0, 0.75), (1.1, 1.0, 1.1), group="equipment", material="comp_grey", thermal="compressor",
    tag="C-301")
cyl("comp-receiver", (50.4, 7.2, 0.2), (50.4, 7.2, 2.4), 0.55, group="equipment", material="receiver_grey",
    tag="R-301")
pipe("comp-discharge", [(47.75, 7.0, 1.0), (49.85, 7.0, 1.0)], 0.04)
pg301 = (50.4, 7.2 - 0.55 - 0.1, 1.3)
box("pg301-boss", (50.4, 7.2 - 0.55 - 0.03, 1.3), (0.06, 0.06, 0.06), group="equipment", material="steel_galv",
    collide=False)
instruments.append({
    "tag": "PG-301", "kind": "gauge", "place": "compressor-301", "measure": "pressure", "unit": "bar",
    "label": "Air receiver pressure", "range": [0, 16], "normal": [7.0, 10.0], "nominal": 8.6,
    "dial_m": 0.16, "pos": list(pg301), "facing": [0, -1], "stand": stand_for(pg301, (0, -1), 1.15),
})
pipe("comp-outlet", [(50.95, 7.2, 1.0), (51.6, 7.2, 1.0), (51.6, 7.2, -0.3)], 0.04)
xv301 = (51.6, 7.2 - 0.04, 0.75)
instruments.append({
    "tag": "XV-301", "kind": "valve", "style": "gate", "place": "compressor-301", "label": "Air receiver outlet valve",
    "normal_state": "open", "pos": list(xv301), "facing": [0, -1], "stand": stand_for((51.6, 7.0, 0.9), (0, -1), 1.6),
})
m301 = (45.1, 7.0 - 0.32, 0.62)
instruments.append({
    "tag": "M-301", "kind": "motor", "place": "compressor-301", "label": "Compressor motor",
    "nominal_c": 58.0, "warn_c": 80.0, "alarm_c": 90.0, "pos": list(m301), "facing": [0, -1],
    "stand": stand_for(m301, (0, -1), 1.6),
})

# ---------------------------------------------------------------- instrument air panel (control room)
pg401 = (8.0, cy1 - WALL_T - 0.5 - 0.02, 1.1)
instruments.append({
    "tag": "PG-401", "kind": "gauge", "place": "control-room", "measure": "pressure", "unit": "bar",
    "label": "Instrument air header", "range": [0, 10], "normal": [5.5, 7.5], "nominal": 6.6,
    "dial_m": 0.16, "pos": list(pg401), "facing": [0, -1], "stand": stand_for(pg401, (0, -1), 1.1),
})

# ---------------------------------------------------------------- other site furniture
box("eyewash", (34.8, 11.0, 1.1), (0.4, 0.4, 2.2), group="equipment", material="eyewash_green")
box("extinguisher-1", (17.4, 22.3, 0.45), (0.25, 0.25, 0.9), group="equipment", material="extinguisher_red")
box("extinguisher-2", (52.6, 12.8, 0.45), (0.25, 0.25, 0.9), group="equipment", material="extinguisher_red")
box("mcc", (12.0, 24.95, 1.1), (3.0, 0.7, 2.2), group="equipment", material="panel_grey")  # motor control centre

# ---------------------------------------------------------------- lights (poles and wall packs)
lights = [
    {"id": f"pole-{i}", "kind": "pole", "pos": [x, y, 8.0], "watts": 160, "kelvin": 4000}
    for i, (x, y) in enumerate(((18.0, 9.5), (36.0, 9.5), (52.0, 15.0), (30.0, 24.8), (55.0, 32.0), (17.5, 30.0)))
] + [
    {"id": "wall-cr", "kind": "wallpack", "pos": [8.0, 25.8, 3.0], "watts": 60, "kelvin": 4000},
    {"id": "cr-ceiling", "kind": "indoor", "pos": [8.0, 30.0, 3.4], "watts": 120, "kelvin": 5000},
    {"id": "shelter", "kind": "indoor", "pos": [27.0, 14.7, 3.9], "watts": 120, "kelvin": 4000},
]
for lt in lights:
    if lt["kind"] == "pole":
        x, y, _ = lt["pos"]
        cyl(f"{lt['id']}-shaft", (x, y, 0.0), (x, y, 8.0), 0.09, group="structure", material="steel_galv")

# ---------------------------------------------------------------- places (what the planner and the map see)
def center_of(tags: list[str]) -> list[float]:
    pts = [i["pos"] for i in instruments if i["tag"] in tags]
    return [round(sum(p[0] for p in pts) / len(pts), 2), round(sum(p[1] for p in pts) / len(pts), 2)]


places += [
    {"id": "control-room", "name": "Control room", "kind": "station", "indoor": True,
     "center": center_of(["PG-401"]), "checks": ["PG-401"]},
    *[{"id": f"pump-{n}", "name": f"Pump P-{n}", "kind": "station", "indoor": False,
       "center": [PUMP_X[n], PUMP_Y], "checks": [f"PG-{n}", f"XV-{n}", f"M-{n}", f"DT-{n}"]} for n in PUMP_X],
    {"id": "tank-201-outlet", "name": "Tank T-201 outlet", "kind": "station", "indoor": False,
     "center": [40.0, 25.4], "checks": ["TG-201", "XV-201"]},
    {"id": "tank-202-drain", "name": "Tank T-202 drain", "kind": "station", "indoor": False,
     "center": [47.4, 25.4], "checks": ["TG-202", "HV-202"]},
    {"id": "compressor-301", "name": "Air compressor C-301", "kind": "station", "indoor": False,
     "center": [47.5, 7.0], "checks": ["PG-301", "XV-301", "M-301"]},
    {"id": "rack-stairs", "name": "Pipe rack stairs", "kind": "stay_out", "indoor": False,
     "center": [50.5, 18.3], "checks": [],
     "poly": [[STAIR_RACK[0] - 0.3, STAIR_RACK[1] - 0.3], [STAIR_RACK[2] + 0.3, STAIR_RACK[1] - 0.3],
              [STAIR_RACK[2] + 0.3, 19.9], [STAIR_RACK[0] - 0.3, 19.9]]},
    {"id": "bund-stile", "name": "Tank bund stile", "kind": "stay_out", "indoor": False, "center": [44.0, 25.4],
     "checks": [], "poly": [[43.1, 24.6], [44.9, 24.6], [44.9, 26.3], [43.1, 26.3]]},
]
for p in places:
    if p["kind"] == "station":
        stands = [i["stand"] for i in instruments if i["tag"] in p["checks"]]
        # the dog's goto pose for the station (`go_to_place` walks there): the first check's stand point
        p["pose"] = {"x": stands[0][0], "y": stands[0][1], "yaw": stands[0][2]}
DOCK = {"id": "dock", "name": "Go2 dock", "kind": "dock", "indoor": False, "center": [5.0, 23.6],
        "pose": {"x": 5.0, "y": 23.6, "yaw": 0.0}, "checks": []}
places.append(DOCK)

# ---------------------------------------------------------------- fixed cameras (2 MP bullets, dimos_worlds/lot/camera_model)
cameras = {
    "cam1": {"name": "Cam 1 pump bay", "mount": [18.25, 9.6, 4.6], "target": [27.5, 14.2, 0.8], "lens_mm": 4.0,
             "places": ["pump-101", "pump-102", "pump-103"]},
    "cam2": {"name": "Cam 2 tank outlets", "mount": [55.6, 22.0, 5.0], "target": [44.0, 25.3, 0.8], "lens_mm": 4.0,
             "places": ["tank-201-outlet", "tank-202-drain", "bund-stile"]},
    "cam3": {"name": "Cam 3 compressor", "mount": [55.6, 1.0, 4.6], "target": [48.0, 6.6, 0.8], "lens_mm": 4.0,
             "places": ["compressor-301"]},
    "cam4": {"name": "Cam 4 control room", "mount": [13.4, 26.6, 3.1], "target": [8.0, 33.0, 0.9], "lens_mm": 2.8,
             "places": ["control-room"]},
    "cam5": {"name": "Cam 5 west yard", "mount": [2.0, 9.0, 5.0], "target": [8.0, 23.0, 0.4], "lens_mm": 2.8,
             "places": ["dock"]},
    "cam6": {"name": "Cam 6 rack east", "mount": [55.6, 12.0, 5.0], "target": [50.5, 18.3, 1.0], "lens_mm": 4.0,
             "places": ["rack-stairs"]},
}
for c in cameras.values():
    c["hfov_deg"] = 100.0 if c["lens_mm"] == 2.8 else 80.0  # lot/camera_model.LENSES (effective, UNVERIFIED)

# ---------------------------------------------------------------- rounds route and robot
ROUNDS = ["control-room", "pump-101", "pump-102", "pump-103", "compressor-301", "tank-202-drain", "tank-201-outlet"]

layout = {
    "version": VERSION,
    "generated_by": "dimos_worlds/plant/gen_layout.py",
    "site": SITE,
    "frame": "metres; origin SW corner of the slab; x east, y north, z up (glTF Y-up: three = (x, z, -y))",
    "slab": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
    "boundary": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
    "walkways": walkways,
    "solids": solids,
    "pipes": pipes,
    "instruments": instruments,
    "places": places,
    "cameras": cameras,
    "lights": lights,
    "robot": {"start": [5.0, 23.6, 0.0], "dock": "dock", "camera_z_m": 0.40,
              "payload": {"note": "pan-tilt inspection camera + radiometric thermal core on the Go2's back; "
                                  "a payload assumption, UNVERIFIED (the Go2 Air ships with neither)",
                          "zoom_hfov_deg": 12.0, "zoom_px": [960, 720], "thermal_px": [160, 120],
                          "thermal_hfov_deg": 57.0, "mount_z_m": 0.55}},
    "routes": {"rounds": ROUNDS},
    "events": {
        "gauge_high": "gauge_high:<tag>[:value]   gauge reads above its normal band (default: 85% of span)",
        "gauge_low": "gauge_low:<tag>[:value]     gauge reads below its normal band",
        "motor_hot": "motor_hot:<tag>[:degC]      motor frame temperature (default: alarm + 9 C)",
        "valve_wrong": "valve_wrong:<tag>         valve in the opposite of its normal state",
        "leak": "leak:<drip tray tag>              liquid on the slab under a pump seal",
        "scenario faults": ["gauge_high:PG-102:8.9", "motor_hot:M-103:94", "valve_wrong:HV-202"],
    },
}

if __name__ == "__main__":
    OUT.write_text(json.dumps(layout, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(solids)} solids, {len(pipes)} pipes, {len(instruments)} instruments, "
          f"{len(places)} places, {len(cameras)} cameras")
