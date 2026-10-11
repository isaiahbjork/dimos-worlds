"""Instrument truth for the Demo plant: what every gauge reads, how hot every motor runs, where every valve is.

Pure function of (layout, events). Stdlib only, so Blender's python, the services and the tests share it.

    from dimos_worlds.plant.plant_state import truth
    state = truth(["gauge_high:PG-102:8.9", "motor_hot:M-103:94", "valve_wrong:HV-202"])
    state["gauges"]["PG-102"]   -> {"value": 8.9, "angle_deg": ..., "in_normal": False, ...}

Gauge dials sweep 270 degrees clockwise: the minimum at the 7:30 position (225 deg, maths angle: 0 = 3 o'clock,
counter-clockwise positive), the maximum at 4:30 (-45 deg). `needle_angle(value)` is what the Blender scene and the 2D
fallback draw; `value_at(angle)` is what a reader gets back. Motor temperatures are frame (surface) temperatures in C.
Event grammar: layout.json "events".

Running state file (sim services): STATE_FILE env DIMOS_WORLDS_PLANT_STATE (JSON {"events": [...], "updated_at": ...});
`load_events()` / `save_events()` read and write it atomically.
"""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYOUT = HERE / "layout.json"
SWEEP_DEG = 270.0
START_DEG = 225.0  # maths angle of the dial minimum
AMBIENT_C = 24.0
SCENARIO_FAULTS = ("gauge_high:PG-102:8.9", "motor_hot:M-103:94", "valve_wrong:HV-202")


class EventError(ValueError):
    pass


@lru_cache(maxsize=1)
def layout() -> dict:
    return json.loads(LAYOUT.read_text())


def instruments() -> dict[str, dict]:
    return {i["tag"]: i for i in layout()["instruments"]}


def needle_angle(spec: dict, value: float) -> float:
    lo, hi = spec["range"]
    frac = min(1.0, max(0.0, (value - lo) / (hi - lo)))
    return START_DEG - SWEEP_DEG * frac


def value_at(spec: dict, angle_deg: float) -> float:
    """Inverse of needle_angle. angle in maths degrees, any wrap; the dead zone (bottom) clamps to the nearer end."""
    lo, hi = spec["range"]
    a = (START_DEG - angle_deg) % 360.0  # degrees clockwise from the minimum
    if a > SWEEP_DEG:
        a = SWEEP_DEG if a - SWEEP_DEG < (360.0 - a) else 0.0
    return lo + (hi - lo) * a / SWEEP_DEG


def parse_event(spec: str) -> tuple[str, str, float | None]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise EventError(f"bad event {spec!r}: want kind:tag[:value]")
    kind, tag = parts[0], parts[1]
    ins = instruments().get(tag)
    if ins is None:
        raise EventError(f"unknown instrument {tag!r}")
    want = {"gauge_high": "gauge", "gauge_low": "gauge", "motor_hot": "motor", "valve_wrong": "valve",
            "leak": "drip_tray"}.get(kind)
    if want is None:
        raise EventError(f"unknown event kind {kind!r}")
    if ins["kind"] != want:
        raise EventError(f"{kind} needs a {want}, {tag} is a {ins['kind']}")
    value = None
    if len(parts) > 2 and parts[2] != "":
        try:
            value = float(parts[2])
        except ValueError as e:
            raise EventError(f"bad value in {spec!r}") from e
    return kind, tag, value


def truth(events: list[str] | tuple[str, ...] = ()) -> dict:
    """{"gauges": {tag: {...}}, "motors": {tag: {...}}, "valves": {tag: {...}}, "trays": {tag: {...}}, "events": [...]}"""
    ins = instruments()
    gauges, motors, valves, trays = {}, {}, {}, {}
    for tag, i in ins.items():
        if i["kind"] == "gauge":
            gauges[tag] = {"value": float(i["nominal"])}
        elif i["kind"] == "motor":
            motors[tag] = {"temp_c": float(i["nominal_c"])}
        elif i["kind"] == "valve":
            valves[tag] = {"state": i["normal_state"]}
        elif i["kind"] == "drip_tray":
            trays[tag] = {"leak": False}
    for ev in events:
        kind, tag, value = parse_event(ev)
        i = ins[tag]
        if kind == "gauge_high":
            lo, hi = i["range"]
            gauges[tag]["value"] = value if value is not None else round(lo + 0.85 * (hi - lo), 2)
        elif kind == "gauge_low":
            lo, hi = i["range"]
            gauges[tag]["value"] = value if value is not None else round(lo + 0.12 * (hi - lo), 2)
        elif kind == "motor_hot":
            motors[tag]["temp_c"] = value if value is not None else float(i["alarm_c"]) + 9.0
        elif kind == "valve_wrong":
            valves[tag]["state"] = "closed" if i["normal_state"] == "open" else "open"
        elif kind == "leak":
            trays[tag]["leak"] = True
    for tag, g in gauges.items():
        i = ins[tag]
        lo, hi = i["normal"]
        g.update(angle_deg=round(needle_angle(i, g["value"]), 3), unit=i["unit"], normal=[lo, hi],
                 in_normal=lo <= g["value"] <= hi)
    for tag, m in motors.items():
        i = ins[tag]
        m.update(warn_c=i["warn_c"], alarm_c=i["alarm_c"], status=status_for_temp(i, m["temp_c"]))
    for tag, v in valves.items():
        v.update(normal_state=ins[tag]["normal_state"], as_expected=v["state"] == ins[tag]["normal_state"])
    return {"gauges": gauges, "motors": motors, "valves": valves, "trays": trays, "events": list(events)}


def status_for_temp(spec: dict, temp_c: float) -> str:
    if temp_c >= spec["alarm_c"]:
        return "alarm"
    if temp_c >= spec["warn_c"]:
        return "warning"
    return "normal"


def surface_temps(state: dict) -> dict[str, float]:
    """Apparent temperature per solid / instrument tag for the thermal channel (C). Pumps run a bit above ambient
    and pick up heat from a hot motor; tanks follow their outlet gauge; everything else is ambient."""
    out: dict[str, float] = {}
    for tag, m in state["motors"].items():
        out[tag] = m["temp_c"]
        pump = "P-" + tag.split("-")[1]
        out[pump] = max(AMBIENT_C + 8.0, AMBIENT_C + 0.35 * (m["temp_c"] - AMBIENT_C))
    for tag, g in state["gauges"].items():
        if g["unit"] == "°C" and tag.startswith("TG-"):
            out["T-" + tag.split("-")[1]] = g["value"]
    out.setdefault("C-301", AMBIENT_C + 22.0)
    return out


# ---------------------------------------------------------------- running state file (sim services)
def state_file() -> Path:
    return Path(os.environ.get("DIMOS_WORLDS_PLANT_STATE") or "/tmp/dimos-worlds-plant-state.json")


def load_events(path: Path | None = None) -> list[str]:
    try:
        return list(json.loads((path or state_file()).read_text()).get("events") or [])
    except (OSError, ValueError):
        return []


def save_events(events: list[str], path: Path | None = None) -> None:
    for ev in events:
        parse_event(ev)  # validate before writing
    p = path or state_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"events": list(events), "updated_at": time.time()}))
    os.replace(tmp, p)
