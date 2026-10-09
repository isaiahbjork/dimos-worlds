"""Scene events: explicit edits of the lot's MuJoCo state.

Every movable thing in the lot is a mocap body (see gen_lot.py), so an event is a write to
`data.mocap_pos` / `data.mocap_quat` followed by `mj_forward`. Nothing is stepped, so the same events
always give the same state, and a robot loaded next to the lot keeps its own qpos layout.

    reset(model, data)                              baseline: gates shut, vehicles parked, person out of view
    person_at(model, data, place_id)                person stands at a place, with that place's heading
    gate_open(model, data, gate_id)                 front-gate slides 6.2 m west; gate-2 swings 85 deg in
    gate_close(model, data, gate_id)
    vehicle_moved(model, data, row, from_spot, to_spot)
    apply(model, data, spec)                        one event from a spec string (below)

Spec strings (also used by the DimOS event topic and the render pipeline):

    person_at:<place_id>        e.g. person_at:north-fence
    person_clear
    gate_open:<gate_id>         e.g. gate_open:gate-2
    gate_close:<gate_id>
    vehicle_moved:<row>:<from>:<to>   e.g. vehicle_moved:row-c:14:16
    reset

The functions work on any model that contains the lot's bodies: the standalone scene/lot.xml, or the lot
merged with a robot for DimOS's Go2 sim (where the person is DimOS's own `person` mesh body, also mocap).

`render_state(model, data)` exports the event state for the optional Blender renderer (render/).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dimos_worlds.lot import layout
from dimos_worlds.lot.places import load_places

PLACES = load_places()
PERSON_BODY = "person"
PERSON_PARK = np.array([*layout.PERSON_PARK, 0.0])
SPOT_TOLERANCE_M = 1.0
ROW_SPOTS = {
    "row-a": range(1, 7),
    "row-b": range(7, 13),
    "row-c": range(13, 19),
    "row-d": range(19, 25),
}


def _mj():  # type: ignore[no-untyped-def]
    import mujoco

    return mujoco


# ---- events -------------------------------------------------------------------------------------


def reset(model: Any, data: Any) -> None:
    """Back to the baseline: every mocap body at its XML pose, person parked out of view."""
    for bid in range(model.nbody):
        mid = model.body_mocapid[bid]
        if mid >= 0:
            data.mocap_pos[mid] = model.body_pos[bid]
            data.mocap_quat[mid] = model.body_quat[bid]
    if _has_body(model, PERSON_BODY):
        _mocap_pos(model, data, PERSON_BODY)[:] = PERSON_PARK
    _mj().mj_forward(model, data)


def person_at(model: Any, data: Any, place_id: str) -> None:
    place = _place(place_id)
    _mocap_pos(model, data, PERSON_BODY)[:] = [place.x, place.y, 0.0]
    _, yaw = person_pose(place_id)
    _mocap_quat(model, data, PERSON_BODY)[:] = yaw_quat(yaw)
    _mj().mj_forward(model, data)


def person_clear(model: Any, data: Any) -> None:
    _mocap_pos(model, data, PERSON_BODY)[:] = PERSON_PARK
    _mj().mj_forward(model, data)


def person_pose(place_id: str) -> tuple[str, float]:
    """(body pose, heading yaw deg) the person takes at a place. Unlisted places: standing, facing north."""
    return layout.PERSON_POSES.get(place_id, ("stand", 0.0))


def person_place(model: Any, data: Any) -> str | None:
    """The place the person stands at, or None if elsewhere."""
    pos = _mocap_pos(model, data, PERSON_BODY)
    for pid, p in PLACES.items():
        if abs(pos[0] - p.x) < 0.05 and abs(pos[1] - p.y) < 0.05:
            return pid
    return None


def gate_open(model: Any, data: Any, gate_id: str) -> None:
    g = _gate(gate_id)
    lo, hi = g["range"]
    _set_gate(model, data, gate_id, hi if abs(hi) >= abs(lo) else lo)


def gate_close(model: Any, data: Any, gate_id: str) -> None:
    _gate(gate_id)
    _set_gate(model, data, gate_id, 0.0)


def gate_value(model: Any, data: Any, gate_id: str) -> float:
    """Slide offset in metres (front-gate) or hinge angle in degrees (gate-2); 0 = shut."""
    g = _gate(gate_id)
    bid = model.body(gate_id).id
    if g["type"] == "slide":
        delta = _mocap_pos(model, data, gate_id)[:2] - model.body_pos[bid][:2]
        return float(np.dot(delta, g["axis"]))
    q = _mocap_quat(model, data, gate_id)
    return math.degrees(2 * math.atan2(q[3], q[0]))


def vehicle_moved(model: Any, data: Any, row: str, from_spot: int, to_spot: int) -> None:
    if row not in ROW_SPOTS:
        raise ValueError(f"unknown row {row!r}; known: {sorted(ROW_SPOTS)}")
    for spot in (from_spot, to_spot):
        if spot not in ROW_SPOTS[row]:
            raise ValueError(f"spot {spot} is not in {row} (spots {ROW_SPOTS[row].start}-{ROW_SPOTS[row].stop - 1})")
    if from_spot == to_spot:
        raise ValueError("from and to are the same spot")
    body_id = vehicle_at_spot(model, data, from_spot)
    if body_id is None:
        raise ValueError(f"no vehicle at spot {from_spot}")
    if vehicle_at_spot(model, data, to_spot) is not None:
        raise ValueError(f"spot {to_spot} is occupied")
    site = model.site(f"spot-{to_spot}")
    mid = model.body_mocapid[body_id]
    data.mocap_pos[mid] = [site.pos[0], site.pos[1], 0.0]
    data.mocap_quat[mid] = model.site_quat[site.id]
    _mj().mj_forward(model, data)


def vehicle_at_spot(model: Any, data: Any, spot: int) -> int | None:
    """Body id of the movable vehicle standing within SPOT_TOLERANCE_M of a spot, or None."""
    target = model.site(f"spot-{spot}").pos[:2]
    for bid in _movable_vehicles(model):
        pos = data.mocap_pos[model.body_mocapid[bid]]
        if np.linalg.norm(pos[:2] - target) <= SPOT_TOLERANCE_M:
            return bid
    return None


def apply(model: Any, data: Any, spec: str) -> None:
    """Apply one event spec string (see the module docstring). Raises ValueError on a bad spec."""
    kind, _, rest = spec.strip().partition(":")
    args = rest.split(":") if rest else []
    if kind == "reset" and not args:
        reset(model, data)
    elif kind == "person_at" and len(args) == 1:
        person_at(model, data, args[0])
    elif kind == "person_clear" and not args:
        person_clear(model, data)
    elif kind == "gate_open" and len(args) == 1:
        gate_open(model, data, args[0])
    elif kind == "gate_close" and len(args) == 1:
        gate_close(model, data, args[0])
    elif kind == "vehicle_moved" and len(args) == 3:
        try:
            frm, to = int(args[1]), int(args[2])
        except ValueError as exc:
            raise ValueError(f"bad spot numbers in {spec!r}") from exc
        vehicle_moved(model, data, args[0], frm, to)
    else:
        raise ValueError(
            f"bad event spec {spec!r}; expected person_at:<place>, person_clear, gate_open:<gate>, "
            "gate_close:<gate>, vehicle_moved:<row>:<from>:<to> or reset"
        )


def render_state(model: Any, data: Any) -> dict:
    """Event state for render/render_frames.py: moved vehicles, gate values, person."""
    state: dict = {"vehicles": {}, "gates": {}, "person": None}
    for bid in _movable_vehicles(model):
        mid = model.body_mocapid[bid]
        q, p = data.mocap_quat[mid], data.mocap_pos[mid]
        if np.allclose(p, model.body_pos[bid], atol=1e-6) and np.allclose(q, model.body_quat[bid], atol=1e-6):
            continue
        yaw = math.degrees(2 * math.atan2(q[3], q[0]))
        state["vehicles"][model.body(bid).name] = [float(p[0]), float(p[1]), yaw]
    for gid in layout.GATES:
        val = gate_value(model, data, gid)
        if abs(val) > 1e-9:
            state["gates"][gid] = val
    pid = person_place(model, data)
    if pid is not None:
        pose, yaw = person_pose(pid)
        state["person"] = {"x": PLACES[pid].x, "y": PLACES[pid].y, "yaw": yaw, "pose": pose, "place": pid}
    return state


def yaw_quat(yaw_deg: float) -> list[float]:
    h = math.radians(yaw_deg) / 2
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


# ---- helpers --------------------------------------------------------------------------------------


def _place(place_id: str):  # type: ignore[no-untyped-def]
    if place_id not in PLACES:
        raise ValueError(f"unknown place {place_id!r}; known: {sorted(PLACES)}")
    return PLACES[place_id]


def _gate(gate_id: str) -> dict:
    if gate_id not in layout.GATES:
        raise ValueError(f"unknown gate {gate_id!r}; known: {sorted(layout.GATES)}")
    return layout.GATES[gate_id]


def _set_gate(model: Any, data: Any, gate_id: str, value: float) -> None:
    g = layout.GATES[gate_id]
    bid = model.body(gate_id).id
    if g["type"] == "slide":
        ax, ay = g["axis"]
        _mocap_pos(model, data, gate_id)[:] = model.body_pos[bid] + np.array([ax * value, ay * value, 0.0])
    else:
        _mocap_quat(model, data, gate_id)[:] = yaw_quat(value)
    _mj().mj_forward(model, data)


def _has_body(model: Any, name: str) -> bool:
    mujoco = _mj()
    return bool(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0)


def _mocap_id(model: Any, body: str) -> int:
    if not _has_body(model, body):
        raise ValueError(f"model has no body {body!r}")
    mid = int(model.body_mocapid[model.body(body).id])
    if mid < 0:
        raise ValueError(f"body {body!r} is not a mocap body")
    return mid


def _mocap_pos(model: Any, data: Any, body: str) -> np.ndarray:
    return data.mocap_pos[_mocap_id(model, body)]


def _mocap_quat(model: Any, data: Any, body: str) -> np.ndarray:
    return data.mocap_quat[_mocap_id(model, body)]


def _movable_vehicles(model: Any) -> list[int]:
    return [
        bid
        for bid in range(model.nbody)
        if model.body(bid).name.startswith("veh-") and model.body_mocapid[bid] >= 0
    ]
