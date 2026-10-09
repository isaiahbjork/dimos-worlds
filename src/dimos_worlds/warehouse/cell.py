"""The warehouse floor plan: places, slots, totes, the arm cell and walking routes, all read from layout.json (made
by gen_layout.py). Pure Python (no MuJoCo).

Frame: x east, y north, z up, metres, origin = SW interior corner of the 30 x 20 m building.

Places:
  shelf-a .. shelf-d   the pick bay (bay 1) of pallet rack rows A-D: two wire decks (0.40 m, 0.85 m), 3 totes each,
                       at the rack's aisle edge
  floor                the floor of aisle A-B by the shelves
  conveyor             infeed conveyor, top 0.75 m; totes queue on it and stop at its east end, by the arm
  pick-table           the arm's table, top 0.75 m, 2 slots
  pallet-1             outbound staging pallet by the dock (front row of 3 slots)
  pallet-2             the arm's half pallet (800 x 600), 2 layers x 4 slots
  charging-spot        where the humanoid parks
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

LAYOUT_PATH = Path(__file__).resolve().with_name("layout.json")
LAYOUT: dict = json.loads(LAYOUT_PATH.read_text())

TOTE_SIZE: tuple[float, float, float] = tuple(LAYOUT["totes"]["size"])  # type: ignore[assignment]  # x, y, z
TOTE_MASS_KG = float(LAYOUT["totes"]["mass_kg"])
CELL_SIZE = (float(LAYOUT["building"]["size"][0]), float(LAYOUT["building"]["size"][1]))

_ARM = LAYOUT["arm_cell"]
# The arm station: KUKA iiwa 14 on its pedestal, between the conveyor's end, the pick table and pallet 2.
ARM_BASE = (float(_ARM["pedestal_center"][0]), float(_ARM["pedestal_center"][1]), float(_ARM["pedestal_height"]))
ARM_PEDESTAL_HALF = (float(_ARM["pedestal_size"][0]) / 2, float(_ARM["pedestal_size"][1]) / 2)
ARM_REACH_M = float(_ARM["reach_m"])  # joint-2 to wrist centre, minus a margin (iiwa 14: 0.42 + 0.40)
ARM_TOOL_M = float(_ARM["tool_m"])  # suction tool below the flange
ARM_WRIST_TO_TIP_M = float(_ARM["wrist_to_tip_m"])  # joint 6 to the tool tip, pointing down
ARM_CLEARANCE_M = float(_ARM["clearance_m"])  # the tool comes down onto a tote from this high, and lifts it this high
ARM_TRANSIT = tuple(float(v) for v in _ARM["transit_offset"])  # high point the tool passes, from the base

_HUM = LAYOUT["humanoid"]
HUMANOID_HOME = (float(_HUM["home"][0]), float(_HUM["home"][1]), float(_HUM["home"][2]))  # x, y, yaw
EDGE_GAP_M = float(_HUM["edge_gap_m"])  # humanoid pelvis to the near edge of a shelf, table, conveyor or pallet
FLOOR_STANDOFF_M = float(_HUM["floor_standoff_m"])  # humanoid pelvis to a tote on the floor
HUMANOID_REACH_M = float(_HUM["reach_m"])  # pelvis to the tote centre, horizontally, that its arms can still close on
HUMANOID_REACH_Z = (float(_HUM["reach_z"][0]), float(_HUM["reach_z"][1]))  # tote base heights it picks from
HUMANOID_RADIUS_M = float(_HUM["body_radius_m"])  # clearance a walking route keeps from racks and equipment

_CONV = LAYOUT["conveyor"]
CONVEYOR_TOP = float(_CONV["top_z"])
CONVEYOR_PICK_X = float(_CONV["pick_x"])  # the tote at the end waits here, north of the arm base
CONVEYOR_PITCH = float(_CONV["pitch"])  # spacing of queued totes
CONVEYOR_SPEED_MS = float(_CONV["speed_ms"])
CONVEYOR_RECT = tuple(float(v) for v in _CONV["rect"])


@dataclass(frozen=True)
class Slot:
    id: str
    x: float
    y: float
    z: float  # height of the surface the tote stands on

    @property
    def center(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z + TOTE_SIZE[2] / 2)


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    kind: str  # shelf | floor | table | conveyor | pallet | dock
    center: tuple[float, float]
    size: tuple[float, float]  # footprint x, y
    top_z: float  # surface height (0 = floor)
    approach_yaw: float  # which way a humanoid faces when it works here
    slots: tuple[Slot, ...] = field(default_factory=tuple)

    def contains(self, x: float, y: float, z: float, margin: float = 0.05) -> bool:
        """Whether a tote centred at (x, y, z) stands on this location."""
        cx, cy = self.center
        sx, sy = self.size
        if abs(x - cx) > sx / 2 + margin or abs(y - cy) > sy / 2 + margin:
            return False
        if self.kind == "floor":
            return z < 0.25
        if self.kind == "shelf":
            return z > 0.2
        return z > self.top_z - 0.05

    def stand_point(self, x: float, y: float) -> tuple[float, float, float]:
        """Where a humanoid stands to work on a tote at (x, y) here, facing approach_yaw: EDGE_GAP in front of the
        location's near edge (for the floor: FLOOR_STANDOFF from the tote itself)."""
        yaw = self.approach_yaw
        ux, uy = math.cos(yaw), math.sin(yaw)
        if self.kind in ("floor", "dock"):
            return (x - FLOOR_STANDOFF_M * ux, y - FLOOR_STANDOFF_M * uy, yaw) if self.kind == "floor" else (x, y, yaw)
        cx, cy = self.center
        sx, sy = self.size
        half_depth = abs(ux) * sx / 2 + abs(uy) * sy / 2
        # distance along the facing direction from the tote to the near edge, then the gap
        along = (x - cx) * ux + (y - cy) * uy + half_depth
        back = along + EDGE_GAP_M
        return (x - back * ux, y - back * uy, yaw)


def _location(row: dict) -> Location:
    return Location(
        id=row["id"], name=row["name"], kind=row["kind"], center=(float(row["center"][0]), float(row["center"][1])),
        size=(float(row["size"][0]), float(row["size"][1])), top_z=float(row["top_z"]),
        approach_yaw=math.radians(float(row["approach_yaw_deg"])),
        slots=tuple(Slot(s["id"], float(s["xyz"][0]), float(s["xyz"][1]), float(s["xyz"][2])) for s in row["slots"]),
    )


LOCATIONS: tuple[Location, ...] = tuple(_location(row) for row in LAYOUT["places"])
BY_ID: dict[str, Location] = {loc.id: loc for loc in LOCATIONS}
HOME_ID = "charging-spot"

# Where the totes start: slot id or (x, y, z of the tote's base, yaw).
INITIAL_TOTES: tuple[tuple[str, str | tuple[float, float, float, float]], ...] = tuple(
    (str(tid), where if isinstance(where, str) else tuple(float(v) for v in where))  # type: ignore[misc]
    for tid, where in LAYOUT["totes"]["initial"]
)


def slot(slot_id: str) -> Slot:
    for loc in LOCATIONS:
        for s in loc.slots:
            if s.id == slot_id:
                return s
    raise KeyError(slot_id)


def initial_poses() -> dict[str, tuple[float, float, float, float]]:
    """tote id -> (x, y, z of the centre, yaw)."""
    out: dict[str, tuple[float, float, float, float]] = {}
    for tote_id, where in INITIAL_TOTES:
        if isinstance(where, str):
            s = slot(where)
            out[tote_id] = (*s.center, 0.0)
        else:
            x, y, z, yaw = where
            out[tote_id] = (x, y, z + TOTE_SIZE[2] / 2, yaw)
    return out


def location_of(x: float, y: float, z: float) -> str | None:
    """The location a tote centred at (x, y, z) is on, or None (in a hand, mid-air, somewhere else)."""
    for loc in LOCATIONS:
        if loc.kind in ("floor", "dock"):
            continue
        if loc.contains(x, y, z):
            return loc.id
    if z < 0.25 and 0.0 <= x <= CELL_SIZE[0] and 0.0 <= y <= CELL_SIZE[1]:
        return "floor"  # anything lying on the floor counts as "on the floor"
    return None


def arm_can_reach(x: float, y: float, z_top: float) -> bool:
    """Whether the iiwa can put its suction tool on a tote top at (x, y, z_top), straight down: the wrist centre
    (tool + flange above the top) must be within reach of joint 2, and not right over the base."""
    bx, by, bz = ARM_BASE
    shoulder_z = bz + 0.36
    horizontal = math.hypot(x - bx, y - by)
    if horizontal < 0.30:
        return False
    for lift in (0.0, ARM_CLEARANCE_M):  # touch down, and the approach above it
        wrist_z = z_top + lift + ARM_WRIST_TO_TIP_M
        if math.hypot(horizontal, wrist_z - shoulder_z) > ARM_REACH_M:
            return False
    return True


def humanoid_reach_refusal(loc: Location, x: float, y: float, base_z: float) -> str | None:
    """None when a standing (or crouching) humanoid can get its hands on a tote whose base is at base_z on `loc`."""
    if loc.kind == "dock":
        return f"{loc.name} is not somewhere totes go"
    low, high = HUMANOID_REACH_Z
    if not low <= base_z <= high:
        return f"{loc.name} at {base_z:.2f} m is out of a humanoid's reach ({low:.1f} to {high:.2f} m)"
    sx, sy, _ = loc.stand_point(x, y)
    if math.hypot(x - sx, y - sy) > HUMANOID_REACH_M + 1e-6:
        return f"that spot on {loc.name} is too deep for a humanoid to reach from the edge"
    return None


# ---------------------------------------------------------------- walking routes
# The building has racks across most of it, so a straight line between two places usually goes through steel. A
# route is A* on a 0.1 m occupancy grid of everything solid on the floor (racks, conveyor, table, pallets, the arm's
# pedestal, the walls), grown by the humanoid's body radius, then shortened to the fewest straight legs that keep
# that clearance. Start and goal are usually work spots right at an edge (inside the grown zone): the route backs
# straight out to free floor first and comes straight in last.

GRID_M = 0.1


def obstacles() -> list[tuple[float, float, float, float]]:
    """Footprints [x0, y0, x1, y1] of everything a humanoid cannot walk through."""
    out: list[tuple[float, float, float, float]] = []
    for sh in LAYOUT["shelves"].values():
        out.append(tuple(sh["rect"]))  # type: ignore[arg-type]
    out.append(CONVEYOR_RECT)  # type: ignore[arg-type]
    out.append(tuple(LAYOUT["pick_table"]["rect"]))  # type: ignore[arg-type]
    for p in LAYOUT["pallets"]:
        (cx, cy), (sx, sy) = p["center"], p["size"][:2]
        out.append((cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2))
    bx, by, _ = ARM_BASE
    hx, hy = ARM_PEDESTAL_HALF
    out.append((bx - hx, by - hy, bx + hx, by + hy))
    return out


@lru_cache(maxsize=1)
def _grid() -> tuple[int, int, bytearray]:
    nx, ny = int(round(CELL_SIZE[0] / GRID_M)), int(round(CELL_SIZE[1] / GRID_M))
    blocked = bytearray(nx * ny)
    r = HUMANOID_RADIUS_M
    for x0, y0, x1, y1 in [*obstacles()]:
        i0, i1 = max(0, int((x0 - r) / GRID_M)), min(nx - 1, int((x1 + r) / GRID_M))
        j0, j1 = max(0, int((y0 - r) / GRID_M)), min(ny - 1, int((y1 + r) / GRID_M))
        for j in range(j0, j1 + 1):
            row = j * nx
            for i in range(i0, i1 + 1):
                blocked[row + i] = 1
    wall = int(math.ceil(r / GRID_M))
    for j in range(ny):
        for i in range(nx):
            if i < wall - 2 or j < wall or i >= nx - wall or j >= ny - wall:  # the west strip by the dock is tight
                blocked[j * nx + i] = 1
    return nx, ny, blocked


def _cell(x: float, y: float) -> tuple[int, int]:
    nx, ny, _ = _grid()
    return min(nx - 1, max(0, int(x / GRID_M))), min(ny - 1, max(0, int(y / GRID_M)))


def _free(i: int, j: int) -> bool:
    nx, ny, blocked = _grid()
    return 0 <= i < nx and 0 <= j < ny and not blocked[j * nx + i]


def _nearest_free(x: float, y: float, yaw: float | None) -> tuple[int, int] | None:
    """The free cell to step to first: straight back along -yaw when that works (backing out of a work spot), else
    the nearest free cell."""
    if yaw is not None:
        for k in range(1, 15):
            d = k * GRID_M
            c = _cell(x - d * math.cos(yaw), y - d * math.sin(yaw))
            if _free(*c):
                return c
    ci, cj = _cell(x, y)
    for rad in range(0, 20):
        best = None
        for i in range(ci - rad, ci + rad + 1):
            for j in range(cj - rad, cj + rad + 1):
                if max(abs(i - ci), abs(j - cj)) == rad and _free(i, j):
                    d = (i - ci) ** 2 + (j - cj) ** 2
                    if best is None or d < best[0]:
                        best = (d, (i, j))
        if best:
            return best[1]
    return None


def _line_free(a: tuple[int, int], b: tuple[int, int]) -> bool:
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1) * 2
    for k in range(n + 1):
        t = k / n
        if not _free(round(a[0] + (b[0] - a[0]) * t), round(a[1] + (b[1] - a[1]) * t)):
            return False
    return True


def _astar(s: tuple[int, int], g: tuple[int, int]) -> list[tuple[int, int]] | None:
    if s == g:
        return [s]
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    openq: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, s)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {s: 0.0}
    while openq:
        _, c, cur = heapq.heappop(openq)
        if cur == g:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if c > cost.get(cur, math.inf):
            continue
        for di, dj, w in steps:
            nxt = (cur[0] + di, cur[1] + dj)
            if not _free(*nxt):
                continue
            nc = c + w
            if nc < cost.get(nxt, math.inf):
                cost[nxt] = nc
                came[nxt] = cur
                h = math.hypot(g[0] - nxt[0], g[1] - nxt[1])
                heapq.heappush(openq, (nc + h, nc, nxt))
    return None


def route(start: tuple[float, float, float], goal: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    """Waypoints (x, y, yaw) from `start` to `goal` (both x, y, yaw), ending with `goal` itself. A straight walk when
    nothing is in the way; [goal] when no route exists (the walk then goes straight and the physics decides)."""
    sx, sy, syaw = start
    gx, gy, gyaw = goal
    s0, g0 = _cell(sx, sy), _cell(gx, gy)
    if _free(*s0) and _free(*g0) and _line_free(s0, g0):
        return [goal]
    s = s0 if _free(*s0) else _nearest_free(sx, sy, syaw)
    g = g0 if _free(*g0) else _nearest_free(gx, gy, gyaw)
    if s is None or g is None:
        return [goal]
    path = _astar(s, g)
    if path is None:
        return [goal]
    # string pulling: keep only the corners a straight walk with clearance cannot cut
    keep = [path[0]]
    k = 0
    while k < len(path) - 1:
        far = len(path) - 1
        while far > k + 1 and not _line_free(path[k], path[far]):
            far -= 1
        keep.append(path[far])
        k = far
    pts = [((i + 0.5) * GRID_M, (j + 0.5) * GRID_M) for i, j in keep]
    if s == s0:
        pts = pts[1:]  # already there; when the start was blocked, the first point is the step back out
    out: list[tuple[float, float, float]] = []
    prev = (sx, sy)
    targets = [*pts, (gx, gy)]
    for n, (x, y) in enumerate(targets):
        if math.hypot(x - prev[0], y - prev[1]) < 0.15 and n < len(targets) - 1:
            continue
        nxt = targets[n + 1] if n + 1 < len(targets) else None
        if nxt is None:
            out.append(goal)
        else:
            out.append((round(x, 3), round(y, 3), math.atan2(nxt[1] - y, nxt[0] - x)))
        prev = (x, y)
    return out
