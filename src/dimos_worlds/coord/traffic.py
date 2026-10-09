"""Right of way for robots that each run their own planner in one shared world.

Each robot keeps its own planner. The coordinator sits between whoever sends goals and those planners: it takes the
goal, forwards it, reads back the path the planner is following, and compares every pair of robots' remaining paths.
When two remaining paths come closer than the two bodies need to pass (radius + radius + margin) at a spot both
robots reach within `horizon_m`, that pair has a conflict and one robot gets the right of way:

  1. A robot with no goal (parked, or settling at its goal) never keeps the right of way. If it stands on an active
     robot's path it is sent to a pocket and comes back afterwards.
  2. A robot already standing on the other's path while the other is not standing on its path goes first (it clears
     the spot by moving on: a crossing, or a robot being followed). The other holds.
  3. Both on each other's path (head-on in an aisle): the lower priority robot yields to a pocket, the nearest free
     spot off the winner's whole remaining path that it can reach without passing the winner. No pocket: the other
     robot yields instead.
  4. Neither yet on the other's path (approaching a crossing): the one that reaches the shared spot first by a clear
     margin goes, otherwise the higher priority. The other holds.

A holding robot keeps walking until it is `stop_margin_m` (along its path) before the shared stretch, then stops
(its planner's goal is cancelled) and is given its goal again once the stretch has moved away or cleared. A yielding
robot is given the pocket as a goal, waits there, and is given its own goal back once the winner's remaining path no
longer comes near its own. Nobody resends goals by hand.

Deadlock handling: a planner that gives up (its goal_reached says False) on a goal the coordinator still holds is
given the goal again after `retry_s`. If the robot with the right of way has not moved `progress_m` in `deadlock_s`
while the other waits, the right of way is swapped. A robot with a goal and no conflict that has not moved in
`deadlock_s` is given its goal again.

Arrival: when a planner reports its robot arrived at the goal, a robot with a `SettleConfig` closes the remaining
position and heading error itself (velocity commands, see SettleConfig) before the arrival is reported. DimOS's planner
accepts arrival within 0.2 m and 15-20 degrees, or within 0.5 m when a replan lands near the goal.

Pure numpy (+ scipy for distance transforms): no DimOS imports, so the logic runs and is tested without a running
blueprint. coord.module wraps it as a DimOS module; coord.inproc runs it against DimOS planners in one process.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
import logging
import math
from typing import Literal

import numpy as np

log = logging.getLogger("dimos_worlds.coord.traffic")

Mode = Literal["idle", "go", "hold", "yield", "settle"]
STEP_M = 0.2  # paths are compared at this spacing


@dataclass(frozen=True)
class SettleConfig:
    """Closing the last bit of pose error after the planner reports arrival, with velocity commands in the robot's
    body frame. Tolerances are what counts as arrived; within them for `hold_s` and the robot is stopped."""

    pos_tol: float = 0.12  # m
    yaw_tol: float = math.radians(6.0)
    translate: bool = True  # False: heading only (a body whose small forward commands are inside a dead band)
    k_pos: float = 0.8  # 1/s
    k_yaw: float = 1.2  # 1/s
    max_v: float = 0.15  # m/s
    max_wz: float = 0.35  # rad/s
    min_wz: float = 0.0  # smallest turn command that turns the body at all (0: none needed)
    hold_s: float = 0.6
    timeout_s: float = 20.0


@dataclass(frozen=True)
class Agent:
    id: str
    radius: float  # m, footprint for clearance checks
    priority: int = 0  # higher keeps the right of way in a tie
    speed: float = 0.5  # m/s, nominal, for arrival estimates only
    settle: SettleConfig | None = None


@dataclass(frozen=True)
class TrafficConfig:
    margin_m: float = 0.3  # two bodies pass when their centres are radius + radius + this apart
    horizon_m: float = 8.0  # a shared spot further along either path than this is not a conflict yet
    stop_margin_m: float = 1.0  # a holding robot stops this far before the shared stretch ...
    release_m: float = 0.8  # ... and walks on once it is this much further away again
    eta_tie_s: float = 3.0  # arrival times closer than this are a tie (priority decides)
    pocket_margin_m: float = 0.25  # a pocket keeps radius + radius + this from the winner's remaining path
    pocket_wall_m: float = 0.08  # ... and its own radius + this from anything occupied
    pocket_max_m: float = 25.0  # how far (path length) a robot may be sent to a pocket
    clear_extra_m: float = 0.15  # a conflict is over once the paths are this much further apart ...
    clear_s: float = 1.5  # ... for this long
    retry_s: float = 3.0  # a planner that gave up gets the goal again after this
    stale_s: float = 1.0  # planner results this soon after a command from here are its answer to that command
    deadlock_s: float = 25.0
    progress_m: float = 0.3
    arrive_m: float = 0.6  # a planner arrival this close to the goal counts (further: the goal is sent again)


@dataclass(frozen=True)
class Goal:
    robot: str
    x: float
    y: float
    yaw: float
    reason: str = ""


@dataclass(frozen=True)
class Stop:
    """Stand here: give the planner its robot's own pose as the goal. (Cancelling a DimOS planner's goal from
    outside races its monitor thread: a stop message it is handling then finds no goal and the thread dies.)"""

    robot: str
    x: float
    y: float
    yaw: float
    reason: str = ""


@dataclass(frozen=True)
class Velocity:
    robot: str
    vx: float
    vy: float
    wz: float


@dataclass(frozen=True)
class Arrived:
    robot: str
    ok: bool


Action = Goal | Stop | Velocity | Arrived


@dataclass
class Grant:
    winner: str
    loser: str
    kind: Literal["hold", "yield"]
    since: float
    note: str = ""
    clear_t: float | None = None  # since when the conflict has looked clear


@dataclass
class _Robot:
    agent: Agent
    pose: tuple[float, float, float] | None = None
    goal: tuple[float, float, float] | None = None  # what was asked for; None when there is nothing to do
    home: tuple[float, float, float] | None = None  # where it parked last (it returns here after yielding)
    mode: Mode = "idle"
    intent: np.ndarray | None = None  # the planner's path to `goal`, resampled to STEP_M
    pocket: tuple[float, float, float] | None = None
    pocket_fails: int = 0
    last_cmd_t: float = -1e9
    retry_t: float | None = None
    moved_t: float = 0.0
    kick_t: float = -1e9
    moved_xy: tuple[float, float] | None = None
    settle_t: float = 0.0
    settle_ok_t: float | None = None
    returning: bool = False  # the goal is `home`, after a yield while parked


def _resample(pts: np.ndarray, step: float = STEP_M) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-9:
        return pts[:1]
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)


def _remaining(path: np.ndarray | None, xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Points from the robot's position along the rest of `path` (from its closest point on), and arc lengths."""
    here = np.array([xy], dtype=float)
    if path is None or len(path) == 0:
        return here, np.zeros(1)
    i = int(np.argmin(np.linalg.norm(path - here, axis=1)))
    pts = np.concatenate([here, path[i:]], axis=0)
    if np.linalg.norm(pts[1] - pts[0]) > STEP_M if len(pts) > 1 else False:
        pts = np.concatenate([_resample(pts[:2]), pts[2:]], axis=0)  # off the path (a pocket): fill the way back
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return pts, s


@dataclass
class Overlap:
    near: np.ndarray  # bool (len(a), len(b)): points closer than the clearance
    a_inside: bool  # a's position is on b's remaining path
    b_inside: bool
    a_entry: float  # arc along a to its first point near b's path
    b_entry: float
    relevant: bool  # both reach a shared spot within the horizon


def overlap(pa: np.ndarray, sa: np.ndarray, pb: np.ndarray, sb: np.ndarray, clearance: float,
            horizon: float) -> Overlap | None:
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    near = d < clearance
    if not near.any():
        return None
    rows = np.flatnonzero(near.any(axis=1))
    cols = np.flatnonzero(near.any(axis=0))
    rel = near[np.ix_(sa <= horizon, sb <= horizon)].any()
    return Overlap(near=near, a_inside=bool(near[0].any()), b_inside=bool(near[:, 0].any()),
                   a_entry=float(sa[rows[0]]), b_entry=float(sb[cols[0]]), relevant=bool(rel))


class Traffic:
    """The coordinator's state machine. Feed it poses, goals, planner paths and planner results; call step()."""

    def __init__(self, agents: list[Agent] | tuple[Agent, ...], config: TrafficConfig | None = None) -> None:
        ids = [a.id for a in agents]
        if len(set(ids)) != len(ids):
            raise ValueError(f"robot ids must be unique: {ids}")
        self.cfg = config or TrafficConfig()
        self.robots: dict[str, _Robot] = {a.id: _Robot(agent=a) for a in agents}
        self.grants: dict[frozenset[str], Grant] = {}
        self.events: deque[tuple[float, str]] = deque(maxlen=500)
        self.on_event: Callable[[float, str], None] | None = None  # e.g. a logger, for every decision
        self._grid: np.ndarray | None = None  # int8, row = y; 0 free
        self._origin = (0.0, 0.0)
        self._res = 0.1
        self._clear_m: np.ndarray | None = None  # distance to the nearest non-free cell

    # ------------------------------------------------------------------ inputs
    def set_map(self, grid: np.ndarray, origin_x: float, origin_y: float, resolution: float) -> None:
        """Static occupancy (0 free, anything else blocked) for finding pockets."""
        from scipy.ndimage import distance_transform_edt

        g = np.asarray(grid)
        if self._grid is not None and g.shape == self._grid.shape and np.array_equal(g, self._grid) \
                and (origin_x, origin_y) == self._origin and resolution == self._res:
            return
        self._grid = np.array(g, copy=True)
        self._origin = (float(origin_x), float(origin_y))
        self._res = float(resolution)
        # centre-to-centre distances, less half a cell: a conservative distance to the obstacle's edge
        self._clear_m = distance_transform_edt(self._grid == 0) * self._res - 0.5 * self._res

    def update_pose(self, rid: str, x: float, y: float, yaw: float, now: float) -> None:
        r = self.robots[rid]
        r.pose = (float(x), float(y), float(yaw))
        if r.moved_xy is None or math.hypot(x - r.moved_xy[0], y - r.moved_xy[1]) >= self.cfg.progress_m:
            r.moved_xy = (float(x), float(y))
            r.moved_t = now

    def request_goal(self, rid: str, x: float, y: float, yaw: float, now: float) -> list[Action]:
        r = self.robots[rid]
        r.goal = (float(x), float(y), float(yaw))
        r.home = None
        r.returning = False
        r.intent = None
        r.retry_t = None
        r.moved_t = now
        self._event(now, f"{rid}: goal ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f} deg)")
        out: list[Action] = []
        if r.mode == "settle":
            out.append(Velocity(rid, 0.0, 0.0, 0.0))
        # a loser keeps its role; step() sends what it should do
        r.mode = "go" if not self._loses(rid) else r.mode
        if r.mode == "go":
            out.append(self._send_goal(r, now, "requested"))
        elif r.mode == "hold":
            out.append(self._send_goal(r, now, "requested"))
            r.mode = "go"  # re-evaluated on the next step (a hold stops it again near the shared stretch)
        return out

    def on_path(self, rid: str, pts: np.ndarray, now: float) -> None:
        r = self.robots[rid]
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(pts) == 0 or r.mode != "go" or r.goal is None:
            return  # an empty path is a planner cancelling; a path to a pocket is not the robot's intent
        if math.hypot(pts[-1, 0] - r.goal[0], pts[-1, 1] - r.goal[1]) > 1.5:
            return  # a path to some other goal (a stale pocket plan)
        r.intent = _resample(pts)

    def on_goal_reached(self, rid: str, ok: bool, now: float) -> list[Action]:
        r = self.robots[rid]
        fresh = now - r.last_cmd_t < self.cfg.stale_s
        if r.mode == "go" and r.goal is not None:
            if ok and r.pose is not None and math.hypot(r.pose[0] - r.goal[0], r.pose[1] - r.goal[1]) \
                    <= self.cfg.arrive_m:
                return self._arrive(r, now)
            if not fresh:
                r.retry_t = now + self.cfg.retry_s
                self._event(now, f"{rid}: planner gave up (or stopped short); goal again in {self.cfg.retry_s:.0f} s")
        elif r.mode == "yield" and r.pocket is not None:
            if ok:
                self._event(now, f"{rid}: in pocket ({r.pocket[0]:.2f}, {r.pocket[1]:.2f}), waiting")
            elif not fresh:
                r.pocket_fails += 1
                r.retry_t = now + self.cfg.retry_s
                self._event(now, f"{rid}: could not reach pocket (try {r.pocket_fails})")
        return []

    # ------------------------------------------------------------------ queries
    def status(self) -> dict[str, dict]:
        return {rid: {"mode": r.mode, "goal": r.goal, "pocket": r.pocket, "home": r.home}
                for rid, r in self.robots.items()} | {
            "grants": [f"{g.winner}>{g.loser}:{g.kind}" for g in self.grants.values()]}

    # ------------------------------------------------------------------ the loop
    def step(self, now: float) -> list[Action]:
        out: list[Action] = []
        for r in self.robots.values():
            if r.mode == "settle":
                out += self._settle(r, now)
        self._maintain_grants(now)
        self._detect(now)
        self._deadlocks(now)
        out += self._apply(now)
        return out

    # ---- occupancy as others see it
    def _active(self, r: _Robot) -> bool:
        return r.mode in ("go", "hold") and r.goal is not None

    def _occupancy(self, r: _Robot) -> tuple[np.ndarray, np.ndarray]:
        assert r.pose is not None
        if self._active(r):
            return self._wanted(r)
        return _remaining(None, r.pose[:2])

    def _wanted(self, r: _Robot) -> tuple[np.ndarray, np.ndarray]:
        """Where the robot wants to go from here: the rest of its intent (or home / goal while it has no plan)."""
        assert r.pose is not None
        if r.intent is not None and r.goal is not None:
            return _remaining(r.intent, r.pose[:2])
        target = r.goal or r.home
        if target is None:
            return _remaining(None, r.pose[:2])
        return _remaining(_resample(np.array([r.pose[:2], target[:2]])), r.pose[:2])

    def _clearance(self, a: _Robot, b: _Robot) -> float:
        return a.agent.radius + b.agent.radius + self.cfg.margin_m

    def _loses(self, rid: str) -> bool:
        return any(g.loser == rid for g in self.grants.values())

    # ---- grants
    def _maintain_grants(self, now: float) -> None:
        for key, g in list(self.grants.items()):
            w, l = self.robots[g.winner], self.robots[g.loser]
            if w.pose is None or l.pose is None:
                continue
            if not self._active(w):
                self._drop(key, now, f"{g.winner} no longer moving")
                continue
            pw, sw = self._occupancy(w)
            pl, sl = self._wanted(l)
            far = self._clearance(l, w) + self.cfg.clear_extra_m
            ov = overlap(pl, sl, pw, sw, far, self.cfg.horizon_m)
            body_near = math.dist(w.pose[:2], l.pose[:2]) < far
            if (ov is None or not ov.relevant) and not body_near:
                if g.clear_t is None:
                    g.clear_t = now
                if now - g.clear_t >= self.cfg.clear_s:
                    self._drop(key, now, "clear")
                continue
            g.clear_t = None
            if g.kind == "hold" and ov is not None and ov.a_inside and ov.b_inside and l.mode != "hold":
                g.kind = "yield"  # the winner came onto the loser's spot: holding cannot clear it any more
                self._event(now, f"{g.loser}: now yields to {g.winner} (both on each other's path)")

    def _drop(self, key: frozenset[str], now: float, why: str) -> None:
        g = self.grants.pop(key)
        self._event(now, f"right of way {g.winner} over {g.loser} ended: {why}")

    def _detect(self, now: float) -> None:
        ids = sorted(self.robots)
        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                key = frozenset((a_id, b_id))
                if key in self.grants:
                    continue
                a, b = self.robots[a_id], self.robots[b_id]
                if a.pose is None or b.pose is None or not (self._active(a) or self._active(b)):
                    continue
                if (a.mode == "yield" and not self._active(b)) or (b.mode == "yield" and not self._active(a)):
                    continue
                pa, sa = self._occupancy(a) if self._active(a) else self._wanted(a)
                pb, sb = self._occupancy(b) if self._active(b) else self._wanted(b)
                if not self._active(a):
                    pa, sa = pa[:1], sa[:1]
                if not self._active(b):
                    pb, sb = pb[:1], sb[:1]
                ov = overlap(pa, sa, pb, sb, self._clearance(a, b), self.cfg.horizon_m)
                if ov is None or not ov.relevant:
                    continue
                self._grant(key, a, b, ov, now)

    def _grant(self, key: frozenset[str], a: _Robot, b: _Robot, ov: Overlap, now: float) -> None:
        cfg = self.cfg
        act_a, act_b = self._active(a), self._active(b)
        if act_a != act_b:
            w, l, kind, why = (a, b, "yield", "parked on its path") if act_a else (b, a, "yield", "parked on its path")
        elif ov.a_inside != ov.b_inside:
            w, l = (a, b) if ov.a_inside else (b, a)
            kind, why = "hold", "already on the other's path"
        elif ov.a_inside and ov.b_inside:
            w, l = (a, b) if a.agent.priority >= b.agent.priority else (b, a)
            kind, why = "yield", "head-on, lower priority yields"
            if self._find_pocket(l, w) is None:
                if self._find_pocket(w, l) is not None:
                    w, l = l, w
                    why = "head-on, no pocket for the lower priority robot"
                else:
                    kind, why = "hold", "head-on, no pocket for either (hold, deadlock rules apply)"
        else:
            eta_a = ov.a_entry / max(a.agent.speed, 1e-3)
            eta_b = ov.b_entry / max(b.agent.speed, 1e-3)
            if abs(eta_a - eta_b) > cfg.eta_tie_s:
                w, l = (a, b) if eta_a < eta_b else (b, a)
                why = f"gets there first ({min(eta_a, eta_b):.0f} s vs {max(eta_a, eta_b):.0f} s)"
            else:
                w, l = (a, b) if a.agent.priority >= b.agent.priority else (b, a)
                why = "same arrival, priority"
            kind = "hold"
        self.grants[key] = Grant(winner=w.agent.id, loser=l.agent.id, kind=kind, since=now, note=why)
        self._event(now, f"right of way {w.agent.id} over {l.agent.id} ({kind}): {why}")

    def _deadlocks(self, now: float) -> None:
        cfg = self.cfg
        for key, g in list(self.grants.items()):
            w, l = self.robots[g.winner], self.robots[g.loser]
            stalled = now - max(w.moved_t, g.since) > cfg.deadlock_s
            pocket_dead = g.kind == "yield" and l.pocket_fails >= 3
            if not (stalled or pocket_dead):
                continue
            if w.pose is None or l.pose is None:
                continue
            pw, sw = self._wanted(w)
            pl, sl = self._occupancy(l) if self._active(l) else self._wanted(l)
            ov = overlap(pw, sw, pl, sl, self._clearance(w, l), 1e9)
            kind: Literal["hold", "yield"] = "yield" if (ov is not None and ov.a_inside) else "hold"
            self.grants[key] = Grant(winner=g.loser, loser=g.winner, kind=kind, since=now, note="deadlock swap")
            l.pocket_fails = 0
            if l.mode in ("yield", "hold"):  # the new winner walks on at once (its goal again, or home)
                l.pocket = None
                if l.goal is None and l.home is not None:
                    l.goal, l.returning = l.home, True
                l.mode = "go" if l.goal is not None else "idle"
                l.retry_t = now
            w.moved_t = l.moved_t = now
            why = "winner made no progress" if stalled else "no reachable pocket"
            self._event(now, f"deadlock ({why}): right of way swapped, {g.loser} over {g.winner} ({kind})")
        for r in self.robots.values():
            kick_s = 0.5 * cfg.deadlock_s
            if r.mode == "go" and r.goal is not None and not self._loses(r.agent.id) \
                    and now - max(r.moved_t, r.kick_t) > kick_s and now - r.last_cmd_t > kick_s:
                r.kick_t = now
                r.retry_t = now  # kick its planner
                self._event(now, f"{r.agent.id}: no progress for {kick_s:.0f} s with a goal; goal again")

    # ---- what each robot should be doing
    def _apply(self, now: float) -> list[Action]:
        out: list[Action] = []
        cfg = self.cfg
        for rid, r in self.robots.items():
            if r.pose is None:
                continue
            mine = [g for g in self.grants.values() if g.loser == rid]
            yield_to = next((g for g in mine if g.kind == "yield"), None)
            if yield_to is not None:
                if r.mode != "yield":
                    if r.mode == "settle":
                        out.append(Velocity(rid, 0.0, 0.0, 0.0))
                    if r.goal is None:  # parked: come back here afterwards
                        r.home = r.home or r.pose
                    r.mode = "yield"
                    r.pocket = None
                    r.pocket_fails = 0
                if r.pocket is None or (r.retry_t is not None and now >= r.retry_t):
                    r.retry_t = None
                    w = self.robots[yield_to.winner]
                    pocket = self._find_pocket(r, w, skip=r.pocket if r.pocket_fails else None)
                    if pocket is None:
                        if r.pocket is None:
                            out.append(Stop(rid, *r.pose, reason=f"no pocket from {yield_to.winner}"))
                            r.last_cmd_t = now
                            r.pocket = (*r.pose[:2], r.pose[2])
                        continue
                    r.pocket = pocket
                    self._event(now, f"{rid}: yields to {yield_to.winner} at pocket ({pocket[0]:.2f}, {pocket[1]:.2f})")
                    out.append(Goal(rid, *pocket, reason=f"yield to {yield_to.winner}"))
                    r.last_cmd_t = now
                continue
            hold = False
            for g in mine:
                w = self.robots[g.winner]
                pw, sw = self._occupancy(w)
                pl, sl = self._wanted(r)
                ov = overlap(pl, sl, pw, sw, self._clearance(r, w), self.cfg.horizon_m)
                if ov is None:
                    continue
                if r.mode == "hold":
                    hold = hold or ov.a_entry < cfg.stop_margin_m + cfg.release_m
                else:
                    hold = hold or ov.a_entry < cfg.stop_margin_m
            if r.mode == "yield":  # released
                r.pocket = None
                if r.goal is None and r.home is not None:
                    r.goal = r.home
                    r.returning = True
                r.mode = "go" if r.goal is not None else "idle"
                if r.mode == "go":
                    r.intent = None
                    out.append(self._send_goal(r, now, "released from pocket"))
            if hold and r.mode == "go":
                r.mode = "hold"
                out.append(Stop(rid, *r.pose, reason="holds for right of way"))
                r.last_cmd_t = now
                self._event(now, f"{rid}: holds")
            elif not hold and r.mode == "hold":
                r.mode = "go" if r.goal is not None else "idle"
                if r.mode == "go":
                    out.append(self._send_goal(r, now, "right of way clear"))
                    self._event(now, f"{rid}: walks on")
            elif r.mode == "go" and r.retry_t is not None and now >= r.retry_t:
                r.retry_t = None
                out.append(self._send_goal(r, now, "retry"))
                self._event(now, f"{rid}: goal sent again (retry)")
        return out

    def _send_goal(self, r: _Robot, now: float, reason: str) -> Goal:
        assert r.goal is not None
        r.last_cmd_t = now
        r.retry_t = None
        return Goal(r.agent.id, *r.goal, reason=reason)

    def _arrive(self, r: _Robot, now: float) -> list[Action]:
        if r.agent.settle is not None:
            r.mode = "settle"
            r.settle_t = now
            r.settle_ok_t = None
            self._event(now, f"{r.agent.id}: planner arrived; settling")
            return []
        return self._finish(r, now)

    def _finish(self, r: _Robot, now: float) -> list[Action]:
        rid = r.agent.id
        r.home = r.goal
        r.goal = None
        r.intent = None
        r.mode = "idle"
        returning, r.returning = r.returning, False
        self._event(now, f"{rid}: {'back home' if returning else 'arrived'}")
        return [] if returning else [Arrived(rid, True)]

    # ---- settling at the goal
    def _settle(self, r: _Robot, now: float) -> list[Action]:
        sc = r.agent.settle
        assert sc is not None and r.goal is not None and r.pose is not None
        x, y, yaw = r.pose
        gx, gy, gyaw = r.goal
        cmd = settle_command(sc, (x, y, yaw), (gx, gy, gyaw))
        done = cmd is None
        if done:
            if r.settle_ok_t is None:
                r.settle_ok_t = now
        else:
            r.settle_ok_t = None
        if (r.settle_ok_t is not None and now - r.settle_ok_t >= sc.hold_s) or now - r.settle_t > sc.timeout_s:
            if r.settle_ok_t is None:
                self._event(now, f"{r.agent.id}: settle timed out")
            return [Velocity(r.agent.id, 0.0, 0.0, 0.0), *self._finish(r, now)]
        return [Velocity(r.agent.id, *(cmd or (0.0, 0.0, 0.0)))]

    # ---- pockets
    def _find_pocket(self, r: _Robot, w: _Robot,
                     skip: tuple[float, float, float] | None = None) -> tuple[float, float, float] | None:
        """Nearest free spot (path length from r) off w's whole remaining path, reached without passing w."""
        if self._grid is None or self._clear_m is None or r.pose is None or w.pose is None:
            return None
        from scipy.ndimage import distance_transform_edt

        cfg = self.cfg
        res = self._res
        ox, oy = self._origin
        h, wd = self._grid.shape
        pw, _ = self._wanted(w) if w.goal is not None else _remaining(None, w.pose[:2])
        mark = np.ones((h, wd), dtype=bool)
        ij = np.floor((pw - [ox, oy]) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < wd) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
        mark[ij[ok, 1], ij[ok, 0]] = False
        to_path = distance_transform_edt(mark) * res - 0.5 * res
        need = r.agent.radius + w.agent.radius + cfg.pocket_margin_m
        body = r.agent.radius + cfg.pocket_wall_m
        passable = self._clear_m >= r.agent.radius
        ys, xs = np.mgrid[0:h, 0:wd]
        cx = ox + (xs + 0.5) * res
        cy = oy + (ys + 0.5) * res
        passable &= np.hypot(cx - w.pose[0], cy - w.pose[1]) >= r.agent.radius + w.agent.radius
        target = (self._clear_m >= body) & (to_path >= need) & passable
        if skip is not None:
            target &= np.hypot(cx - skip[0], cy - skip[1]) > 0.5
        si = int(math.floor((r.pose[0] - ox) / res))
        sj = int(math.floor((r.pose[1] - oy) / res))
        if not (0 <= si < wd and 0 <= sj < h):
            return None
        # Dijkstra-lite: BFS on 8-neighbours, cost in metres, starting cell always allowed
        dist = np.full((h, wd), np.inf)
        dist[sj, si] = 0.0
        frontier = [(si, sj)]
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        found: list[tuple[float, int, int]] = []
        limit = cfg.pocket_max_m
        while frontier:
            nxt: list[tuple[int, int]] = []
            for i, j in frontier:
                d0 = dist[j, i]
                if target[j, i]:
                    found.append((d0, i, j))
                    continue
                for di, dj, c in steps:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < wd and 0 <= nj < h and passable[nj, ni]:
                        nd = d0 + c * res
                        if nd < dist[nj, ni] and nd <= limit:
                            dist[nj, ni] = nd
                            nxt.append((ni, nj))
            if found:
                break
            frontier = nxt
        if not found:
            return None
        found.sort()
        _, i, j = found[0]
        # walk a little further into the pocket: the spot with the most clearance among the nearby targets
        near = target & (np.hypot(cx - cx[j, i], cy - cy[j, i]) < 0.6)
        score = np.where(near, self._clear_m + 0.5 * np.minimum(to_path, need + 0.6), -1.0)
        j2, i2 = np.unravel_index(int(np.argmax(score)), score.shape)
        return float(cx[j2, i2]), float(cy[j2, i2]), float(r.pose[2])

    def _event(self, now: float, text: str) -> None:
        self.events.append((now, text))
        log.info("traffic %.1f: %s", now, text)
        if self.on_event is not None:
            self.on_event(now, text)


def settle_command(sc: SettleConfig, pose: tuple[float, float, float],
                   goal: tuple[float, float, float]) -> tuple[float, float, float] | None:
    """Body-frame (vx, vy, wz) closing the pose error, or None when inside both tolerances."""
    x, y, yaw = pose
    gx, gy, gyaw = goal
    dx, dy = gx - x, gy - y
    dist = math.hypot(dx, dy)
    eyaw = (gyaw - yaw + math.pi) % (2 * math.pi) - math.pi
    pos_ok = dist <= sc.pos_tol or not sc.translate
    yaw_ok = abs(eyaw) <= sc.yaw_tol
    if pos_ok and yaw_ok:
        return None
    vx = vy = wz = 0.0
    if not pos_ok:
        c, s = math.cos(yaw), math.sin(yaw)
        ex, ey = c * dx + s * dy, -s * dx + c * dy
        scale = min(1.0, sc.max_v / max(sc.k_pos * dist, 1e-9))
        vx, vy = sc.k_pos * ex * scale, sc.k_pos * ey * scale
    if not yaw_ok:
        wz = float(np.clip(sc.k_yaw * eyaw, -sc.max_wz, sc.max_wz))
        if sc.min_wz:
            wz = math.copysign(max(abs(wz), sc.min_wz), wz)
    return vx, vy, wz
