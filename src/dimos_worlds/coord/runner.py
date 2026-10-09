"""Traffic on its own thread: inputs from any thread go through a queue, actions go out through callbacks.
Without the thread (`run_once`), a caller that steps a simulation drives the same queue on its own clock.

Planner callbacks can fire synchronously inside a call made from here (cancelling a DimOS planner's goal publishes
goal_reached on the caller's thread), so nothing is dispatched while the state machine is being updated and every
input, whatever thread it comes from, is handled on the coordinator's thread in arrival order.
"""
from __future__ import annotations

from collections.abc import Callable
import logging
import queue
import threading
import time
from typing import Any

import numpy as np

from dimos_worlds.coord.traffic import Action, Agent, Arrived, Goal, Stop, Traffic, TrafficConfig, Velocity

log = logging.getLogger("dimos_worlds.coord.runner")


class Coordinator:
    def __init__(self, agents: list[Agent] | tuple[Agent, ...], config: TrafficConfig | None = None, *,
                 on_goal: Callable[[Goal], None], on_stop: Callable[[Stop], None],
                 on_velocity: Callable[[Velocity], None], on_arrived: Callable[[Arrived], None] | None = None,
                 rate_hz: float = 10.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.traffic = Traffic(agents, config)
        self._handlers: dict[type, Callable[[Any], None]] = {
            Goal: on_goal, Stop: on_stop, Velocity: on_velocity, Arrived: on_arrived or (lambda a: None)}
        self._q: queue.Queue[tuple[str, tuple]] = queue.Queue()
        self._period = 1.0 / rate_hz
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.arrivals: list[tuple[float, str]] = []

    # ---- inputs (any thread)
    def pose(self, rid: str, x: float, y: float, yaw: float) -> None:
        self._q.put(("pose", (rid, x, y, yaw)))

    def goal(self, rid: str, x: float, y: float, yaw: float) -> None:
        self._q.put(("goal", (rid, x, y, yaw)))

    def path(self, rid: str, pts: np.ndarray) -> None:
        self._q.put(("path", (rid, np.asarray(pts, dtype=float))))

    def goal_reached(self, rid: str, ok: bool) -> None:
        self._q.put(("reached", (rid, bool(ok))))

    def costmap(self, grid: np.ndarray, origin_x: float, origin_y: float, resolution: float) -> None:
        self._q.put(("map", (grid, origin_x, origin_y, resolution)))

    # ---- loop
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="traffic", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        next_t = self._clock()
        while not self._stop.is_set():
            out: list[Action] = []
            timeout = max(0.0, next_t - self._clock())
            try:
                kind, args = self._q.get(timeout=timeout)
                out += self._handle(kind, args)
            except queue.Empty:
                pass
            out += self._drain()
            if self._clock() >= next_t:
                next_t += self._period
                out += self._step()
            self._dispatch(out)

    def run_once(self, *, step: bool = True, max_rounds: int = 100) -> None:
        """Synchronous use (no thread; the caller owns the clock): handle every queued input, step the rules if
        `step`, dispatch, then handle what the dispatch queued (a planner answering a goal) until nothing is left.
        The same order the loop above gives, without its timing."""
        out = self._drain()
        if step:
            out += self._step()
        self._dispatch(out)
        for _ in range(max_rounds):
            if self._q.empty():
                return
            self._dispatch(self._drain())
        log.warning("coordinator inputs still queued after %d rounds", max_rounds)

    def _drain(self) -> list[Action]:
        out: list[Action] = []
        while True:
            try:
                kind, args = self._q.get_nowait()
            except queue.Empty:
                return out
            out += self._handle(kind, args)

    def _step(self) -> list[Action]:
        try:
            return self.traffic.step(self._clock())
        except Exception:
            log.exception("traffic step failed")
            return []

    def _dispatch(self, out: list[Action]) -> None:
        for a in out:
            if isinstance(a, Arrived):
                self.arrivals.append((self._clock(), a.robot))
            try:
                self._handlers[type(a)](a)
            except Exception:
                log.exception("dispatching %s failed", a)

    def _handle(self, kind: str, args: tuple) -> list[Action]:
        t = self.traffic
        now = self._clock()
        if kind == "pose":
            t.update_pose(*args, now)
        elif kind == "goal":
            return t.request_goal(*args, now)
        elif kind == "path":
            t.on_path(*args, now)
        elif kind == "reached":
            return t.on_goal_reached(*args, now)
        elif kind == "map":
            t.set_map(*args)
        return []
