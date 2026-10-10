"""DimOS's GlobalPlanner (and the LocalPlanner inside it) without their threads, stepped on sim time.

In a live blueprint the planner runs two threads on the wall clock: a local loop at 10 Hz (path clearance, state
machine, cmd_vel) and a monitor at 10 Hz (arrival, path deviation, stuck detection, replanning). Run against a
simulation on the same machine, how far the world gets between two planner iterations depends on how busy the
machine is, so the same scenario turned out differently on a fast laptop and a 2-vCPU CI runner.

`SteppedPlanner` keeps every decision DimOS's planner makes (safe goal, A*, path resampling, PathClearance,
PathDistancer, the P controller, ReplanLimiter, the stop/replan handling) and only replaces the scheduling: the
caller runs one local iteration and one monitor iteration per `step()`, every 0.1 s of sim time, from the same loop
that ticks the world. The two loop bodies mirror `LocalPlanner._loop` and `GlobalPlanner._thread_entrypoint`
(DimOS 0.0.14; unchanged on DimOS main as of 2026-10). Two clocks are swapped for sim time: the monitor's stuck
check (perf_counter) and the position tracker (time.time() stored as float32, which in a live run quantizes
timestamps to 128 s; here a timestamp is the sim time it was taken at).

The live `dimos run` path does not use this module.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
import importlib
import logging
from typing import Any

import numpy as np

log = logging.getLogger("dimos_worlds.coord.stepped")

PERIOD_S = 0.1  # LocalPlanner._control_frequency and the monitor's wait timeout, both 10 Hz


def nav_module(name: str) -> Any:
    """A module of DimOS's replanning A* planner, wherever this DimOS keeps it (moved under go2/ after 0.0.14)."""
    try:
        return importlib.import_module(f"dimos.navigation.replanning_a_star.{name}")
    except ModuleNotFoundError:
        return importlib.import_module(f"dimos.navigation.go2.replanning_a_star.{name}")


class SimPositionTracker:
    """PositionTracker's interface (add_position, reset_data, is_stuck) on a caller's clock."""

    def __init__(self, time_window: float, threshold: float, clock: Callable[[], float]) -> None:
        self._time_window = time_window
        self._threshold = threshold
        self._clock = clock
        self._points: deque[tuple[float, float, float]] = deque()

    def reset_data(self) -> None:
        self._points.clear()

    def add_position(self, pose: Any) -> None:
        now = self._clock()
        self._points.append((now, float(pose.position.x), float(pose.position.y)))
        while self._points and self._points[0][0] < now - self._time_window:
            self._points.popleft()

    def is_stuck(self) -> bool:
        cutoff = self._clock() - self._time_window
        recent = np.array([(x, y) for t, x, y in self._points if t >= cutoff], dtype=float).reshape(-1, 2)
        if len(recent) == 0:
            return False
        distances = np.linalg.norm(recent - recent.mean(axis=0), axis=1)
        return bool(np.all(distances < self._threshold))


class SteppedPlanner:
    """Wraps one DimOS GlobalPlanner (never `start()`ed) so `step()` runs its two loops once each."""

    def __init__(self, planner: Any, clock: Callable[[], float]) -> None:
        self.p = planner
        self.lp = planner._local_planner  # noqa: SLF001 - see the module docstring
        self.clock = clock
        old = planner._position_tracker  # noqa: SLF001
        planner._position_tracker = SimPositionTracker(old._time_window, old._threshold, clock)  # noqa: SLF001
        # The local planner's own start_planning spawns its loop thread; this one sets up the same state only.
        self.lp.start_planning = self._start_planning
        self._sub = self.lp.stopped_navigating.subscribe(planner._on_stopped_navigating)  # noqa: SLF001
        self._path_clearance = nav_module("path_clearance").PathClearance
        self._path_distancer = nav_module("path_distancer").PathDistancer
        self._last_id = -1
        self._last_stuck_check = clock()

    def close(self) -> None:
        self._sub.dispose()
        self.p.stop()

    def step(self) -> None:
        self._local_step()
        self._monitor_step()

    # ---- LocalPlanner
    def _start_planning(self, path: Any) -> None:
        """LocalPlanner.start_planning and the set-up at the top of LocalPlanner._loop, without the thread."""
        from dimos.utils.trigonometry import angle_diff

        lp = self.lp
        lp.stop_planning()
        with lp._lock:  # noqa: SLF001
            lp._path = path  # noqa: SLF001
            lp._path_clearance = self._path_clearance(lp._global_config, path)  # noqa: SLF001
            lp._path_distancer = self._path_distancer(path)  # noqa: SLF001
            lp._pose_index = 0  # noqa: SLF001
            odom = lp._current_odom  # noqa: SLF001
            new_state = "initial_rotation"
            if odom is not None and len(path.poses) > 0:
                yaw_error = angle_diff(path.poses[0].orientation.euler[2], odom.orientation.euler[2])
                lp._controller.reset_yaw_error(yaw_error)  # noqa: SLF001
                if abs(yaw_error) < lp._orientation_tolerance:  # noqa: SLF001
                    at_start = path.poses[0].position.distance(odom.position) < 0.01
                    new_state = "final_rotation" if at_start else "path_following"
            lp._change_state(new_state)  # noqa: SLF001

    def _local_step(self) -> None:
        """One pass of LocalPlanner._loop; ending the loop runs what its thread's `finally` does."""
        lp = self.lp
        with lp._lock:  # noqa: SLF001
            path, clearance = lp._path, lp._path_clearance  # noqa: SLF001
        if path is None or clearance is None:
            return  # not following a path (idle, or stopped)
        try:
            with lp._lock:  # noqa: SLF001
                clearance.update_costmap(lp._navigation_map.binary_costmap)  # noqa: SLF001
                clearance.update_pose_index(lp._pose_index)  # noqa: SLF001
            if clearance.is_obstacle_ahead():
                log.info("local planner: obstacle ahead")
                lp.stopped_navigating.on_next("obstacle_found")
                self._end_local()
                return
            with lp._lock:  # noqa: SLF001
                state = lp._state  # noqa: SLF001
            if state == "arrived":
                lp.stopped_navigating.on_next("arrived")
                self._end_local()
                return
            compute = {"initial_rotation": lp._compute_initial_rotation,  # noqa: SLF001
                       "path_following": lp._compute_path_following,  # noqa: SLF001
                       "final_rotation": lp._compute_final_rotation}.get(state)  # noqa: SLF001
            cmd = compute() if compute is not None else None
            if cmd is not None:
                lp.cmd_vel.on_next(cmd)
        except Exception as e:  # LocalPlanner._thread_entrypoint: an error ends the loop and asks for a replan
            log.info("local planner error: %s", e)
            lp.stopped_navigating.on_next("error")
            self._end_local()

    def _end_local(self) -> None:
        from dimos.msgs.geometry_msgs.Twist import Twist

        self.lp._reset_state()  # noqa: SLF001
        self.lp.cmd_vel.on_next(Twist())

    # ---- GlobalPlanner monitor
    def _monitor_step(self) -> None:
        """One pass of GlobalPlanner._thread_entrypoint (sim time for the stuck check)."""
        from dimos.utils.trigonometry import angle_diff

        p = self.p
        now = self.clock()
        if p._replan_event.is_set():  # noqa: SLF001
            p._replan_event.clear()  # noqa: SLF001
            with p._lock:  # noqa: SLF001
                reason, p._replan_reason = p._replan_reason, None  # noqa: SLF001
            if reason is not None:
                p._handle_stop_message(reason)  # noqa: SLF001
                self._last_stuck_check = now
                return
        with p._lock:  # noqa: SLF001
            goal, odom = p._current_goal, p._current_odom  # noqa: SLF001
        if not goal or not odom:
            return
        if (goal.position.distance(odom.position) < p._goal_tolerance  # noqa: SLF001
                and abs(angle_diff(goal.orientation.euler[2], odom.orientation.euler[2]))
                < p._rotation_tolerance):  # noqa: SLF001
            log.info("planner: close enough to goal, accepting as arrived")
            p.cancel_goal(arrived=True)
            return
        deviation = self.lp.get_distance_to_path()
        if deviation is not None and deviation > p._max_path_deviation:  # noqa: SLF001
            log.info("planner: veered off the path (%.2f m), replanning", deviation)
            p._replan_path()  # noqa: SLF001
            self._last_stuck_check = now
            return
        _, new_id = self.lp.get_unique_state()
        if new_id != self._last_id:
            self._last_id = new_id
            self._last_stuck_check = now
            return
        if now - self._last_stuck_check > p._stuck_time_window and p._position_tracker.is_stuck():  # noqa: SLF001
            log.info("planner: stuck, replanning")
            p._replan_path()  # noqa: SLF001
            self._last_stuck_check = now
