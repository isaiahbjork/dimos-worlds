"""DimOS modules for the lot: prior map, named places, events.

* `LotPriorMap`  publishes the lot's occupancy as `global_costmap` (the planner's map). The lot is known, so
  the Go2 does not have to explore it first. It follows `lot_event`, so an open gate or a moved vehicle
  changes the map too.
* `LotPlaces`    sends the robot to a named place: `go_to_place("north-fence")` publishes a `goal_request`
  for DimOS's ReplanningAStarPlanner. Also an agent skill.
* `LotEvents`    `trigger("gate_open:gate-2")` validates an event spec and publishes it on `lot_event`.
  The physics child (`sim_process.py`) and `LotCctv` (`cctv.py`) apply it to their copies of the lot.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

from dimos_lcm.std_msgs import Bool

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.std_msgs.String import String
from dimos.utils.logging_config import setup_logger

from dimos_worlds.lot import events, go2_scene
from dimos_worlds.lot.places import load_model, load_places

logger = setup_logger()


def _event_text(msg: Any) -> str:
    return str(getattr(msg, "data", msg))


class _LotState:
    """A private standalone copy of the lot that follows the event stream."""

    def __init__(self) -> None:
        import mujoco

        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        events.reset(self.model, self.data)
        self.lock = threading.Lock()

    def apply(self, spec: str) -> None:
        with self.lock:
            events.apply(self.model, self.data, spec)


# ---------------------------------------------------------------- prior map


class LotPriorMapConfig(ModuleConfig):
    publish_period_s: float = 1.0


class LotPriorMap(Module):
    config: LotPriorMapConfig
    lot_event: In[String]
    global_costmap: Out[OccupancyGrid]

    _thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._state = _LotState()
        self._grid = go2_scene.occupancy_grid()
        self._stop_evt = threading.Event()
        self.lot_event.subscribe(self._on_event)
        self._thread = threading.Thread(target=self._publish_loop, name="lot-prior-map", daemon=True)
        self._thread.start()

    def _on_event(self, msg: Any) -> None:
        spec = _event_text(msg)
        try:
            self._state.apply(spec)
        except ValueError as exc:
            logger.warning(f"prior map ignored event: {exc}")
            return
        with self._state.lock:
            prior = go2_scene.occupancy_from(self._state.model, self._state.data)
        self._grid = go2_scene.occupancy_grid(prior)
        self._grid.ts = time.time()
        self.global_costmap.publish(self._grid)

    def _publish_loop(self) -> None:
        while not self._stop_evt.is_set():
            grid = self._grid
            grid.ts = time.time()
            self.global_costmap.publish(grid)
            self._stop_evt.wait(self.config.publish_period_s)

    @rpc
    def stop(self) -> None:
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=2.0)
        super().stop()


# ---------------------------------------------------------------- places


class LotPlaces(Module):
    odom: In[PoseStamped]
    goal_reached: In[Bool]
    goal_request: Out[PoseStamped]

    _pose: PoseStamped | None = None
    _goal: str | None = None
    _arrived: bool = False

    @rpc
    def start(self) -> None:
        super().start()
        self._places = load_places(with_home=True)
        self.odom.subscribe(self._on_odom)
        self.goal_reached.subscribe(self._on_goal_reached)

    def _on_odom(self, msg: PoseStamped) -> None:
        self._pose = msg

    def _on_goal_reached(self, msg: Any) -> None:
        if bool(getattr(msg, "data", msg)):
            self._arrived = True

    @rpc
    def list_places(self) -> dict[str, dict[str, Any]]:
        """{place id: {name, kind, x, y, camera}}; x, y is where the robot is sent."""
        return {
            p.id: {"name": p.name, "kind": p.kind, "x": p.goal_xy[0], "y": p.goal_xy[1], "camera": p.camera}
            for p in self._places.values()
        }

    @skill
    def go_to_place(self, place_id: str) -> str:
        """Walk to a named place in the lot. Places: home, front-gate, gate-2, service-bay-doors, row-a, row-b,
        row-c, row-d, fuel-tank, office-windows, north-fence, east-fence, south-fence. Returns at once; the
        robot keeps walking."""
        place = self._places.get(place_id)
        if place is None:
            return f"Unknown place {place_id!r}. Known: {', '.join(self._places)}."
        x, y = place.goal_xy
        yaw = 0.0
        if self._pose is not None:
            yaw = math.atan2(y - self._pose.position.y, x - self._pose.position.x)
        goal = PoseStamped(
            ts=time.time(),
            frame_id="world",
            position=[x, y, 0.0],
            orientation=[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)],
        )
        self._goal, self._arrived = place_id, False
        self.goal_request.publish(goal)
        return f"Walking to {place.name} ({x:.1f}, {y:.1f})."

    @rpc
    def status(self) -> dict[str, Any]:
        """Current goal, whether the planner reported it reached, and the robot's distance to it."""
        out: dict[str, Any] = {"goal": self._goal, "arrived": self._arrived, "distance_m": None}
        if self._goal is not None and self._pose is not None:
            x, y = self._places[self._goal].goal_xy
            out["distance_m"] = math.hypot(x - self._pose.position.x, y - self._pose.position.y)
        return out


# ---------------------------------------------------------------- events


class LotEvents(Module):
    lot_event: Out[String]

    @rpc
    def start(self) -> None:
        super().start()
        self._state = _LotState()

    @skill
    def trigger(self, spec: str) -> str:
        """Change the lot: person_at:<place>, person_clear, gate_open:<gate>, gate_close:<gate>,
        vehicle_moved:<row>:<from>:<to> or reset. Gates: front-gate, gate-2."""
        try:
            self._state.apply(spec)  # validates against the current state (e.g. occupied spots)
        except ValueError as exc:
            return f"Rejected: {exc}"
        self.lot_event.publish(String(spec))
        return f"Applied {spec}."

    @rpc
    def state(self) -> dict[str, Any]:
        with self._state.lock:
            return events.render_state(self._state.model, self._state.data)
