"""DimOS modules for the plant: prior map and named places.

* `PlantPriorMap`  publishes the plant's occupancy (go2_scene.occupancy) as `global_costmap`. The plant is known and
  static, so the planner does not need a lidar-built map first; stay-out zones (stairs) are walls in it.
* `PlantPlaces`    sends the robot to a named place: `go_to_place("pump-102")` publishes a `goal_request` for
  DimOS's ReplanningAStarPlanner. Also an agent skill.
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

from dimos_worlds.plant import go2_scene


class PlantPriorMapConfig(ModuleConfig):
    publish_period_s: float = 1.0


class PlantPriorMap(Module):
    config: PlantPriorMapConfig
    global_costmap: Out[OccupancyGrid]

    @rpc
    def start(self) -> None:
        super().start()
        self._grid = go2_scene.occupancy_grid(go2_scene.occupancy())
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._publish_loop, name="plant-prior-map", daemon=True)
        self._thread.start()

    def _publish_loop(self) -> None:
        while not self._stop_evt.is_set():
            self._grid.ts = time.time()
            self.global_costmap.publish(self._grid)
            self._stop_evt.wait(self.config.publish_period_s)

    @rpc
    def stop(self) -> None:
        if getattr(self, "_stop_evt", None) is not None:
            self._stop_evt.set()
        super().stop()


class PlantPlaces(Module):
    odom: In[PoseStamped]
    goal_reached: In[Bool]
    goal_request: Out[PoseStamped]

    _pose: PoseStamped | None = None
    _goal: str | None = None
    _arrived: bool = False

    @rpc
    def start(self) -> None:
        super().start()
        self._places = {p["id"]: p for p in go2_scene.places() if p["kind"] != "stay_out"}
        self.odom.subscribe(self._on_odom)
        self.goal_reached.subscribe(self._on_goal_reached)

    def _on_odom(self, msg: PoseStamped) -> None:
        self._pose = msg

    def _on_goal_reached(self, msg: Any) -> None:
        if bool(getattr(msg, "data", msg)):
            self._arrived = True

    @rpc
    def list_places(self) -> dict[str, dict[str, Any]]:
        """{place id: {name, kind, x, y}}."""
        return {k: {"name": p["name"], "kind": p["kind"], "x": p["pose"]["x"], "y": p["pose"]["y"]}
                for k, p in self._places.items()}

    @skill
    def go_to_place(self, place_id: str) -> str:
        """Walk to a named place in the plant: dock, control-room, pump-101, pump-102, pump-103, tank-201-outlet,
        tank-202-drain, compressor-301. Returns at once; the robot keeps walking."""
        place = self._places.get(place_id)
        if place is None:
            return f"Unknown place {place_id!r}. Known: {', '.join(self._places)}."
        x, y = place["pose"]["x"], place["pose"]["y"]
        yaw = 0.0
        if self._pose is not None:
            yaw = math.atan2(y - self._pose.position.y, x - self._pose.position.x)
        goal = PoseStamped(ts=time.time(), frame_id="world", position=[x, y, 0.0],
                           orientation=[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
        self._goal, self._arrived = place_id, False
        self.goal_request.publish(goal)
        return f"Walking to {place['name']} ({x:.1f}, {y:.1f})."

    @rpc
    def status(self) -> dict[str, Any]:
        """Current goal, whether the planner reported it reached, and the robot's distance to it."""
        out: dict[str, Any] = {"goal": self._goal, "arrived": self._arrived, "distance_m": None}
        if self._goal is not None and self._pose is not None:
            p = self._places[self._goal]["pose"]
            out["distance_m"] = math.hypot(p["x"] - self._pose.position.x, p["y"] - self._pose.position.y)
        return out
