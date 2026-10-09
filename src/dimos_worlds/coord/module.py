"""The traffic coordinator as a DimOS module, for any set of robots that each run a DimOS planner.

`traffic_coordinator(ids)` builds a module class with one stream set per robot id. For robot `r`:

    in   r_goal_request  PoseStamped   goals, from whoever sends them (wire to the topic users publish on)
    in   r_clicked_point PointStamped  same, as a clicked point
    in   r_odom          PoseStamped
    in   r_path          Path          the path r's planner is following (ReplanningAStarPlanner.path)
    in   r_goal_reached  Bool          r's planner result (ReplanningAStarPlanner.goal_reached)
    out  r_nav_goal      PoseStamped   the goal r's planner should pursue (wire to the planner's goal_request)
    out  r_cmd_vel       Twist         settling at the goal only (the robot's cmd_vel)
    out  r_arrived       Bool          r reached the goal it was given (after settling)
    in   global_costmap  OccupancyGrid static occupancy, for finding pockets

The class has to live at module level under its own name so DimOS workers can import it, e.g.

    FleetTraffic = traffic_coordinator(("go2", "g1"), "FleetTraffic")

Agents (radius, priority, settling) come from the `agents` config: [{"id", "radius", "priority", "speed",
"settle": {SettleConfig fields} | None}, ...]. Right-of-way rules: coord.traffic.
"""
from __future__ import annotations

import math
import threading
from typing import Any

from dimos_lcm.std_msgs import Bool
import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PointStamped import PointStamped
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.nav_msgs.Path import Path
from dimos.utils.logging_config import setup_logger

from dimos_worlds.coord.runner import Coordinator
from dimos_worlds.coord.traffic import Agent, Arrived, Goal, SettleConfig, Stop, TrafficConfig, Velocity

logger = setup_logger()


class TrafficCoordinatorConfig(ModuleConfig):
    agents: list[dict[str, Any]] = []
    margin_m: float = 0.3
    horizon_m: float = 8.0
    stop_margin_m: float = 1.0
    retry_s: float = 3.0
    deadlock_s: float = 25.0
    rate_hz: float = 10.0


def _agents(rows: list[dict[str, Any]]) -> list[Agent]:
    out = []
    for row in rows:
        row = dict(row)
        settle = row.pop("settle", None)
        out.append(Agent(**row, settle=SettleConfig(**settle) if settle is not None else None))
    return out


def _yaw(q: Any) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class _TrafficCoordinatorBase(Module):
    """Base: ports are added per robot id by traffic_coordinator()."""

    config: TrafficCoordinatorConfig
    ROBOT_IDS: tuple[str, ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._coord: Coordinator | None = None
        self._lock = threading.Lock()

    @rpc
    def start(self) -> None:
        super().start()
        c = self.config
        agents = _agents(c.agents)
        if sorted(a.id for a in agents) != sorted(self.ROBOT_IDS):
            raise RuntimeError(f"agents {[a.id for a in agents]} do not match this module's ports {self.ROBOT_IDS}")
        cfg = TrafficConfig(margin_m=c.margin_m, horizon_m=c.horizon_m, stop_margin_m=c.stop_margin_m,
                            retry_s=c.retry_s, deadlock_s=c.deadlock_s)
        self._coord = Coordinator(agents, cfg, on_goal=self._goal_out, on_stop=self._stop_out,
                                  on_velocity=self._vel_out, on_arrived=self._arrived_out, rate_hz=c.rate_hz)
        sub = self.register_disposable
        sub(Disposable(self.global_costmap.subscribe(self._on_costmap)))
        for rid in self.ROBOT_IDS:
            sub(Disposable(self._port(rid, "goal_request").subscribe(lambda m, rid=rid: self._on_goal(rid, m))))
            sub(Disposable(self._port(rid, "clicked_point").subscribe(
                lambda m, rid=rid: self._on_goal(rid, m.to_pose_stamped()))))
            sub(Disposable(self._port(rid, "odom").subscribe(lambda m, rid=rid: self._on_odom(rid, m))))
            sub(Disposable(self._port(rid, "path").subscribe(lambda m, rid=rid: self._on_path(rid, m))))
            sub(Disposable(self._port(rid, "goal_reached").subscribe(
                lambda m, rid=rid: self._coord.goal_reached(rid, bool(m.data)))))
        self._coord.start()
        logger.info(f"traffic coordinator up for {list(self.ROBOT_IDS)}")

    @rpc
    def stop(self) -> None:
        if self._coord is not None:
            self._coord.stop()
        super().stop()

    def _port(self, rid: str, name: str) -> Any:
        return getattr(self, f"{rid}_{name}")

    # ---- inputs
    def _on_costmap(self, msg: OccupancyGrid) -> None:
        assert self._coord is not None
        self._coord.costmap(np.asarray(msg.grid), float(msg.origin.position.x), float(msg.origin.position.y),
                            float(msg.resolution))

    def _on_goal(self, rid: str, msg: PoseStamped) -> None:
        x, y = float(msg.position.x), float(msg.position.y)
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        assert self._coord is not None
        self._coord.goal(rid, x, y, _yaw(msg.orientation))

    def _on_odom(self, rid: str, msg: PoseStamped) -> None:
        assert self._coord is not None
        self._coord.pose(rid, float(msg.position.x), float(msg.position.y), _yaw(msg.orientation))

    def _on_path(self, rid: str, msg: Path) -> None:
        assert self._coord is not None
        pts = np.array([[p.position.x, p.position.y] for p in msg.poses], dtype=float).reshape(-1, 2)
        self._coord.path(rid, pts)

    # ---- outputs
    def _goal_out(self, g: Goal) -> None:
        logger.info(f"traffic: {g.robot} goal ({g.x:.2f}, {g.y:.2f}) [{g.reason}]")
        self._port(g.robot, "nav_goal").publish(PoseStamped(
            frame_id="world", position=Vector3(g.x, g.y, 0.0),
            orientation=Quaternion(0.0, 0.0, math.sin(g.yaw / 2), math.cos(g.yaw / 2))))

    def _stop_out(self, s: Stop) -> None:
        logger.info(f"traffic: {s.robot} stand at ({s.x:.2f}, {s.y:.2f}) [{s.reason}]")
        self._goal_out(Goal(s.robot, s.x, s.y, s.yaw, s.reason))

    def _vel_out(self, v: Velocity) -> None:
        self._port(v.robot, "cmd_vel").publish(Twist(linear=Vector3(v.vx, v.vy, 0.0), angular=Vector3(0.0, 0.0, v.wz)))

    def _arrived_out(self, a: Arrived) -> None:
        logger.info(f"traffic: {a.robot} arrived")
        self._port(a.robot, "arrived").publish(Bool(data=a.ok))


def traffic_coordinator(robot_ids: tuple[str, ...], name: str, module: str | None = None) -> type[Module]:
    """A coordinator module class with ports for `robot_ids`. Assign it at module level under `name`."""
    import sys

    ann: dict[str, Any] = {"config": TrafficCoordinatorConfig, "global_costmap": In[OccupancyGrid]}
    for rid in robot_ids:
        ann |= {f"{rid}_goal_request": In[PoseStamped], f"{rid}_clicked_point": In[PointStamped],
                f"{rid}_odom": In[PoseStamped], f"{rid}_path": In[Path], f"{rid}_goal_reached": In[Bool],
                f"{rid}_nav_goal": Out[PoseStamped],
                f"{rid}_cmd_vel": Out[Twist], f"{rid}_arrived": Out[Bool]}
    mod = module or sys._getframe(1).f_globals.get("__name__", __name__)  # noqa: SLF001
    ns = {"__annotations__": ann, "__module__": mod, "__qualname__": name, "ROBOT_IDS": tuple(robot_ids),
          "__doc__": f"Traffic coordinator for robots {', '.join(robot_ids)} (see dimos_worlds.coord.module)."}
    return type(name, (_TrafficCoordinatorBase,), ns)
