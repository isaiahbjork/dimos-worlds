"""The warehouse-fleet navigation stack in one process, for scripted scenarios and tests.

Same parts as `dimos run dimos-worlds.warehouse-fleet`, minus the transport: one SharedWorld, one DimOS
`GlobalPlanner` per robot (the planner inside DimOS's ReplanningAStarPlanner module, same config the blueprint gives
it), each fed its odometry and its robot-aware costmap (the other robots painted, as RobotAwareCostmaps does), and
the traffic Coordinator between goals and planners. Planner velocity commands go straight to the world, as
MovementManager forwards them when nobody teleoperates.

    fleet = InProcessFleet(coordinate=True)
    fleet.start()
    fleet.goal("go2", 18.0, 15.35, 3.14); fleet.goal("g1", 4.0, 15.35, 0.0)
    fleet.run_until_arrived(timeout_s=240)

Two ways to run it:

  * stepped (default): one loop on sim time does everything, in a fixed order. Each 20 ms world tick; odometry to
    the planners and the coordinator every 2 ticks; every 0.1 s each planner's local and monitor loop once
    (coord.stepped, robots in id order) and then the coordinator (inputs, rules, dispatch); the costmaps every
    0.5 s. No threads, no sleeps, no wall clock: the same scenario gives the same result however busy the machine
    is, and runs as fast as the CPU allows. Rates are the live ones (planners and coordinator 10 Hz, odometry
    25 Hz, costmaps 2 Hz).
  * realtime=True: the world on its own paced thread, DimOS's planner threads and the coordinator thread on the
    wall clock, as in a live run. Outcomes then depend on machine load; kept to check the live timing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
import threading
import time
from typing import Any

import numpy as np

from dimos_worlds.coord.runner import Coordinator
from dimos_worlds.coord.settle import agents_for
from dimos_worlds.coord.stepped import PERIOD_S, SteppedPlanner
from dimos_worlds.coord.traffic import Arrived, Goal, Stop, TrafficConfig, Velocity
from dimos_worlds.fleet import costmap
from dimos_worlds.fleet.robots import RobotSpec, warehouse_fleet
from dimos_worlds.fleet.world import CTRL_DT

log = logging.getLogger("dimos_worlds.coord.inproc")

ODOM_TICKS = 2  # odometry every 40 ms
PLAN_TICKS = round(PERIOD_S / CTRL_DT)  # planners and coordinator every 0.1 s
COSTMAP_TICKS = 25  # costmaps every 0.5 s


@dataclass
class Outcome:
    arrived: dict[str, float] = field(default_factory=dict)  # robot -> seconds after start
    final: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    goals: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    falls: dict[str, int] = field(default_factory=dict)
    min_gap_m: float = math.inf
    events: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0  # sim seconds (stepped) or wall seconds (realtime)

    def errors(self, rid: str) -> tuple[float, float]:
        x, y, yaw = self.final[rid]
        gx, gy, gyaw = self.goals[rid]
        return math.hypot(x - gx, y - gy), abs((gyaw - yaw + math.pi) % (2 * math.pi) - math.pi)

    @property
    def all_arrived(self) -> bool:
        return set(self.goals) <= set(self.arrived)


class InProcessFleet:
    def __init__(self, specs: tuple[RobotSpec, ...] | None = None, *, coordinate: bool = True,
                 config: TrafficConfig | None = None, run_log: Any = None, speed: float = 1.0,
                 near_m: float = 3.0, pad_m: float = 0.25, realtime: bool = False) -> None:
        from dimos.core.global_config import GlobalConfig

        from dimos_worlds.coord.stepped import nav_module

        from dimos_worlds.fleet.places import load_places
        from dimos_worlds.fleet.world import warehouse_world
        from dimos_worlds.warehouse.cell import LAYOUT
        from dimos_worlds.warehouse.modules import prior_for
        from dimos_worlds.warehouse.scene import PLACES_JSON

        self.specs = specs or warehouse_fleet()
        self.world = warehouse_world(self.specs, run_log=run_log, speed=speed)
        model = next(iter(self.world.bodies.values())).model
        self.base, self.prior = prior_for(costmap.prior_from_model(model), load_places(PLACES_JSON), LAYOUT)
        self.near_m, self.pad_m = near_m, pad_m
        self.radii = {s.id: s.body_radius for s in self.specs}
        self.planners: dict[str, Any] = {}
        for s in self.specs:
            gc = GlobalConfig(robot_width=2 * s.body_radius, robot_rotation_diameter=s.turn_diameter,
                              nerf_speed=s.nav_speed)
            self.planners[s.id] = nav_module("global_planner").GlobalPlanner(gc)
        self.realtime = realtime
        self._clock = time.monotonic if realtime else (lambda: self.world.sim_time)
        self.stepped: dict[str, SteppedPlanner] = {}
        self.coordinate = coordinate
        self.t0 = self._clock()
        self.outcome = Outcome()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.coord: Coordinator | None = None
        if coordinate:
            self.coord = Coordinator(agents_for(self.specs), config, on_goal=self._send_goal,
                                     on_stop=self._send_stop, on_velocity=self._send_velocity,
                                     on_arrived=self._arrived, clock=self._clock)

    # ---- outputs of the coordinator (what the DimOS module publishes)
    def _send_goal(self, g: Goal) -> None:
        self.planners[g.robot].handle_goal_request(self._pose(g.x, g.y, g.yaw))

    def _send_stop(self, s: Stop) -> None:
        self.planners[s.robot].handle_goal_request(self._pose(s.x, s.y, s.yaw))

    def _send_velocity(self, v: Velocity) -> None:
        self.world.set_velocity(v.robot, v.vx, v.vy, v.wz)

    def _arrived(self, a: Arrived) -> None:
        with self._lock:
            self.outcome.arrived.setdefault(a.robot, self._clock() - self.t0)

    def _pose(self, x: float, y: float, yaw: float) -> Any:
        return _pose_stamped(x, y, yaw, time.time() if self.realtime else self.world.sim_time)

    # ---- wiring
    def start(self) -> None:
        for rid, p in self.planners.items():
            p.cmd_vel.subscribe(lambda tw, rid=rid: self.world.set_velocity(
                rid, tw.linear.x, tw.linear.y, tw.angular.z))
            p.path.subscribe(lambda path, rid=rid: self._on_path(rid, path))
            p.goal_reached.subscribe(lambda b, rid=rid: self._on_reached(rid, bool(b.data)))
            if self.realtime:
                p.start()
            else:
                self.stepped[rid] = SteppedPlanner(p, self._clock)
        self.world.on_tick.append(self._on_tick)
        self._publish_costmaps()
        if self.coord is not None:
            self.coord.costmap(self.base, self.prior.origin_x, self.prior.origin_y, self.prior.resolution)
        self._on_tick(self.world)
        if self.realtime:
            if self.coord is not None:
                self.coord.start()
            self.world.start()
            threading.Thread(target=self._costmap_loop, name="inproc-costmaps", daemon=True).start()
        elif self.coord is not None:
            self.coord.run_once(step=False)
        self.t0 = self._clock()

    def close(self) -> None:
        self._stop.set()
        if self.coord is not None:
            self.coord.stop()
        for p in self.planners.values():
            p.stop()
        for sp in self.stepped.values():
            sp.close()
        self.world.close()

    def tick(self) -> None:
        """Stepped mode: one world tick and whatever falls due on it (see the module docstring)."""
        self.world.tick()
        n = self.world.ticks
        if n % PLAN_TICKS == 0:
            for rid in sorted(self.stepped):
                self.stepped[rid].step()
            if self.coord is not None:
                self.coord.run_once(step=True)
        if n % COSTMAP_TICKS == 0:
            self._publish_costmaps()

    def goal(self, rid: str, x: float, y: float, yaw: float) -> None:
        self.outcome.goals[rid] = (x, y, yaw)
        self.outcome.arrived.pop(rid, None)
        if self.coord is not None:
            self.coord.goal(rid, x, y, yaw)
        else:
            self.planners[rid].handle_goal_request(self._pose(x, y, yaw))

    def _on_path(self, rid: str, path: Any) -> None:
        if self.coord is not None:
            pts = np.array([[p.position.x, p.position.y] for p in path.poses], dtype=float).reshape(-1, 2)
            self.coord.path(rid, pts)

    def _on_reached(self, rid: str, ok: bool) -> None:
        if self.coord is not None:
            self.coord.goal_reached(rid, ok)
        elif ok:
            self._arrived(Arrived(rid, True))

    def _on_tick(self, world: Any) -> None:
        if world.ticks % ODOM_TICKS:
            return
        poses = {rid: b.pose() for rid, b in world.bodies.items()}
        for rid, (x, y, yaw) in sorted(poses.items()):
            self.planners[rid].handle_odom(self._pose(x, y, yaw))
            if self.coord is not None:
                self.coord.pose(rid, x, y, yaw)
        ids = sorted(poses)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                self.outcome.min_gap_m = min(self.outcome.min_gap_m, math.dist(poses[a][:2], poses[b][:2]))

    def _costmap_loop(self) -> None:
        while not self._stop.wait(0.5):
            self._publish_costmaps()

    def _publish_costmaps(self) -> None:
        from dimos_worlds.warehouse.modules import grid_msg

        poses = {rid: p[:2] for rid, p in self.world.poses().items()}
        prior = costmap.Prior(grid=self.base, origin_x=self.prior.origin_x, origin_y=self.prior.origin_y,
                              resolution=self.prior.resolution)
        masks = costmap.robot_masks(prior, [], poses, self.radii, pad_m=self.pad_m, near_m=self.near_m)
        ts = time.time() if self.realtime else self.world.sim_time
        for rid, grid in masks.items():
            self.planners[rid].handle_global_costmap(grid_msg(grid, prior, ts))

    # ---- running a scenario
    def run_until_arrived(self, timeout_s: float = 300.0) -> Outcome:
        """Until every robot with a goal has arrived or `timeout_s` (sim seconds when stepped) has passed, then 1 s
        more for a settled robot to come to rest."""
        if self.realtime:
            end = time.monotonic() + timeout_s
            while time.monotonic() < end and not self.outcome.all_arrived:
                time.sleep(0.2)
            time.sleep(1.0)
        else:
            end_tick = self.world.ticks + round(timeout_s / CTRL_DT)
            while self.world.ticks < end_tick and not self.outcome.all_arrived:
                self.tick()
            for _ in range(round(1.0 / CTRL_DT)):
                self.tick()
        o = self.outcome
        o.elapsed_s = self._clock() - self.t0
        o.final = self.world.poses()
        o.falls = {rid: b.falls for rid, b in self.world.bodies.items()}
        if self.coord is not None:
            o.events = [f"{t - self.t0:6.1f} {text}" for t, text in self.coord.traffic.events]
        return o


def _pose_stamped(x: float, y: float, yaw: float, ts: float) -> Any:
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from dimos.msgs.geometry_msgs.Quaternion import Quaternion
    from dimos.msgs.geometry_msgs.Vector3 import Vector3

    return PoseStamped(ts=ts, frame_id="world", position=Vector3(float(x), float(y), 0.0),
                       orientation=Quaternion(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
