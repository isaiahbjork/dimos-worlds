"""DimOS modules around the shared world.

WarehouseFleetSim runs one SharedWorld (Go2 + G1 in the warehouse) in its own thread and publishes one stream set
per robot: `{id}_odom`, `{id}_color_image`, `{id}_camera_info`, `{id}_joint_state`, and takes `{id}_cmd_vel`. The
blueprint renames them to `{id}/odom` and so on, the namespace each robot's planner listens on. It also publishes the
scene's occupancy prior with the places' stay-outs as `global_costmap`.

Every cmd_vel that changes what a robot is asked to do is logged with its tick, so the run can be re-executed
bit for bit afterwards: `dimos-worlds-replay <run_log>`.

RobotAwareCostmaps turns `global_costmap` into one grid per robot with the OTHER robot's footprint painted occupied
(`{id}_costmap`), so each planner routes around the other body instead of through it.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.utils.logging_config import setup_logger

from dimos_worlds.fleet import costmap
from dimos_worlds.fleet.places import load_places
from dimos_worlds.fleet.robots import warehouse_fleet

logger = setup_logger()

ROBOT_IDS = ("go2", "g1")  # the ports below; robots.warehouse_fleet() must match


def _grid_msg(grid: np.ndarray, prior: costmap.Prior, ts: float) -> OccupancyGrid:
    origin = Pose()
    origin.position.x = prior.origin_x
    origin.position.y = prior.origin_y
    origin.orientation.w = 1.0
    return OccupancyGrid(grid=grid, resolution=prior.resolution, origin=origin, frame_id="world", ts=ts)


class WarehouseFleetSimConfig(ModuleConfig):
    run_log: str | None = "warehouse-fleet-run.jsonl"  # None: keep the log in memory only
    speed: float = 1.0  # x real time (best effort; sim time never stretches a tick)
    odom_every_ticks: int = 2  # 25 Hz
    joints_every_ticks: int = 5  # 10 Hz
    camera_hz: float = 2.0  # per robot; 0 disables head-camera rendering
    camera_width: int = 640
    camera_height: int = 360
    costmap_every_s: float = 1.0
    # Policy dead-band compensation (robots.Compensation): on by default; False sends commands to the policies as is
    compensate: bool = True


class WarehouseFleetSim(Module):
    """Go2 + G1 in one warehouse, each walking on its DimOS policy at its own physics step."""

    config: WarehouseFleetSimConfig
    global_costmap: Out[OccupancyGrid]
    go2_odom: Out[PoseStamped]
    go2_color_image: Out[Image]
    go2_camera_info: Out[CameraInfo]
    go2_joint_state: Out[JointState]
    go2_cmd_vel: In[Twist]
    g1_odom: Out[PoseStamped]
    g1_color_image: Out[Image]
    g1_camera_info: Out[CameraInfo]
    g1_joint_state: Out[JointState]
    g1_cmd_vel: In[Twist]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._world: Any = None
        self._renderers: dict[str, Any] = {}
        self._prior: costmap.Prior | None = None
        self._places: list[dict] = []
        self._stop = threading.Event()
        self._costmap_thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        from dimos_worlds.fleet.world import warehouse_world
        from dimos_worlds.replay.log import RunLog
        from dimos_worlds.warehouse.scene import PLACES_JSON

        specs = warehouse_fleet(self.config.compensate)
        if tuple(s.id for s in specs) != ROBOT_IDS:
            raise RuntimeError(f"robots {[s.id for s in specs]} do not match this module's ports {ROBOT_IDS}")
        self._world = warehouse_world(specs, run_log=RunLog(self.config.run_log), speed=self.config.speed)
        if self.config.run_log:
            logger.info(f"shared world run log: {self.config.run_log} (dimos-worlds-replay checks it)")
        any_model = next(iter(self._world.bodies.values())).model
        self._prior = costmap.prior_from_model(any_model)
        self._places = load_places(PLACES_JSON)
        for rid in ROBOT_IDS:
            port = getattr(self, f"{rid}_cmd_vel")
            self.register_disposable(Disposable(port.subscribe(lambda msg, rid=rid: self._on_cmd_vel(rid, msg))))
        self._world.on_tick.append(self._on_tick)
        self._world.start()
        self._costmap_thread = threading.Thread(target=self._costmap_loop, name="fleet-costmap", daemon=True)
        self._costmap_thread.start()
        if self.config.camera_hz > 0:
            threading.Thread(target=self._camera_loop, name="fleet-cameras", daemon=True).start()

    @rpc
    def stop(self) -> None:
        self._stop.set()
        if self._world is not None:
            self._world.close()
        super().stop()

    def _on_cmd_vel(self, robot_id: str, twist: Twist) -> None:
        if self._world is not None:
            self._world.set_velocity(robot_id, twist.linear.x, twist.linear.y, twist.angular.z)

    # ---- publishing (world thread, after each tick, outside the world lock; reads state only)
    def _on_tick(self, world: Any) -> None:
        t = world.ticks
        ts = time.time()
        if t % self.config.odom_every_ticks == 0:
            for rid in ROBOT_IDS:
                d = world.bodies[rid].data
                getattr(self, f"{rid}_odom").publish(PoseStamped(
                    ts=ts, frame_id="world",
                    position=Vector3(float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2])),
                    orientation=Quaternion(float(d.qpos[4]), float(d.qpos[5]), float(d.qpos[6]), float(d.qpos[3])),
                ))
        if t % self.config.joints_every_ticks == 0:
            for rid in ROBOT_IDS:
                body = world.bodies[rid]
                m, d = body.model, body.data
                names = [m.joint(j).name for j in range(1, m.njnt)]
                getattr(self, f"{rid}_joint_state").publish(JointState(
                    ts=ts, name=names, position=[float(v) for v in d.qpos[7:]],
                    velocity=[float(v) for v in d.qvel[6:]]))

    def _camera_loop(self) -> None:
        """Head cameras on their own thread, from a copy of each robot's state: rendering on the physics thread
        (software GL in a DimOS worker on macOS) slows the whole world down."""
        import mujoco

        world = self._world
        scratch = {rid: mujoco.MjData(world.bodies[rid].model) for rid in ROBOT_IDS}
        period = 1.0 / self.config.camera_hz
        while not self._stop.wait(period):
            for rid in ROBOT_IDS:
                world.snapshot(rid, scratch[rid])
                if not self._publish_camera(world, rid, scratch[rid], time.time()):
                    return

    def _publish_camera(self, world: Any, rid: str, data: Any, ts: float) -> bool:
        import mujoco

        body = world.bodies[rid]
        try:
            r = self._renderers.get(rid)
            if r is None:  # created on the camera thread: GL contexts belong to the thread that made them
                r = mujoco.Renderer(body.model, height=self.config.camera_height, width=self.config.camera_width)
                self._renderers[rid] = r
            r.update_scene(data, camera="head_camera")
            frame = r.render()
        except Exception as e:  # no GL on this machine: keep the sim running without cameras
            logger.warning(f"head-camera rendering disabled: {e}")
            return False
        img = Image.from_numpy(np.ascontiguousarray(frame), format=ImageFormat.RGB)
        img.frame_id = f"{rid}/camera_optical"
        img.ts = ts
        getattr(self, f"{rid}_color_image").publish(img)
        getattr(self, f"{rid}_camera_info").publish(CameraInfo.from_fov(
            45.0, self.config.camera_width, self.config.camera_height, frame_id=f"{rid}/camera_optical"))
        return True

    def _costmap_loop(self) -> None:
        assert self._prior is not None
        from dimos_worlds.warehouse.cell import LAYOUT
        from dimos_worlds.warehouse.modules import prior_for

        grid, p = prior_for(self._prior, self._places, LAYOUT)  # same map go2-warehouse plans on
        while not self._stop.wait(self.config.costmap_every_s):
            self.global_costmap.publish(_grid_msg(grid, p, time.time()))


class RobotAwareCostmapsConfig(ModuleConfig):
    pad_m: float = 0.25  # added to the other robot's body radius
    republish_s: float = 0.5
    move_eps_m: float = 0.1
    near_m: float = 3.0  # paint the other robot only within this distance (see costmap.robot_masks)


class RobotAwareCostmaps(Module):
    """global_costmap + the OTHER robot's footprint -> go2_costmap / g1_costmap."""

    config: RobotAwareCostmapsConfig
    global_costmap: In[OccupancyGrid]
    go2_odom: In[PoseStamped]
    g1_odom: In[PoseStamped]
    go2_costmap: Out[OccupancyGrid]
    g1_costmap: Out[OccupancyGrid]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._base: OccupancyGrid | None = None
        self._poses: dict[str, tuple[float, float]] = {}
        self._sent: dict[str, tuple[float, float]] = {}
        self._radii = {s.id: s.body_radius for s in warehouse_fleet()}

    @rpc
    def start(self) -> None:
        super().start()
        self.register_disposable(Disposable(self.global_costmap.subscribe(self._on_costmap)))
        for rid in ROBOT_IDS:
            port = getattr(self, f"{rid}_odom")
            self.register_disposable(Disposable(port.subscribe(lambda msg, rid=rid: self._on_odom(rid, msg))))
        threading.Thread(target=self._watch, name="robot-aware-costmaps", daemon=True).start()

    @rpc
    def stop(self) -> None:
        self._stop.set()
        super().stop()

    def _on_odom(self, rid: str, msg: PoseStamped) -> None:
        x, y = float(msg.position.x), float(msg.position.y)
        if np.isfinite(x) and np.isfinite(y):
            with self._lock:
                self._poses[rid] = (x, y)

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self._base = msg
        self._publish()

    def _watch(self) -> None:
        while not self._stop.wait(self.config.republish_s):
            with self._lock:
                moved = any(rid not in self._sent or max(abs(p[0] - self._sent[rid][0]), abs(p[1] - self._sent[rid][1]))
                            > self.config.move_eps_m for rid, p in self._poses.items())
            if moved:
                self._publish()

    def _publish(self) -> None:
        with self._lock:
            base, poses = self._base, dict(self._poses)
        if base is None:
            return
        prior = costmap.Prior(grid=np.asarray(base.grid, dtype=np.int8), origin_x=float(base.origin.position.x),
                              origin_y=float(base.origin.position.y), resolution=float(base.resolution))
        masks = costmap.robot_masks(prior, [], poses, self._radii, pad_m=self.config.pad_m,
                                    near_m=self.config.near_m)
        for rid in ROBOT_IDS:
            getattr(self, f"{rid}_costmap").publish(_grid_msg(masks[rid], prior, base.ts))
        with self._lock:
            self._sent = poses


# The traffic coordinator for this fleet: between goals and the robots' planners (coord.traffic has the rules).
from dimos_worlds.coord.module import traffic_coordinator  # noqa: E402

FleetTraffic = traffic_coordinator(ROBOT_IDS, "FleetTraffic")


def fleet_agents(compensate: bool = True) -> list[dict[str, Any]]:
    """FleetTraffic's `agents` config for warehouse_fleet()."""
    from dataclasses import asdict

    from dimos_worlds.coord.settle import agents_for

    return [asdict(a) for a in agents_for(warehouse_fleet(compensate))]
