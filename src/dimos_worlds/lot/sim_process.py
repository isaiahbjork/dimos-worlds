"""Physics child process for the Go2 in the lot (replaces DimOS's mujoco_process.py for this world).

Started by `connection.LotMujocoConnection` with the same arguments DimOS passes its own child
(pickled GlobalConfig, shared-memory names), so the parent side is DimOS's unchanged `MujocoConnection`
reading the same shared memory. Differences from DimOS's child:

* the scene is the lot (`go2_scene.scene_xml()`), loaded through DimOS's own `load_model`, so the robot
  body, ONNX policy and DimOS's `person` mesh are exactly DimOS's
* headless by default (no `viewer.launch_passive`, so no `mjpython` and no window on macOS);
  `DIMOS_WORLDS_VIEWER=1` opens DimOS's passive viewer instead
* lot events (`events.apply` spec strings) arrive on the `/lot_event` topic and are applied between steps;
  DimOS's `/person_pose` topic (person_on_track) still moves the person as in DimOS
* the head camera and depth lidar render on their own thread from physics snapshots, so a slow renderer
  lowers the frame rate instead of slowing the robot; video rate is `DIMOS_WORLDS_VIDEO_FPS` (default 5;
  DimOS uses 20)
* physics is paced to real time and logs its real-time factor every 30 s
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import queue
import signal
import sys
import threading
import time
from typing import Any

import mujoco
import numpy as np

from dimos.core.global_config import GlobalConfig
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.simulation.mujoco.constants import DEPTH_CAMERA_FOV, LIDAR_FPS, LIDAR_RESOLUTION, VIDEO_HEIGHT, VIDEO_WIDTH
from dimos.simulation.mujoco.depth_camera import depth_image_to_point_cloud
from dimos.simulation.mujoco.model import load_model
from dimos.simulation.mujoco.mujoco_process import MockController
from dimos.simulation.mujoco.person_on_track import PersonPositionController
from dimos.simulation.mujoco.shared_memory import ShmReader
from dimos.utils.logging_config import setup_logger

from dimos_worlds.lot import events
from dimos_worlds.lot.go2_scene import scene_xml

logger = setup_logger()

EVENT_TOPIC = "/lot_event"


class _EventInbox:
    """Lot event specs from the bus, applied by the physics thread between steps."""

    def __init__(self, g: GlobalConfig) -> None:
        from dimos.core.transport_factory import make_transport
        from dimos.msgs.std_msgs.String import String

        self._q: queue.Queue[str] = queue.Queue()
        self._transport = make_transport(EVENT_TOPIC, String, g=g)
        self._transport.subscribe(lambda msg: self._q.put(str(getattr(msg, "data", msg))))

    def drain(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        while True:
            try:
                spec = self._q.get_nowait()
            except queue.Empty:
                return
            try:
                events.apply(model, data, spec)
                logger.info(f"lot event applied: {spec}")
            except ValueError as exc:
                logger.warning(f"lot event rejected: {exc}")

    def stop(self) -> None:
        self._transport.stop()


def _lidar(model: mujoco.MjModel, data: mujoco.MjData, renderers: list, cam_ids: list[int], shm: ShmReader) -> None:
    import open3d as o3d  # type: ignore[import-untyped]

    depths = []
    for r, cid in zip(renderers, cam_ids):
        r.update_scene(data, camera=cid)
        depths.append(r.render())
    shm.write_depth(*depths)
    points = []
    for depth, cid in zip(depths, cam_ids):
        pts = depth_image_to_point_cloud(
            depth, data.cam_xpos[cid], data.cam_xmat[cid].reshape(3, 3), fov_degrees=DEPTH_CAMERA_FOV
        )
        if pts.size > 0:
            points.append(pts)
    if points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(points))
        pcd = pcd.voxel_down_sample(voxel_size=LIDAR_RESOLUTION)
        shm.write_lidar(PointCloud2(pointcloud=pcd, ts=time.time(), frame_id="world"))


class _Sensors(threading.Thread):
    """Head camera and depth-lidar rendering on their own thread, from snapshots of the physics state.

    Rendering is the slow part (tens to hundreds of ms per frame on a busy laptop); on the physics thread it
    would drag the sim below real time and DimOS's planner would read slow walking as "stuck"."""

    def __init__(self, model: mujoco.MjModel, shm: ShmReader, video_fps: float) -> None:
        super().__init__(name="lot-sim-sensors", daemon=True)
        self.model = model
        self.shm = shm
        self.video_dt = 1.0 / video_fps
        self.lidar_dt = 1.0 / LIDAR_FPS
        self.lock = threading.Lock()
        self.snapshot = mujoco.MjData(model)
        self.fresh = False
        self.stop_evt = threading.Event()

    def publish_state(self, data: mujoco.MjData) -> None:
        with self.lock:
            mujoco.mj_copyData(self.snapshot, self.model, data)
            self.fresh = True

    def run(self) -> None:
        model = self.model
        head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        lidar_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n)
            for n in ("lidar_front_camera", "lidar_left_camera", "lidar_right_camera")
        ]
        rgb = mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
        depth = [mujoco.Renderer(model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH) for _ in lidar_ids]
        for r in depth:
            r.enable_depth_rendering()
        local = mujoco.MjData(model)
        last_video = last_lidar = 0.0
        try:
            while not self.stop_evt.is_set():
                now = time.time()
                due_video = now - last_video >= self.video_dt
                due_lidar = now - last_lidar >= self.lidar_dt
                if (due_video or due_lidar) and self.fresh:
                    with self.lock:
                        mujoco.mj_copyData(local, model, self.snapshot)
                        self.fresh = False
                    if due_video:
                        rgb.update_scene(local, camera=head)
                        self.shm.write_video(rgb.render())
                        last_video = now
                    if due_lidar:
                        _lidar(model, local, depth, lidar_ids, self.shm)
                        last_lidar = now
                self.stop_evt.wait(0.01)
        finally:
            rgb.close()
            for r in depth:
                r.close()


def run(config: GlobalConfig, shm: ShmReader) -> None:
    robot = config.robot_model or "unitree_go1"
    if robot == "unitree_go2":
        robot = "unitree_go1"  # DimOS's Go2 sim runs the Go1 body and policy
    if robot != "unitree_go1":
        raise ValueError(f"the lot sim supports the Go2 (Go1 body) only, got {robot!r}")

    model, data = load_model(MockController(shm), robot=robot, scene_xml=scene_xml())
    start = config.mujoco_start_pos_float
    data.qpos[0:3] = [start[0], start[1], 0.3]
    events.reset(model, data)

    person = PersonPositionController(model)
    inbox = _EventInbox(config)
    sensors = _Sensors(model, shm, float(os.environ.get("DIMOS_WORLDS_VIDEO_FPS", "5")))
    sensors.publish_state(data)
    sensors.start()

    viewer_ctx = None
    if os.environ.get("DIMOS_WORLDS_VIEWER", "") == "1":
        from mujoco import viewer

        viewer_ctx = viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)

    loop_dt = model.opt.timestep * config.mujoco_steps_per_frame
    wall0, sim0, last_report = time.time(), data.time, time.time()
    shm.signal_ready()
    logger.info(f"lot sim ready: robot {robot} at {start[0]:.1f}, {start[1]:.1f}; headless={viewer_ctx is None}")
    try:
        while not shm.should_stop() and (viewer_ctx is None or viewer_ctx.is_running()):
            t0 = time.time()
            inbox.drain(model, data)
            for _ in range(config.mujoco_steps_per_frame):
                mujoco.mj_step(model, data)
            person.tick(data)
            if viewer_ctx is not None:
                viewer_ctx.sync()
            shm.write_odom(data.qpos[0:3].copy(), data.qpos[3:7].copy(), time.time())
            sensors.publish_state(data)
            now = time.time()
            if now - last_report >= 30.0:
                rtf = (data.time - sim0) / max(1e-6, now - wall0)
                logger.info(f"lot sim: real-time factor {rtf:.2f}, base at {data.qpos[0]:.2f}, {data.qpos[1]:.2f}")
                last_report = now
            # Real time: one loop advances mujoco_steps_per_frame physics steps.
            sleep = loop_dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        sensors.stop_evt.set()
        sensors.join(timeout=5.0)
        person.stop()
        inbox.stop()
        if viewer_ctx is not None:
            viewer_ctx.close()


def main(argv: list[str]) -> None:
    config: GlobalConfig = pickle.loads(base64.b64decode(argv[1]))
    # Make the parent's settings this process's global config too: DimOS's PersonPositionController and the
    # shared zenoh session read the global one, and must join the same bus (e.g. --zenoh-scout-addr).
    from dimos.core.global_config import global_config

    global_config.update(**{k: getattr(config, k) for k in type(config).model_fields})
    shm = ShmReader(json.loads(argv[2]))

    def _stop(_signum: int, _frame: Any) -> None:
        shm.signal_stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        run(config, shm)
    finally:
        shm.cleanup()


if __name__ == "__main__":
    main(sys.argv)
