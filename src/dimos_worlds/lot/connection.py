"""DimOS Go2 connection whose MuJoCo child runs the lot.

`LotMujocoConnection` is DimOS's `MujocoConnection` with one change: `start()` launches
`dimos_worlds.lot.sim_process` instead of DimOS's `mujoco_process.py`. Shared memory, odom/video/lidar
streams and shutdown are DimOS's code; `move()` only raises slow in-place turns (see MIN_PURE_TURN_RAD_S).
`LotGo2Connection` is DimOS's `GO2Connection` module built on that connection, so every port (cmd_vel, odom, color_image, lidar, tf, ...) is unchanged.

Headless by default. On macOS the child runs with the venv's own python (DimOS needs `mjpython` only for
its passive viewer); `DIMOS_WORLDS_VIEWER=1` opens that viewer and then uses `mjpython` like DimOS.
"""

from __future__ import annotations

import atexit
import base64
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import sysconfig
import threading
import time
from typing import Any
import weakref

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.unitree.go2.connection import GO2Connection, _prefixed
from dimos.robot.unitree.mujoco_connection import MujocoConnection
from dimos.simulation.mujoco.shared_memory import ShmWriter
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

READY_TIMEOUT_S = 300.0
# DimOS's Go1 walking policy barely turns in place at low rates (measured in the lot, 4 s per rate:
# 0.2 rad/s commanded -> 0.01 actual, 0.5 -> 0.17, 0.8 -> 0.60, 1.0 -> 0.83). DimOS's local planner turns in place
# at 0.2-0.5 rad/s, so it never finishes its initial rotation and declares the robot stuck. Pure turns are
# raised to this rate. Sim-only; a real Go2 does not need it.
MIN_PURE_TURN_RAD_S = 0.8


class LotMujocoConnection(MujocoConnection):
    """MuJoCo Go2 connection that runs the night vehicle lot."""

    def start(self) -> None:
        self.shm_data = ShmWriter()
        config_pickle = base64.b64encode(pickle.dumps(self.global_config)).decode("ascii")
        shm_names_json = json.dumps(self.shm_data.shm.to_names())

        viewer = os.environ.get("DIMOS_WORLDS_VIEWER", "") == "1"
        executable = "mjpython" if (viewer and sys.platform == "darwin") else sys.executable
        env = os.environ.copy()
        if sys.platform == "darwin":
            libdir = Path(sysconfig.get_config_var("LIBDIR") or "")
            if libdir.is_dir():
                existing = env.get("DYLD_LIBRARY_PATH", "")
                env["DYLD_LIBRARY_PATH"] = f"{libdir}:{existing}" if existing else str(libdir)
        try:
            self.process = subprocess.Popen(
                [executable, "-m", "dimos_worlds.lot.sim_process", config_pickle, shm_names_json],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except Exception as e:
            self.shm_data.cleanup()
            raise RuntimeError(f"Failed to start the lot MuJoCo subprocess: {e}") from e

        self._output_thread = threading.Thread(
            target=self._pump_subprocess_output, name="lot-mujoco-output-pump", daemon=True
        )
        self._output_thread.start()

        t0 = time.time()
        while time.time() - t0 < READY_TIMEOUT_S:
            if self.process.poll() is not None:
                code = self.process.returncode
                self.stop()
                raise RuntimeError(f"lot MuJoCo process failed to start (exit code {code})")
            if self.shm_data.is_ready():
                logger.info("lot MuJoCo process started")
                weak_self = weakref.ref(self)

                def cleanup_on_exit(ref: weakref.ReferenceType[LotMujocoConnection] = weak_self) -> None:
                    instance = ref()
                    if instance is not None:
                        instance.stop()

                atexit.register(cleanup_on_exit)
                return
            time.sleep(0.1)
        self.stop()
        raise RuntimeError("lot MuJoCo process failed to start (timeout)")


    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        lin = math.hypot(twist.linear.x, twist.linear.y)
        wz = twist.angular.z
        if lin < 0.05 and 0.05 < abs(wz) < MIN_PURE_TURN_RAD_S:
            twist = Twist(
                linear=Vector3(twist.linear.x, twist.linear.y, twist.linear.z),
                angular=Vector3(twist.angular.x, twist.angular.y, math.copysign(MIN_PURE_TURN_RAD_S, wz)),
            )
        return super().move(twist, duration)


class LotGo2Connection(GO2Connection):
    """DimOS's GO2Connection, always on the lot simulator (ignores --simulation / --robot-ip)."""

    def __init__(self, **kwargs: Any) -> None:
        # Skip GO2Connection.__init__'s make_connection(): it would pick WebRTC/replay/DimOS's own sim.
        super(GO2Connection, self).__init__(**kwargs)
        self.connection = LotMujocoConnection(self.config.g)
        if hasattr(self.connection, "camera_info_static"):
            self.camera_info_static = self.connection.camera_info_static
        if self.config.frame_id_prefix and self.camera_info_static.frame_id:
            import copy

            self.camera_info_static = copy.copy(self.camera_info_static)
            self.camera_info_static.frame_id = _prefixed(self.config.frame_id_prefix, self.camera_info_static.frame_id)


__all__ = ["LotGo2Connection", "LotMujocoConnection"]
