"""The Go2 connection for the warehouse: DimOS's MuJoCo Go2 connection, launching the warehouse scene process.

Only the subprocess it starts differs from dimos.robot.unitree.mujoco_connection.MujocoConnection (adapted from its
start(), Copyright Dimensional Inc., Apache-2.0): same shared memory, streams and commands, except that `move()`
raises slow in-place turns to MIN_PURE_TURN_RAD_S (the Go1 policy barely turns in place below it).
"""
from __future__ import annotations

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

from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.mujoco_connection import MujocoConnection
from dimos.simulation.mujoco.shared_memory import ShmWriter
from dimos.utils.logging_config import setup_logger

logger = setup_logger()

PROCESS = Path(__file__).resolve().with_name("sim_process.py")
MIN_PURE_TURN_RAD_S = 0.8  # same floor as the lot's Go2 connection and the fleet's Go2 spec


def _mjpython() -> str:
    """macOS needs mjpython for MuJoCo's passive viewer; prefer the one next to this interpreter (works without an
    activated venv), else whatever is on PATH."""
    if sys.platform != "darwin":
        return sys.executable
    local = Path(sys.executable).with_name("mjpython")
    return str(local) if local.exists() else "mjpython"


class WarehouseMujocoConnection(MujocoConnection):
    def start(self) -> None:
        self.shm_data = ShmWriter()
        config_pickle = base64.b64encode(pickle.dumps(self.global_config)).decode("ascii")
        shm_names_json = json.dumps(self.shm_data.shm.to_names())
        env = os.environ.copy()
        if sys.platform == "darwin":
            libdir = Path(sysconfig.get_config_var("LIBDIR") or "")
            if libdir.is_dir():
                existing = env.get("DYLD_LIBRARY_PATH", "")
                env["DYLD_LIBRARY_PATH"] = f"{libdir}:{existing}" if existing else str(libdir)
        try:
            self.process = subprocess.Popen(
                [_mjpython(), str(PROCESS), config_pickle, shm_names_json],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
            )
        except Exception as e:
            self.shm_data.cleanup()
            raise RuntimeError(f"Failed to start the warehouse MuJoCo process: {e}") from e
        self._output_thread = threading.Thread(target=self._pump_subprocess_output, name="mujoco-output-pump",
                                               daemon=True)
        self._output_thread.start()
        deadline = time.time() + 300.0
        while time.time() < deadline:
            if self.process.poll() is not None:
                code = self.process.returncode
                self.stop()
                raise RuntimeError(f"warehouse MuJoCo process failed to start (exit code {code})")
            if self.shm_data.is_ready():
                logger.info("warehouse MuJoCo process started")
                return
            time.sleep(0.1)
        self.stop()
        raise RuntimeError("warehouse MuJoCo process failed to start (timeout)")

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        wz = twist.angular.z
        if math.hypot(twist.linear.x, twist.linear.y) < 0.05 and 0.05 < abs(wz) < MIN_PURE_TURN_RAD_S:
            twist = Twist(linear=Vector3(twist.linear.x, twist.linear.y, twist.linear.z),
                          angular=Vector3(twist.angular.x, twist.angular.y, math.copysign(MIN_PURE_TURN_RAD_S, wz)))
        return super().move(twist, duration)


class WarehouseGo2Connection(GO2Connection):
    """GO2Connection whose simulator is the warehouse (blueprint sets simulation=mujoco)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(self.connection, MujocoConnection):
            raise RuntimeError("go2-warehouse is simulation only; its blueprint sets simulation='mujoco'")
        self.connection = WarehouseMujocoConnection(self.config.g)
