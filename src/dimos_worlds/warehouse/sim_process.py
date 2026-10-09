"""MuJoCo process for `go2-warehouse`: DimOS's own Go2 sim loop with the warehouse as its scene.

DimOS 0.0.14's Go2 simulator (dimos.simulation.mujoco.mujoco_process) builds its scene from `mujoco_room` names in
DimOS's data directory only. This script runs that same loop, in its own process, with the scene loader pointed at
the warehouse scene package. Nothing in the installed dimos package is modified.

Launched by WarehouseMujocoConnection with the same arguments DimOS's launcher takes: a pickled GlobalConfig and the
shared-memory names.

Headless by default: DimOS 0.0.14's loop always opens MuJoCo's passive viewer, which needs a display (GLFW fails
on a headless Linux box) and `mjpython` on macOS. Here the viewer is replaced by a stand-in with no window unless
`DIMOS_WORLDS_VIEWER=1`, the same switch as the lot. The camera and lidar renderers are unchanged (MUJOCO_GL).
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import signal
import sys
import time
from types import SimpleNamespace
from typing import Any

import dimos.simulation.mujoco.mujoco_process as dimos_process
import mujoco
from dimos.simulation.mujoco.shared_memory import ShmReader

from dimos_worlds.warehouse.scene import SCENE_XML


def _warehouse_scene_xml(_config: Any) -> str:
    return SCENE_XML.read_text()


class _NoWindow:
    """What DimOS's loop uses of the passive viewer handle: a context manager with cam, is_running() and sync().

    sync() also keeps the sim at real time. DimOS's loop steps `mujoco_steps_per_frame` (7) physics steps per frame
    but sleeps for one; with the viewer, sync() blocks on the window and hides that. Without it the sim runs ahead of
    real time (measured on an RTX 3090 box: the Go2 crossed 9 m of aisle in about 3 s of wall time).
    """

    def __init__(self, data: mujoco.MjData) -> None:
        self.cam = SimpleNamespace(lookat=None, distance=0.0, azimuth=0.0, elevation=0.0)
        self._data = data
        self._t0: tuple[float, float] | None = None  # (wall, sim) at the first sync

    def __enter__(self) -> _NoWindow:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        now, sim = time.monotonic(), float(self._data.time)
        if self._t0 is None:
            self._t0 = (now, sim)
            return
        ahead = (sim - self._t0[1]) - (now - self._t0[0])
        if ahead > 0:
            time.sleep(ahead)

    def close(self) -> None:
        return None


def main() -> None:
    dimos_process.load_scene_xml = _warehouse_scene_xml  # this process only
    if os.environ.get("DIMOS_WORLDS_VIEWER", "") != "1":
        dimos_process.viewer = SimpleNamespace(launch_passive=lambda _m, d, **_k: _NoWindow(d))  # this process only
    config = pickle.loads(base64.b64decode(sys.argv[1]))
    shm = ShmReader(json.loads(sys.argv[2]))

    def _stop(_signum: int, _frame: Any) -> None:
        shm.signal_stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        dimos_process._run_simulation(config, shm)  # noqa: SLF001
    finally:
        shm.cleanup()


if __name__ == "__main__":
    main()
