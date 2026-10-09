"""MuJoCo process for `go2-warehouse`: DimOS's own Go2 sim loop with the warehouse as its scene.

DimOS 0.0.14's Go2 simulator (dimos.simulation.mujoco.mujoco_process) builds its scene from `mujoco_room` names in
DimOS's data directory only. This script runs that same loop, in its own process, with the scene loader pointed at
the warehouse scene package. Nothing in the installed dimos package is modified.

Launched by WarehouseMujocoConnection with the same arguments DimOS's launcher takes: a pickled GlobalConfig and the
shared-memory names.
"""
from __future__ import annotations

import base64
import json
import pickle
import signal
import sys
from typing import Any

import dimos.simulation.mujoco.mujoco_process as dimos_process
from dimos.simulation.mujoco.shared_memory import ShmReader

from dimos_worlds.warehouse.scene import SCENE_XML


def _warehouse_scene_xml(_config: Any) -> str:
    return SCENE_XML.read_text()


def main() -> None:
    dimos_process.load_scene_xml = _warehouse_scene_xml  # this process only
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
