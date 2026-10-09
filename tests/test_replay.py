"""Determinism: same commands -> bit-identical state, with the machine idle or every core busy; a live threaded run
replays to every logged state hash; the replay CLI."""
from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from dimos_worlds.fleet.world import CTRL_DT, STATE_EVERY_TICKS, warehouse_world
from dimos_worlds.replay import RunLog, replay
from dimos_worlds.replay.replay import main as replay_main
from dimos_worlds.warehouse import cell


def _spin(seconds: float) -> None:
    end = time.time() + seconds
    x = 0
    while time.time() < end:
        x += 1


def _run_script(load: bool) -> bytes:
    burners = [multiprocessing.Process(target=_spin, args=(120.0,)) for _ in range(os.cpu_count() or 4)] if load \
        else []
    for b in burners:
        b.start()
    try:
        w = warehouse_world()
        loc = cell.BY_ID["shelf-c"]
        w.walk_to("g1", *loc.stand_point(*loc.center), 0.5)
        w.walk_route("go2", [(8.0, 11.4, 0.0), (8.0, 11.8, 1.57)], 0.6)
        w.step(int(4.0 / CTRL_DT))
        w.set_velocity("go2", 0.3, 0.0, 0.4)
        w.step(int(6.0 / CTRL_DT))
        w.stop("go2")
        w.step(int(2.0 / CTRL_DT))
        state = b"".join(w.bodies[r].data.qpos.tobytes() + w.bodies[r].data.qvel.tobytes() for r in sorted(w.bodies))
        w.close()
        return state
    finally:
        for b in burners:
            b.terminate()
            b.join()


def test_same_commands_same_state_idle_or_loaded() -> None:
    idle = _run_script(load=False)
    loaded = _run_script(load=True)
    assert idle == loaded
    assert np.isfinite(np.frombuffer(idle, dtype=np.float64)).all()


def test_live_threaded_run_replays_to_every_logged_hash(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    w = warehouse_world(run_log=RunLog(path), speed=4.0)
    w.start()
    try:
        loc = cell.BY_ID["shelf-d"]
        w.walk_to("g1", *loc.stand_point(*loc.center), 0.5)
        time.sleep(0.4)
        w.walk_route("go2", [(10.0, 11.4, 0.0)], 0.6)
        time.sleep(0.9)
        w.set_velocity("go2", 0.2, 0.0, -0.3)  # wall-clock moments: whatever tick they land on is logged
        time.sleep(0.5)
        w.stop("go2")
        w.stop("g1")
        while w.ticks < 2 * STATE_EVERY_TICKS + 5:
            time.sleep(0.05)
    finally:
        w.close()
    run = RunLog.read(path)
    assert len(run.states()) >= 2
    assert len(run.commands()) == 5
    res = replay(path)
    assert res.ok, f"diverged at {res.diverged_at}"
    assert res.checked == len(run.states())
    assert replay_main([str(path)]) == 0


def test_tampered_log_is_caught(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    w = warehouse_world(run_log=RunLog(path))
    w.walk_route("go2", [(6.0, 11.4, 0.0)], 0.6)
    w.step(STATE_EVERY_TICKS)
    w.close()
    lines = path.read_text().splitlines()
    lines = [ln.replace('"6.0', '"6.5').replace("[6.0,", "[6.5,") for ln in lines]
    path.write_text("\n".join(lines) + "\n")
    res = replay(path)
    assert not res.ok and res.diverged_at == STATE_EVERY_TICKS
