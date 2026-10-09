"""Re-execute a shared-world run from its log and check it reaches every state hash the run logged.

    dimos-worlds-replay RUN.jsonl [--until TICK] [--scene PATH]
    python -m dimos_worlds.replay RUN.jsonl

The world is rebuilt from the log's header (scene + robot specs), every logged command is re-issued at the tick it
landed on, and the world is stepped by hand. Exit status 0 when every logged hash matches, 1 at the first divergence.

What can make an honest replay diverge: a different MuJoCo, ONNX Runtime or numpy build, or another CPU
architecture (floating point is only bit-identical on the same stack); the header records the versions the run
used. A scene file that changed since the run is refused (its sha256 is in the header).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import sys
from typing import Any

from dimos_worlds.replay.log import RunLog

SCENES = {"warehouse": "dimos_worlds.warehouse.scene:SCENE_XML"}


@dataclass
class ReplayResult:
    ticks: int
    final_hash: str
    checked: int  # logged hashes compared
    diverged_at: int | None  # first tick whose hash differs, or None
    events: list[dict[str, Any]] = field(default_factory=list)  # falls/recoveries seen in the replay

    @property
    def ok(self) -> bool:
        return self.diverged_at is None


def _scene_xml(world: dict[str, Any], scene_path: Path | None) -> str:
    if scene_path is not None:
        xml = scene_path.read_text()
    else:
        name = world.get("scene")
        if name not in SCENES:
            raise SystemExit(f"run used scene {name!r}; pass --scene PATH to its MJCF")
        module, attr = SCENES[name].split(":")
        import importlib

        xml = Path(getattr(importlib.import_module(module), attr)).read_text()
    want = world.get("scene_sha256")
    got = hashlib.sha256(xml.encode()).hexdigest()
    if want and got != want:
        raise SystemExit(f"scene differs from the run's (sha256 {got[:12]} != {want[:12]}); cannot replay")
    return xml


def replay(log_or_path: RunLog | str | Path, *, until: int | None = None, scene_path: Path | None = None,
           stop_at_divergence: bool = True) -> ReplayResult:
    from dimos_worlds.fleet.robots import RobotSpec
    from dimos_worlds.fleet.world import SharedWorld

    run = log_or_path if isinstance(log_or_path, RunLog) else RunLog.read(log_or_path)
    header = run.header()
    assert header is not None
    world_desc = header["world"]
    specs = tuple(RobotSpec.from_dict(r) for r in world_desc["robots"])
    cmds = sorted(run.commands(), key=lambda c: c["tick"])  # stable: same-tick commands keep log order
    logged = run.states()
    last = max([0, *[c["tick"] for c in cmds], *logged])
    end = until if until is not None else last
    replay_log = RunLog()
    world = SharedWorld(specs, _scene_xml(world_desc, scene_path), scene_name=world_desc.get("scene", "custom"),
                        run_log=replay_log)
    diverged: int | None = None
    checked = 0
    i = 0
    try:
        while world.ticks < end:
            while i < len(cmds) and cmds[i]["tick"] <= world.ticks:
                world.apply(cmds[i])
                i += 1
            world.tick()
            want = logged.get(world.ticks)
            if want is not None:
                checked += 1
                if diverged is None and want != world.state_hash():
                    diverged = world.ticks
                    if stop_at_divergence:
                        break
        events = [e for e in replay_log.lines if e.get("t") == "event"]
        return ReplayResult(world.ticks, world.state_hash(), checked, diverged, events)
    finally:
        world.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dimos-worlds-replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="run log (JSONL) written by a shared world")
    ap.add_argument("--until", type=int, default=None, help="stop at this tick (default: the last logged tick)")
    ap.add_argument("--scene", type=Path, default=None, help="scene MJCF, when the run used a custom scene")
    ap.add_argument("--keep-going", action="store_true", help="do not stop at the first divergence")
    a = ap.parse_args(argv)
    run = RunLog.read(a.log)
    header = run.header() or {}
    robots = ", ".join(r["id"] for r in header.get("world", {}).get("robots", []))
    print(f"run: scene {header.get('world', {}).get('scene')}, robots {robots}, {len(run.commands())} commands, "
          f"{len(run.states())} state hashes; recorded with {header.get('versions')}")
    res = replay(run, until=a.until, scene_path=a.scene, stop_at_divergence=not a.keep_going)
    for e in res.events:
        print(f"  tick {e['tick']}: {e['robot']} {e['what']}")
    if res.ok:
        print(f"OK: {res.checked} state hashes match; final state {res.final_hash} at tick {res.ticks}")
        return 0
    print(f"DIVERGED at tick {res.diverged_at} ({res.checked} hashes compared)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
