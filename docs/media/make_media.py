"""Regenerate the README media (MuJoCo renders, no Blender).

    python docs/media/make_media.py lot                      # lot-cctv.jpg: the six lot CCTV cameras at night
    python docs/media/make_media.py fleet RUN.jsonl          # warehouse-fleet.gif: a warehouse-fleet run, re-executed

`fleet` replays a run log written by `dimos run dimos-worlds.warehouse-fleet` (the same re-execution
`dimos-worlds-replay` checks) and renders it from a chase camera in the G1's world, where the Go2 is a proxy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent


def lot() -> Path:
    import mujoco

    from dimos_worlds.lot import events
    from dimos_worlds.lot.cctv import CctvRenderer

    r = CctvRenderer()
    events.gate_open(r.model, r.data, "gate-2")
    events.person_at(r.model, r.data, "row-c")
    mujoco.mj_forward(r.model, r.data)
    sheet = Image.new("RGB", (960, 360))
    for i in range(6):
        frame = Image.fromarray(r.frame(f"cam{i + 1}", ("readme",))).resize((320, 180), Image.LANCZOS)
        sheet.paste(frame, ((i % 3) * 320, (i // 3) * 180))
    r.close()
    path = OUT / "lot-cctv.jpg"
    sheet.save(path, quality=85)
    return path


def fleet(run_path: str, *, first_tick: int = 950, last_tick: int = 4150, every: int = 40) -> Path:
    import mujoco

    from dimos_worlds.fleet.robots import RobotSpec
    from dimos_worlds.fleet.world import SharedWorld
    from dimos_worlds.replay.log import RunLog
    from dimos_worlds.replay.replay import _scene_xml

    run = RunLog.read(run_path)
    header = run.header()
    assert header is not None
    desc = header["world"]
    world = SharedWorld(tuple(RobotSpec.from_dict(r) for r in desc["robots"]), _scene_xml(desc, None))
    cmds = sorted(run.commands(), key=lambda c: c["tick"])
    body = world.bodies["g1"]
    renderer = mujoco.Renderer(body.model, 234, 416)
    data = mujoco.MjData(body.model)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 8.5, 180.0, -50.0
    frames: list[Image.Image] = []
    i = 0
    while world.ticks < last_tick:
        while i < len(cmds) and cmds[i]["tick"] <= world.ticks:
            world.apply(cmds[i])
            i += 1
        world.tick()
        if world.ticks >= first_tick and world.ticks % every == 0:
            poses = world.poses()
            cam.lookat[:] = [float(np.mean([p[0] for p in poses.values()])), 11.1, 0.3]
            world.snapshot("g1", data)
            renderer.update_scene(data, camera=cam)
            frames.append(Image.fromarray(renderer.render()))
    renderer.close()
    world.close()
    path = OUT / "warehouse-fleet.gif"
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=120, loop=0, optimize=True)
    return path


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "lot":
        print(lot())
    elif what == "fleet" and len(sys.argv) > 2:
        print(fleet(sys.argv[2]))
    else:
        sys.exit(__doc__)
