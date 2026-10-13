"""Drop the Dock 3-class station (and its M4D-class quadcopter) into any MuJoCo env.

    import mujoco
    from dimos_worlds.dock.fixture import attach_dock
    from dimos_worlds.dock.sim import DockSim
    spec = mujoco.MjSpec.from_file("my_env.xml")
    attach_dock(spec, pos=(-12.0, -23.0, 0.0), yaw_deg=90, prefix="dock_")
    model = spec.compile()
    sim = DockSim.from_model(model, prefix="dock_")   # same open/close/takeoff/land/feeds API

    python -m dimos_worlds.dock.fixture --env my_env.xml --pos -12 -23 0 --yaw 90   # smoke test: attach, compile, fly

Everything is renamed with `prefix` (bodies, joints, actuators, cameras, sensors, meshes), so several docks can
live in one env. The fixture has no ground or lights; the host env supplies them, and its timestep and integrator
(the station's own model uses 2 ms, implicitfast; it also runs at e.g. 5 ms Euler).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
FIXTURE_XML = HERE / "dock_fixture.xml"


def attach_dock(spec: mujoco.MjSpec, pos=(0.0, 0.0, 0.0), yaw_deg: float = 0.0, prefix: str = "dock_",
                fixture: Path = FIXTURE_XML) -> mujoco.MjSpec:
    """Attach the dock + drone into `spec` at `pos`, rotated `yaw_deg` about +Z. Returns `spec`."""
    from .gen_mjcf import ensure_assets

    ensure_assets()  # the meshes are generated (assets/ is untracked)
    child = mujoco.MjSpec.from_file(str(fixture))
    mesh_dir = fixture.parent / (child.meshdir or "")
    for mesh in child.meshes:  # absolute paths: the host env's meshdir must not apply to the dock's meshes
        if mesh.file and not Path(mesh.file).is_absolute():
            mesh.file = str(mesh_dir / mesh.file)
    child.option.timestep = spec.option.timestep  # the host's physics options win (attach keeps them anyway): say so
    child.option.integrator = spec.option.integrator  # here instead of in an attach-conflict warning
    h = math.radians(yaw_deg) / 2
    frame = spec.worldbody.add_frame(pos=list(pos), quat=[math.cos(h), 0.0, 0.0, math.sin(h)])
    spec.attach(child, prefix=prefix, frame=frame)
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(description="attach the dock into an env and compile")
    ap.add_argument("--env", type=Path, required=True)
    ap.add_argument("--pos", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--prefix", default="dock_")
    a = ap.parse_args()
    spec = attach_dock(mujoco.MjSpec.from_file(str(a.env)), a.pos, a.yaw, a.prefix)
    m = spec.compile()
    from .sim import DockSim

    sim = DockSim.from_model(m, prefix=a.prefix)
    sim.open()
    t = sim.run_until_idle()
    sim.takeoff(2.0)
    sim.land()
    sim.close()
    sim.run_until_idle()
    print(f"{a.env.name}: {m.nbody} bodies; dock at {list(a.pos)} yaw {a.yaw}: opened in {t:.2f} s, "
          f"launch+land ok, state {sim.state().state}, drone {sim.state().drone}, "
          f"max penetration {sim.max_pen * 1000:.1f} mm, off-pad contacts {sorted(sim.bad_contacts) or 'none'}")


if __name__ == "__main__":
    main()
