# dimos-worlds

Sim worlds for [DimOS](https://github.com/dimensionalOS/dimos), shipped as DimOS external blueprints.

- **Night vehicle lot.** Outdoor lot with six fixed CCTV cameras (distortion, noise, JPEG-style ISP), named places, and scripted events: person appears, gate opens, vehicle moves.
- **Warehouse.** 30 x 20 m building with pallet racking, pallets, conveyor and pick table, built from a layout file, with a known-occupancy prior map for DimOS's planner.
- **Shared-world fleet.** A Go2 and a G1 in one MuJoCo world, each on its own policy timestep, each with its own DimOS planner, and costmaps that show each robot the other.
- **Deterministic replay.** A command log plus periodic state hashes. A sim run re-executes bit-identically.

![Six lot CCTV cameras at night, MuJoCo render through the camera model](docs/media/lot-cctv.jpg)

*The lot's six CCTV cameras at night, gate 2 open (MuJoCo render + `camera_model.isp`; boxes and flat lighting, not photoreal).*

![Go2 and G1 walking the warehouse cross aisle under DimOS's planner](docs/media/warehouse-fleet.gif)

*`warehouse-fleet`: G1 and Go2 each sent a goal over `*/goal_request`, each driven by its own `ReplanningAStarPlanner`. The GIF is that run's log re-executed and rendered (`docs/media/make_media.py`), sped up about 7x.*

## Why

- DimOS ships two scenes, `office` and `supermarket`. No outdoor, no night, no fixed cameras.
- DimOS multi-robot sim runs one MuJoCo sim per robot, so robots cannot see or collide with each other. Here they share one world.
- DimOS replay replays recorded sensor streams. It does not re-execute a sim run. This does, and checks it with state hashes.

## Status

Early. Targets `dimos==0.0.14`. A CI job tracks DimOS `main` and is allowed to fail. G1 walks only (no arms, no manipulation). See [Known limitations](#known-limitations).

## Install

Python 3.12. Not on PyPI yet; install from a checkout:

```
git clone <this repo> && cd dimos-worlds
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
```

On first use DimOS downloads its MuJoCo robot models and walking policies (`mujoco_sim` data) and MuJoCo Menagerie.

Headless Linux: `export MUJOCO_GL=egl` (untested here, see limitations).

## Run

Blueprints register under the `dimos.blueprints` entry-point group, so DimOS finds them by distribution name:

```
$ dimos list | grep dimos-worlds
  dimos-worlds.go2-lot-night
  dimos-worlds.go2-warehouse
  dimos-worlds.lot-cctv
  dimos-worlds.warehouse-fleet
```

```
dimos --viewer none run dimos-worlds.go2-warehouse      # Go2 in the warehouse: DimOS's Go2 stack on the warehouse prior map
dimos --viewer none run dimos-worlds.warehouse-fleet    # Go2 + G1 in one shared world, one planner each
dimos --viewer none run dimos-worlds.go2-lot-night      # Go2 in the night lot: prior map, named places, events
dimos --viewer none run dimos-worlds.go2-lot-night dimos-worlds.lot-cctv   # ... plus the six CCTV cameras
```

Drop `--viewer none` for the Rerun viewer. Run one DimOS instance at a time (`dimos status`).

Send a goal from another terminal:

```
# go2-warehouse
dimos topic send /goal_request 'PoseStamped(frame_id="world", position=Vector3(12, 11.4, 0), orientation=Quaternion(0, 0, 0, 1))'

# warehouse-fleet: one goal topic per robot
dimos topic send /g1/goal_request  'PoseStamped(frame_id="world", position=Vector3(6, 11, 0), orientation=Quaternion(0, 0, 0, 1))'
dimos topic send /go2/goal_request 'PoseStamped(frame_id="world", position=Vector3(12, 11.4, 0), orientation=Quaternion(0, 0, 0, 1))'
```

Watch odometry with `dimos topic echo /odom PoseStamped` (go2-warehouse), or `dimos topic echo /go2/odom PoseStamped` and `/g1/odom` (warehouse-fleet). Give the type name explicitly. A fresh `dimos topic` process needs a few seconds to discover the running peers. A `send` right after start can be lost, and an `echo` can print nothing for 10-20 s: resend, or echo longer. The planner logs `Got new goal` when a goal lands.

Warehouse coordinates: metres, origin at the south-west inside corner, x east, y north. The cross aisle runs along y = 11.4. The Go2 starts at (3, 11.4), the G1 at (1, 11).

## Replay

`warehouse-fleet` writes `warehouse-fleet-run.jsonl` in the working directory: a header (scene sha256, robot specs, versions), every command with the tick it landed on, and a state hash every 250 ticks (5 s of sim time).

```
$ dimos-worlds-replay warehouse-fleet-run.jsonl
run: scene warehouse, robots go2, g1, 411 commands, 26 state hashes; recorded with {'mujoco': '3.15.0', 'dimos': '0.0.14', 'onnxruntime': '1.31.0', 'numpy': '2.5.3', 'dimos-worlds': '0.1.0'}
OK: 26 state hashes match; final state f9c8c7960d6bee00 at tick 6500
```

`python -m dimos_worlds.replay RUN.jsonl` does the same. Exit status is 0 when every hash matches and 1 at the first divergence. `--until TICK` stops early.

## Cook your own scene

Both worlds are generated, then cooked into a DimOS scene package: MJCF + `places.json` + `scene.meta.json`, the last written with DimOS's own `ScenePackage.write_metadata`.

```
python -m dimos_worlds.warehouse.gen_layout   # layout.json: building, racks, pallets, places, cameras
python -m dimos_worlds.warehouse.scene        # cooks warehouse/scene/ from layout.json
python -m dimos_worlds.lot.gen_lot            # cooks lot/scene/
```

To make your own: copy one of these packages, change the generator and cook it. Register a blueprint in your own `pyproject.toml` under `[project.entry-points."dimos.blueprints"]`, then `dimos run <your-dist>.<blueprint>`. `tests/test_warehouse.py` shows the checks a scene package should pass: DimOS loads it, it is static, and it composes with DimOS's Go1/G1 models.

Optional Blender renders (`*/render/`, `*/assets/fetch.py`) are ported but not run in this release. Poly Haven assets (CC0) are fetched at build time, never committed.

## Known limitations

- **Verified on macOS (Apple Silicon) only.** Linux/EGL and CI have not been run yet.
- **Blender render path untested.** `render/` and `assets/fetch.py` compile; nothing has been rendered with them in this repo.
- **Bit-exact replay needs the same stack.** Replay needs the MuJoCo, ONNX Runtime, numpy and CPU architecture the run was recorded on (the log header records them). A changed scene file is refused.
- **Policy dead bands are compensated, not fixed.** DimOS's Go1/Go2 walking policy barely turns in place below about 0.8 rad/s and does not walk forward below about 0.3. Slow in-place turns from `cmd_vel` are raised to 0.8 rad/s (go2-warehouse, warehouse-fleet), and the fleet's route controller has its own floors. The G1 policy walks about 1.6x the command, under-turns while walking and creeps when standing. So in warehouse-fleet its `cmd_vel` is tracked as a velocity (feed-forward + PI on measured velocity), and it holds its pose at zero command. These numbers were measured in this sim, not on hardware.
- **Planners do not coordinate.** Each robot's planner sees the other robot in its costmap (within 3 m) and replans around it, nothing more. When two robots meet in an aisle, one planner can give up after its replan limit; send the goal again.
- **Slow corners look "stuck" to DimOS.** Its sim stuck check (under 1 m in 8 s) fires repeatedly while the Go2 turns around rack ends. The planner replans each time and still arrives: 2 min 10 s from (3, 11.4) to (15, 7.5) in one verified run.
- **Goal heading is loose.** DimOS's planner can report arrival with the G1 tens of degrees off the goal heading.
- **No arm yet.** The warehouse has the arm cell and pedestal; no arm is simulated.
- **MuJoCo visuals are boxes with flat lighting.** Fine for pipelines and change detection, not for judging what a real night camera sees.
- **Lot CCTV frames are rendered, not recorded.** Camera model parameters are typical datasheet values, UNVERIFIED against any specific camera.

## Credits

DimOS (Dimensional Inc., Apache-2.0), MuJoCo Menagerie Unitree and Drake models (BSD-3-Clause), Poly Haven assets (CC0). Full list in [NOTICE](NOTICE).

## License

Apache-2.0. See [LICENSE](LICENSE).

Built by Isaiah Bjorklund.
