# Changelog

## Unreleased

- Demo plant (`dimos_worlds.plant`): process plant with 7 analog gauges and scripted instrument events, prior map, `go2-plant` blueprint, Blender render service (`dimos-worlds-plant-render`, `/v1/view`), and a gauge-rounds loop that drives the sim Go2 from a gauge reader's agent (the loop needs gauge-rounds, a separate package that is not public).
- Safety watchdog (`dimos_worlds.safety`): a DimOS module (`SafetyWatchdog`) between a robot's command source and the robot. Stops and latches on a fall or tilt (IMU roll/pitch, or odometry orientation without an IMU; Unitree lowstate `imu_state.rpy` via `Watchdog.lowstate` / the `lowstate` RPC), on a lost command link (motion asked for, then no command for 0.5 s), and on stale odometry or IMU. Zero velocity until an explicit reset, which is refused while the robot is tilted or its state is stale. Alerts as JSON on `safety_alert` and as Python callbacks. Tested against a walking Go2 in MuJoCo: tipped over, link cut, odometry cut.
- Coordination for three and four robots: a hold cycle (A holds for B, B for C, C for A) is broken after 5 s by the lowest-priority robot stepping aside for the one holding for it; pockets keep off every other robot's path and body. Stepped scenarios `three-way`, `convoy` and `ring` (Go2, G1 and up to two more Go2 bodies).
- Navigation-level coordination for robots that each run their own DimOS planner (`dimos_worlds.coord`): right of way from planned paths, hold short of a shared stretch, yield to a pocket in head-on meetings, automatic goal retries when a planner gives up, deadlock swap. `traffic_coordinator(ids, name)` builds the DimOS module for any robot ids; `warehouse-fleet` routes goals through it (`{id}/goal_request` -> `{id}/nav_goal`) and publishes `{id}/arrived`.
- Settling at the goal after the planner reports arrival: the G1 closes position and heading to 0.12 m / 6 degrees, the Go2 its heading.
- Scripted head-on and crossing scenarios against DimOS's planners (`python -m dimos_worlds.coord.scenarios`), in-process fleet harness (`coord.inproc`).
- The in-process fleet steps on sim time by default (`coord.stepped`: DimOS's GlobalPlanner and LocalPlanner loops run once per 0.1 s from the world loop, coordinator via `Coordinator.run_once`): same result under any machine load. `--realtime` / `DIMOS_WORLDS_REALTIME=1` keeps the wall-clock variant.
- G1 falls fixed in the scripted scenarios (27 of 300 jittered runs before, 0 of 300 after): above 0.2 m/s forward its tracked vx command stays at the policy's neutral instead of braking backwards (`world.NEUTRAL_VX`), and its DimOS planner runs at 0.4 speed (`RobotSpec.nav_speed`). Changes G1 state hashes.
- `warehouse-fleet` runs each robot's planner at that robot's `nav_speed` (`fleet.planner.FleetPlanner`, a thin subclass of DimOS's `ReplanningAStarPlanner` with a per-module speed setting); command-line overrides still apply. Before, a live `dimos run` drove the G1's planner at full speed.
- Dead-band compensation is explicit per-robot config (`robots.Compensation`, in run log headers; logs from 0.1.0 still replay), switchable with `WarehouseFleetSimConfig.compensate`.
- Optional arm cell: KUKA iiwa 14 (MuJoCo Menagerie, fetched) on the warehouse pedestal, positional IK pick and place of a tote, `warehouse-arm` blueprint.

## 0.1.0 (unreleased)

- Night vehicle lot scene with CCTV camera model, named places, person/gate/vehicle events.
- Warehouse scene built from a layout file, with a known-occupancy prior map for DimOS's planner.
- Shared-world fleet sim: Go2 and G1 in one MuJoCo world, per-robot timesteps, one DimOS planner per robot, robot-aware costmaps.
- G1 `cmd_vel` tracked as a velocity, with pose hold at zero command; Go2 slow in-place turns raised past the policy dead band.
- Deterministic replay: command log, periodic state hashes, re-execution check (`dimos-worlds-replay`).
- DimOS external blueprints: `go2-lot-night`, `lot-cctv`, `go2-warehouse`, `warehouse-fleet`.
- Targets `dimos==0.0.14`. Verified on macOS (Apple Silicon) and Ubuntu 22.04 x86_64 (NVIDIA EGL); CI on GitHub's Ubuntu and macOS runners.
- `dimos-worlds-goal` and `dimos-worlds-send`: publish after peer discovery (`dimos topic send` loses the message on Linux).
- `go2-warehouse` runs headless by default (`DIMOS_WORLDS_VIEWER=1` for MuJoCo's viewer) and keeps real time without the viewer.
- Run logs record OS/CPU architecture; replay names the stack difference when it diverges.
- Blender render path run on Linux: Menagerie Go2 in the warehouse render, README stills from `docs/media/render_heroes.py`.
- Imports DimOS's planner from its new location on DimOS main.
