# Changelog

## Unreleased

- Navigation-level coordination for robots that each run their own DimOS planner (`dimos_worlds.coord`): right of way from planned paths, hold short of a shared stretch, yield to a pocket in head-on meetings, automatic goal retries when a planner gives up, deadlock swap. `traffic_coordinator(ids, name)` builds the DimOS module for any robot ids; `warehouse-fleet` routes goals through it (`{id}/goal_request` -> `{id}/nav_goal`) and publishes `{id}/arrived`.
- Settling at the goal after the planner reports arrival: the G1 closes position and heading to 0.12 m / 6 degrees, the Go2 its heading.
- Scripted head-on and crossing scenarios against DimOS's planners (`python -m dimos_worlds.coord.scenarios`), in-process fleet harness (`coord.inproc`).
- Dead-band compensation is explicit per-robot config (`robots.Compensation`, in run log headers; logs from 0.1.0 still replay), switchable with `WarehouseFleetSimConfig.compensate`.
- Optional arm cell: KUKA iiwa 14 (MuJoCo Menagerie, fetched) on the warehouse pedestal, positional IK pick and place of a tote, `warehouse-arm` blueprint.

## 0.1.0 (unreleased)

- Night vehicle lot scene with CCTV camera model, named places, person/gate/vehicle events.
- Warehouse scene built from a layout file, with a known-occupancy prior map for DimOS's planner.
- Shared-world fleet sim: Go2 and G1 in one MuJoCo world, per-robot timesteps, one DimOS planner per robot, robot-aware costmaps.
- G1 `cmd_vel` tracked as a velocity, with pose hold at zero command; Go2 slow in-place turns raised past the policy dead band.
- Deterministic replay: command log, periodic state hashes, re-execution check (`dimos-worlds-replay`).
- DimOS external blueprints: `go2-lot-night`, `lot-cctv`, `go2-warehouse`, `warehouse-fleet`.
- Targets `dimos==0.0.14`. Verified on macOS only.
