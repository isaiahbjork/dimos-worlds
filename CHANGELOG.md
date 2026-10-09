# Changelog

## 0.1.0 (unreleased)

- Night vehicle lot scene with CCTV camera model, named places, person/gate/vehicle events.
- Warehouse scene built from a layout file, with a known-occupancy prior map for DimOS's planner.
- Shared-world fleet sim: Go2 and G1 in one MuJoCo world, per-robot timesteps, one DimOS planner per robot, robot-aware costmaps.
- G1 `cmd_vel` tracked as a velocity, with pose hold at zero command; Go2 slow in-place turns raised past the policy dead band.
- Deterministic replay: command log, periodic state hashes, re-execution check (`dimos-worlds-replay`).
- DimOS external blueprints: `go2-lot-night`, `lot-cctv`, `go2-warehouse`, `warehouse-fleet`.
- Targets `dimos==0.0.14`. Verified on macOS only.
