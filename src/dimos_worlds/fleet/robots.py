"""The robots a shared world can hold, and where they start in the warehouse.

Each body is the model and walking policy DimOS itself uses in simulation, loaded from the installed dimos package's
data (never copied into this repo):

  quadruped   "unitree_go1"  DimOS's Go2 sim body (MuJoCo Menagerie Go1 meshes) + DimOS's Go1 ONNX policy,
                             physics step 5 ms, policy every 20 ms (dimos.simulation.mujoco.model.load_model)
  humanoid    "unitree_g1"   DimOS's unitree_g1.xml (Menagerie G1 meshes) + DimOS's G1 ONNX policy,
                             physics step 2 ms, policy every 20 ms, DimOS's drift compensation

Command units are the policy's own (DimOS passes cmd_vel straight through as the policy command), except for a body
whose Compensation has `track_velocity`, whose cmd_vel is a velocity it tracks. `cmd_gain` is the measured walking
speed per command unit.

Dead-band compensation (Compensation, one per robot, recorded in every run log header) works around how DimOS's
walking policies respond in this sim; it does not change the policies:

  Go2 (DimOS Go1 policy)  forward commands below ~0.3 do not move it, in-place turns below ~0.8 rad/s barely turn it.
                          cmd_vel in-place turns slower than `min_pure_turn` are raised to it; the route controller
                          raises its own commands to `min_vx` / `min_wz`.
  G1  (DimOS G1 policy)   walks ~1.45x the command, under-turns while walking, drifts sideways, creeps standing.
                          `track_velocity`: cmd_vel is a wanted body velocity, tracked closed-loop (feed-forward
                          1/cmd_gain + PI on measured velocity); a zero command holds the pose it stopped at.

`warehouse_fleet(compensate=False)` (WarehouseFleetSimConfig.compensate) turns all of it off: commands then reach the
policies as sent, which is what the policies were trained on and what to compare against. Numbers were measured in
this sim, not on hardware.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from dimos_worlds.warehouse.cell import LAYOUT


@dataclass(frozen=True)
class Compensation:
    """Policy dead-band compensation for one robot. All zero / False: commands pass straight through."""

    min_vx: float = 0.0  # route controller: a smaller forward command is raised to this (it would not move the body)
    min_wz: float = 0.0  # route controller: same for turning
    # cmd_vel: an in-place turn slower than this is raised to it (the Go1 policy barely turns below 0.8 rad/s in place,
    # so DimOS's planner sits in its initial rotation until it declares the robot stuck). 0 = off.
    min_pure_turn: float = 0.0
    # cmd_vel is the body's wanted m/s and rad/s, tracked closed-loop against its measured velocity (feed-forward
    # 1/cmd_gain + PI), and a zero command holds the pose. For a policy that walks faster than commanded, under-turns
    # while walking and drifts sideways, which makes a path follower veer off its path.
    track_velocity: bool = False

    @property
    def any(self) -> bool:
        return bool(self.min_vx or self.min_wz or self.min_pure_turn or self.track_velocity)


NO_COMPENSATION = Compensation()


@dataclass(frozen=True)
class RobotSpec:
    id: str
    kind: str  # quadruped | humanoid
    mujoco: str  # DimOS model / policy name: unitree_go1 | unitree_g1
    x: float
    y: float
    yaw: float = 0.0
    sim_dt: float = 0.005  # the physics step the policy was trained with
    ctrl_dt: float = 0.02  # policy period; also the world tick
    body_radius: float = 0.35  # footprint for costmaps (other robots' planners keep out of this)
    turn_diameter: float = 0.6
    max_vx: float = 0.6  # policy command units
    max_wz: float = 0.5  # rad/s
    acc_vx: float = 0.8  # command rate limits, per second of sim time (see world.py "Timing")
    acc_wz: float = 1.5
    cmd_gain: float = 1.0  # m/s walked per command unit (route controller and velocity tracking)
    fall_z: float = 0.15  # base height below which the body counts as fallen
    color: str = "#888888"
    comp: Compensation = field(default_factory=Compensation)
    # DimOS planner speed scale (GlobalConfig.nerf_speed): the local planner's top speed and turn rate, x this.
    # Applied by the in-process fleet (coord.inproc) and the warehouse-fleet blueprint (fleet.planner.FleetPlanner).
    nav_speed: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RobotSpec:
        d = dict(d)
        comp = dict(d.pop("comp", None) or {})
        for k in ("min_vx", "min_wz", "min_pure_turn", "track_velocity"):  # logs from 0.1.0 kept these flat
            if k in d:
                comp[k] = d.pop(k)
        return cls(**d, comp=Compensation(**comp))

    def without_compensation(self) -> RobotSpec:
        return replace(self, comp=NO_COMPENSATION)


# Go2 numbers measured in the warehouse (3 s at each command, DimOS Go1 policy, 5 ms steps): forward 0.1 and 0.2
# do not move it, 0.3 walks 0.21 m/s, 0.5 walks 0.49 m/s, 0.8 walks but yaws off course; turning 0.3 rad/s does not
# turn it, 0.6 turns 0.29 rad/s, 1.0 turns 0.86 rad/s.
def go2(x: float, y: float, yaw: float = 0.0, robot_id: str = "go2") -> RobotSpec:
    return RobotSpec(id=robot_id, kind="quadruped", mujoco="unitree_go1", x=x, y=y, yaw=yaw, sim_dt=0.005,
                     body_radius=0.3, turn_diameter=0.6, max_vx=0.6, max_wz=1.0, acc_vx=1.0, acc_wz=2.0,
                     cmd_gain=1.0, fall_z=0.15, color="#2563eb",
                     comp=Compensation(min_vx=0.3, min_wz=0.6, min_pure_turn=0.8))


def g1(x: float, y: float, yaw: float = 0.0, robot_id: str = "g1") -> RobotSpec:
    # cmd_gain 1.45: a 0.5 command walks ~0.73 m/s with DimOS's drift compensation (measured in the warehouse).
    # Open loop it also under-turns while walking (0.3 vx + 0.2 wz -> 0.10 rad/s) and drifts sideways (0.4 vx ->
    # -0.14 m/s lateral): DimOS's planner then reports "veered off track" and gives up. Hence track_velocity.
    # nav_speed 0.4: DimOS's planner at 0.22 m/s, 0.22 rad/s for the G1. At full planner speed (0.55) it walks
    # ~0.7 m/s, cannot stop where the planner stops it, and fell in 9% of jittered scenario runs (27 of 300).
    return RobotSpec(id=robot_id, kind="humanoid", mujoco="unitree_g1", x=x, y=y, yaw=yaw, sim_dt=0.002,
                     body_radius=0.35, turn_diameter=0.8, max_vx=0.5, max_wz=0.5, acc_vx=0.8 / 1.45, acc_wz=1.5,
                     cmd_gain=1.45, fall_z=0.45, color="#c2410c", comp=Compensation(track_velocity=True),
                     nav_speed=0.4)


def warehouse_fleet(compensate: bool = True) -> tuple[RobotSpec, ...]:
    """Go2 in the cross aisle, G1 at the charging spot, both facing east. compensate=False: no dead-band
    compensation (see the module docstring)."""
    qx, qy, qyaw = LAYOUT["quadruped"]["home"]
    hx, hy, hyaw = LAYOUT["humanoid"]["home"]
    specs = (go2(qx, qy, qyaw), g1(hx, hy, hyaw))
    return specs if compensate else tuple(s.without_compensation() for s in specs)


def dimos_mujoco_data() -> Path:
    """DimOS's MuJoCo data directory (robot MJCFs + ONNX policies), fetched by dimos.utils.data.get_data."""
    from dimos.utils.data import get_data

    return Path(str(get_data("mujoco_sim")))


def menagerie_dir() -> Path:
    """MuJoCo Menagerie as DimOS resolves it: mujoco_playground's copy, cloned on first use. Found without importing
    mujoco_playground when it already exists (that import pulls in JAX: ~10 s)."""
    import importlib.util

    found = importlib.util.find_spec("mujoco_playground")
    if found is not None and found.origin is not None:
        path = Path(found.origin).parent / "external_deps" / "mujoco_menagerie"
        if (path / "unitree_g1").is_dir():
            return path
    from mujoco_playground._src import mjx_env

    mjx_env.ensure_menagerie_exists()
    return Path(str(mjx_env.MENAGERIE_PATH))


def policy_path(spec: RobotSpec) -> Path:
    return dimos_mujoco_data() / f"{spec.mujoco}_policy.onnx"


def robot_assets(models: set[str]) -> dict[str, bytes]:
    """MJCF + mesh bytes for the given DimOS robot models, keyed by file name (DimOS's get_assets() convention,
    without the office scene and person meshes)."""
    data = dimos_mujoco_data()
    men = menagerie_dir()
    assets: dict[str, bytes] = {}
    for name in sorted(models):
        assets[f"{name}.xml"] = (data / f"{name}.xml").read_bytes()
        for f in sorted((men / name / "assets").glob("*")):
            if f.is_file():
                assets[f.name] = f.read_bytes()
    return assets
