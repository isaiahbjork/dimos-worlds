"""DimOS blueprints for the night vehicle lot.

    dimos run dimos-worlds.go2-lot-night                         # Go2 sim in the lot, prior map, A*, places
    dimos run dimos-worlds.go2-lot-night dimos-worlds.lot-cctv   # plus the six fixed CCTV cameras

`go2-lot-night`:
  DimOS's Go2 connection on the lot simulator (headless MuJoCo, Go1 body + DimOS's ONNX policy), the lot's
  prior occupancy map as `global_costmap`, DimOS's ReplanningAStarPlanner + MovementManager, `LotPlaces`
  (go_to_place) and `LotEvents` (trigger). The robot starts at `home` (-16, -11).
  It does not run DimOS's voxel mapper / cost mapper: the lot is known, and the lidar-built map would fight
  the prior over `global_costmap`.
"""

from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import rerun_config
from dimos.visualization.vis_module import vis_module

from dimos_worlds.lot.connection import LotGo2Connection
from dimos_worlds.lot.modules import LotEvents, LotPlaces, LotPriorMap
from dimos_worlds.lot.places import HOME


def _vis():  # type: ignore[no-untyped-def]
    from dimos.core.global_config import global_config

    return vis_module(viewer_backend=global_config.viewer, rerun_config=rerun_config)


go2_lot_night = autoconnect(
    _vis(),
    LotGo2Connection.blueprint(),
    LotPriorMap.blueprint(),
    ReplanningAStarPlanner.blueprint(),
    MovementManager.blueprint(),
    LotPlaces.blueprint(),
    LotEvents.blueprint(),
).global_config(
    n_workers=8,
    robot_model="unitree_go2",
    simulation="mujoco",  # DimOS's planner uses its sim thresholds (stuck = < 1 m in 8 s)
    mujoco_start_pos=f"{HOME.x}, {HOME.y}",
)
