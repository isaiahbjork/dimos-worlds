"""`dimos run dimos-worlds.go2-warehouse`: DimOS's standard Go2 stack, simulated in the warehouse.

DimOS's `unitree-go2` blueprint (A* planner, frontier explorer, patrolling, movement manager, Rerun) with two swaps:
the Go2 connection's MuJoCo process loads the warehouse scene package, and the lidar voxel map + cost mapper are
replaced by `WarehousePriorMap`, the warehouse's known occupancy as `global_costmap` (the building is known; a
lidar-built map from a standing start leaves every goal in unknown space, so A* finds no path).
The Go2 starts in the cross aisle at (3.0, 11.4) facing east. Send it somewhere:

    dimos topic send /goal_request 'PoseStamped(frame_id="world", position=Vector3(12, 11.4, 0), orientation=Quaternion(0, 0, 0, 1))'

"""
from __future__ import annotations

from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.voxels.module import VoxelGridMapper

from dimos_worlds.warehouse.cell import LAYOUT
from dimos_worlds.warehouse.go2 import WarehouseGo2Connection
from dimos_worlds.warehouse.modules import WarehousePriorMap

_x, _y, _yaw = LAYOUT["quadruped"]["home"]

go2_warehouse = autoconnect(
    unitree_go2.disabled_modules(GO2Connection, VoxelGridMapper, CostMapper),
    WarehouseGo2Connection.blueprint(),
    WarehousePriorMap.blueprint(),
).global_config(
    simulation="mujoco",
    robot_model="unitree_go2",
    mujoco_start_pos=f"{_x}, {_y}",
    mujoco_camera_position=f"{_x}, {_y}, 0.5, 6, 200, -25",
)
