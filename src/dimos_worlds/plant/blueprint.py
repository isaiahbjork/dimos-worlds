"""`dimos run dimos-worlds.go2-plant`: DimOS's standard Go2 stack, simulated in the Demo plant.

DimOS's `unitree-go2` blueprint with three swaps: the Go2 connection's MuJoCo process loads the plant scene, the lidar
voxel map + cost mapper are replaced by `PlantPriorMap` (the plant's known occupancy as `global_costmap`), and
`PlantPlaces` adds `go_to_place` for the round's stations. The Go2 starts at the dock (5.0, 23.6).

    dimos-worlds-goal /goal_request 22.55 13.0       # in front of pump P-101's discharge gauge

Photoreal frames of the same plant (gauges, valves, tanks) come from the Blender render service
(`python -m dimos_worlds.plant.pov_service`, see README.md, Demo plant); `gauge_loop.py` drives the robot and that service
together for gauge-reading rounds.
"""
from __future__ import annotations

from dimos.core.coordination.blueprints import autoconnect
from dimos.mapping.costmapper import CostMapper
from dimos.mapping.voxels.module import VoxelGridMapper
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.robot.unitree.go2.connection import GO2Connection

from dimos_worlds.plant.go2 import PlantGo2Connection
from dimos_worlds.plant.go2_scene import HOME
from dimos_worlds.plant.modules import PlantPlaces, PlantPriorMap
from dimos_worlds.safety.module import guarded

_x, _y = HOME["pose"]["x"], HOME["pose"]["y"]

go2_plant = guarded(autoconnect(
    unitree_go2.disabled_modules(GO2Connection, VoxelGridMapper, CostMapper),
    PlantGo2Connection.blueprint(),
    PlantPriorMap.blueprint(),
    PlantPlaces.blueprint(),
), robot="go2").global_config(
    simulation="mujoco",
    robot_model="unitree_go2",
    mujoco_start_pos=f"{_x}, {_y}",
    mujoco_camera_position=f"{_x}, {_y}, 0.5, 6, 200, -25",
)
