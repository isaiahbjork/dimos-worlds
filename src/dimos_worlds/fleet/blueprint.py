"""`dimos run dimos-worlds.warehouse-fleet`: Go2 + G1 in one warehouse, each with its own planner, coordinated.

    WarehouseFleetSim ──{id}/odom, {id}/color_image, {id}/joint_state──▶
          │ global_costmap (scene prior + rack footprints + stay-outs)
          ▼
    RobotAwareCostmaps ──go2_costmap / g1_costmap (the OTHER robot painted)──▶ planner per robot
    {id}/goal_request ──▶ FleetTraffic ──{id}/nav_goal──▶ FleetPlanner ──{id}/path, {id}/goal_reached──▶
                          FleetTraffic   (right of way: hold, yield to a pocket, retry; settle at the goal)
    FleetPlanner + MovementManager, namespaced go2/ and g1/ ──{id}/cmd_vel──▶ WarehouseFleetSim

Send a goal: publish a PoseStamped on `go2/goal_request` (or `g1/goal_request`), or click a point routed to
`go2/clicked_point`. FleetTraffic forwards it to that robot's planner (`{id}/nav_goal`) and publishes `{id}/arrived`
when the robot is there. The run log (warehouse-fleet-run.jsonl in the working directory) replays with
`dimos-worlds-replay warehouse-fleet-run.jsonl`.
"""
from __future__ import annotations

from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.visualization.vis_module import vis_module

from dimos_worlds.fleet.module import ROBOT_IDS, FleetTraffic, RobotAwareCostmaps, WarehouseFleetSim, fleet_agents
from dimos_worlds.fleet.planner import FleetPlanner
from dimos_worlds.fleet.robots import RobotSpec, warehouse_fleet


def _nav(spec: RobotSpec) -> Any:
    """Planner + movement manager isolated under the robot id, searching that robot's own costmap, at that robot's
    planner speed (FleetPlanner: DimOS's ReplanningAStarPlanner with nerf_speed x spec.nav_speed)."""
    rid = spec.id
    return (
        autoconnect(
            FleetPlanner.blueprint(robot_width=2 * spec.body_radius, robot_rotation_diameter=spec.turn_diameter,
                                   nav_speed=spec.nav_speed),
            MovementManager.blueprint(),
        )
        .remappings([
            (FleetPlanner, "global_costmap", f"{rid}_costmap"),
            # goals reach the planner through FleetTraffic only
            (FleetPlanner, "goal_request", "nav_goal"),
            (FleetPlanner, "clicked_point", "nav_clicked_point"),
        ])
        # the costmap topic stays global so it meets RobotAwareCostmaps' output
        .namespace(rid, expose={f"{rid}_costmap"})
    )


def _remaps() -> list[tuple[Any, str, str]]:
    rows: list[tuple[Any, str, str]] = []
    for rid in ROBOT_IDS:
        rows += [
            (WarehouseFleetSim, f"{rid}_odom", f"{rid}/odom"),
            (WarehouseFleetSim, f"{rid}_color_image", f"{rid}/color_image"),
            (WarehouseFleetSim, f"{rid}_camera_info", f"{rid}/camera_info"),
            (WarehouseFleetSim, f"{rid}_joint_state", f"{rid}/joint_state"),
            (WarehouseFleetSim, f"{rid}_cmd_vel", f"{rid}/cmd_vel"),
            (RobotAwareCostmaps, f"{rid}_odom", f"{rid}/odom"),
        ]
        rows += [(FleetTraffic, f"{rid}_{port}", f"{rid}/{port}")
                 for port in ("goal_request", "clicked_point", "odom", "path", "goal_reached", "nav_goal", "cmd_vel",
                              "arrived")]
    return rows


def _styled_costmap(grid: Any) -> Any:
    return grid.to_rerun(colormap="Accent", z_offset=0.015, opacity=0.25, background="#484981")


def _no_camera_info(_msg: Any) -> Any:
    return None  # the sim publishes no TF tree, so a pinhole would dangle in the 3D view


_rerun_config = {
    "visual_override": {
        "world/global_costmap": _styled_costmap,
        **{f"world/{rid}_costmap": _styled_costmap for rid in ROBOT_IDS},
        **{f"world/{rid}/camera_info": _no_camera_info for rid in ROBOT_IDS},
    },
    "max_hz": {
        **{f"world/{rid}/color_image": 2.0 for rid in ROBOT_IDS},
        "world/global_costmap": 0.5,
    },
}

assert tuple(s.id for s in warehouse_fleet()) == ROBOT_IDS

warehouse_fleet_blueprint = (
    autoconnect(
        WarehouseFleetSim.blueprint(),
        RobotAwareCostmaps.blueprint(),
        FleetTraffic.blueprint(agents=fleet_agents()),
        *[_nav(spec) for spec in warehouse_fleet()],
        vis_module(global_config.viewer, rerun_config=_rerun_config),
    )
    .remappings(_remaps())
    .global_config(n_workers=8)
)
