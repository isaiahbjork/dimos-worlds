"""`dimos run dimos-worlds.warehouse-fleet`: Go2 + G1 in one warehouse, each with its own planner, coordinated.

    WarehouseFleetSim ──{id}/odom, {id}/color_image, {id}/joint_state──▶
          │ global_costmap (scene prior + rack footprints + stay-outs)
          ▼
    RobotAwareCostmaps ──go2_costmap / g1_costmap (the OTHER robot painted)──▶ planner per robot
    {id}/goal_request ──▶ FleetTraffic ──{id}/nav_goal──▶ FleetPlanner ──{id}/path, {id}/goal_reached──▶
                          FleetTraffic   (right of way: hold, yield to a pocket, retry; settle at the goal)
    FleetPlanner + MovementManager, namespaced go2/ and g1/ ──{id}/cmd_vel_unsafe──▶ SafetyWatchdog ──{id}/cmd_vel──▶
          WarehouseFleetSim   (FleetTraffic's settling commands go through the watchdog too)

Send a goal: publish a PoseStamped on `go2/goal_request` (or `g1/goal_request`), or click a point routed to
`go2/clicked_point`. FleetTraffic forwards it to that robot's planner (`{id}/nav_goal`) and publishes `{id}/arrived`
when the robot is there. The run log (warehouse-fleet-run.jsonl in the working directory) replays with
`dimos-worlds-replay warehouse-fleet-run.jsonl`.

Every robot's commands pass through its own `SafetyWatchdog` (dimos_worlds.safety): a fall, a lost command link or
stale odometry stops that robot only, latched until `True` on `{id}/safety_reset`. Alerts on `{id}/safety_alert`.
`DIMOS_WORLDS_SAFETY=0` in the environment of `dimos run` builds the blueprints without it (commands go straight to
the sim, as before); in Python, `fleet_blueprint(..., safety=False)`.

`dimos run dimos-worlds.warehouse-fleet-4`: the same with four robots (Go2, G1, a second Go2 and a second G1 on the
dock floor; robots.warehouse_fleet_4), goals on `{go2,g1,go2b,g1b}/goal_request`.
"""
from __future__ import annotations

from typing import Any

from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.navigation.movement_manager.movement_manager import MovementManager
from dimos.visualization.vis_module import vis_module

from dimos_worlds.fleet.module import (
    FleetTraffic,
    FleetTraffic4,
    RobotAwareCostmaps,
    RobotAwareCostmaps4,
    WarehouseFleet4Sim,
    WarehouseFleetSim,
    fleet_agents,
    fleet_specs,
)
from dimos_worlds.fleet.planner import FleetPlanner
from dimos_worlds.fleet.robots import RobotSpec
from dimos_worlds.safety.module import SafetyWatchdog, safety_default


def _nav(spec: RobotSpec, safety: bool) -> Any:
    """Planner + movement manager (+ the safety watchdog) isolated under the robot id, searching that robot's own
    costmap, at that robot's planner speed (FleetPlanner: DimOS's ReplanningAStarPlanner with nerf_speed x
    spec.nav_speed)."""
    rid = spec.id
    parts = [
        FleetPlanner.blueprint(robot_width=2 * spec.body_radius, robot_rotation_diameter=spec.turn_diameter,
                               nav_speed=spec.nav_speed),
        MovementManager.blueprint(),
    ]
    remaps: list[tuple[Any, str, str]] = [
        (FleetPlanner, "global_costmap", f"{rid}_costmap"),
        # goals reach the planner through FleetTraffic only
        (FleetPlanner, "goal_request", "nav_goal"),
        (FleetPlanner, "clicked_point", "nav_clicked_point"),
    ]
    if safety:
        parts.append(SafetyWatchdog.blueprint(robot=rid))
        remaps += [(MovementManager, "cmd_vel", "cmd_vel_unsafe"), (SafetyWatchdog, "cmd_vel_in", "cmd_vel_unsafe")]
    return (
        autoconnect(*parts)
        .remappings(remaps)
        # the costmap topic stays global so it meets RobotAwareCostmaps' output
        .namespace(rid, expose={f"{rid}_costmap"})
    )


def _remaps(ids: tuple[str, ...], sim: Any, costmaps: Any, traffic: Any, safety: bool) -> list[tuple[Any, str, str]]:
    rows: list[tuple[Any, str, str]] = []
    for rid in ids:
        rows += [
            (sim, f"{rid}_odom", f"{rid}/odom"),
            (sim, f"{rid}_color_image", f"{rid}/color_image"),
            (sim, f"{rid}_camera_info", f"{rid}/camera_info"),
            (sim, f"{rid}_joint_state", f"{rid}/joint_state"),
            (sim, f"{rid}_cmd_vel", f"{rid}/cmd_vel"),
            (costmaps, f"{rid}_odom", f"{rid}/odom"),
        ]
        rows += [(traffic, f"{rid}_{port}", f"{rid}/{port}")
                 for port in ("goal_request", "clicked_point", "odom", "path", "goal_reached", "nav_goal", "arrived")]
        # settling commands: through the watchdog like the planner's
        rows.append((traffic, f"{rid}_cmd_vel", f"{rid}/cmd_vel_unsafe" if safety else f"{rid}/cmd_vel"))
    return rows


def _styled_costmap(grid: Any) -> Any:
    return grid.to_rerun(colormap="Accent", z_offset=0.015, opacity=0.25, background="#484981")


def _no_camera_info(_msg: Any) -> Any:
    return None  # the sim publishes no TF tree, so a pinhole would dangle in the 3D view


def _rerun_config(ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "visual_override": {
            "world/global_costmap": _styled_costmap,
            **{f"world/{rid}_costmap": _styled_costmap for rid in ids},
            **{f"world/{rid}/camera_info": _no_camera_info for rid in ids},
        },
        "max_hz": {
            **{f"world/{rid}/color_image": 2.0 for rid in ids},
            "world/global_costmap": 0.5,
        },
    }


def fleet_blueprint(fleet: str, sim: Any, costmaps: Any, traffic: Any, safety: bool | None = None) -> Any:
    """The blueprint for robots.FLEETS[fleet] with its module classes (fleet.module). safety=None: the
    DIMOS_WORLDS_SAFETY default (on)."""
    safety = safety_default() if safety is None else safety
    specs = fleet_specs(fleet)
    ids = tuple(s.id for s in specs)
    assert ids == sim.ROBOT_IDS == costmaps.ROBOT_IDS == traffic.ROBOT_IDS, (ids, sim.ROBOT_IDS)
    return (
        autoconnect(
            sim.blueprint(),
            costmaps.blueprint(),
            traffic.blueprint(agents=fleet_agents(fleet=fleet)),
            *[_nav(spec, safety) for spec in specs],
            vis_module(global_config.viewer, rerun_config=_rerun_config(ids)),
        )
        .remappings(_remaps(ids, sim, costmaps, traffic, safety))
        .global_config(n_workers=8)
    )


warehouse_fleet_blueprint = fleet_blueprint("warehouse-fleet", WarehouseFleetSim, RobotAwareCostmaps, FleetTraffic)
warehouse_fleet_4_blueprint = fleet_blueprint("warehouse-fleet-4", WarehouseFleet4Sim, RobotAwareCostmaps4,
                                              FleetTraffic4)
