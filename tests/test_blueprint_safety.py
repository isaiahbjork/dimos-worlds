"""Every shipped blueprint puts a SafetyWatchdog between each robot's command sources and the robot."""
from __future__ import annotations

import pytest


def _cmd_routes(bp) -> dict[tuple[str, str], str]:  # noqa: ANN001
    return {k: v for k, v in bp.remapping_map.items() if "cmd_vel" in k[1]}


@pytest.mark.parametrize("fleet", ["warehouse-fleet", "warehouse-fleet-4"])
def test_fleet_blueprints_route_every_robot_through_its_watchdog(fleet: str) -> None:
    from dimos_worlds.fleet import blueprint as b
    from dimos_worlds.fleet.module import fleet_specs

    bp = b.warehouse_fleet_blueprint if fleet == "warehouse-fleet" else b.warehouse_fleet_4_blueprint
    routes = _cmd_routes(bp)
    sim = next(a.module.__name__.lower() for a in bp.blueprints if a.module.__name__.endswith("Sim"))
    traffic = next(a.module.__name__.lower() for a in bp.blueprints if a.module.__name__.startswith("FleetTraffic"))
    for s in fleet_specs(fleet):
        rid = s.id
        dogs = [a for a in bp.blueprints if a.module.__name__ == "SafetyWatchdog" and a.kwargs["robot"] == rid]
        assert len(dogs) == 1
        assert routes[(f"{rid}/movementmanager", "cmd_vel")] == f"{rid}/cmd_vel_unsafe"
        assert routes[(traffic, f"{rid}_cmd_vel")] == f"{rid}/cmd_vel_unsafe"  # settling goes through it too
        assert routes[(f"{rid}/safetywatchdog", "cmd_vel_in")] == f"{rid}/cmd_vel_unsafe"
        assert routes[(f"{rid}/safetywatchdog", "cmd_vel")] == routes[(sim, f"{rid}_cmd_vel")] == f"{rid}/cmd_vel"


def test_fleet_blueprint_without_the_watchdog() -> None:
    from dimos_worlds.fleet.blueprint import fleet_blueprint
    from dimos_worlds.fleet.module import FleetTraffic, RobotAwareCostmaps, WarehouseFleetSim

    bp = fleet_blueprint("warehouse-fleet", WarehouseFleetSim, RobotAwareCostmaps, FleetTraffic, safety=False)
    assert not [a for a in bp.blueprints if a.module.__name__ == "SafetyWatchdog"]
    assert _cmd_routes(bp)[("go2/movementmanager", "cmd_vel")] == "go2/cmd_vel"


def test_safety_default_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from dimos_worlds.safety.module import safety_default

    assert safety_default()
    for off in ("0", "false", "off", "no"):
        monkeypatch.setenv("DIMOS_WORLDS_SAFETY", off)
        assert not safety_default()


@pytest.mark.parametrize("name", ["go2_warehouse", "go2_lot_night"])
def test_go2_blueprints_route_commands_through_the_watchdog(name: str) -> None:
    if name == "go2_warehouse":
        from dimos_worlds.warehouse.blueprint import go2_warehouse as bp
    else:
        from dimos_worlds.lot.blueprint import go2_lot_night as bp
    routes = _cmd_routes(bp)
    assert [a for a in bp.blueprints if a.module.__name__ == "SafetyWatchdog"]
    assert routes[("movementmanager", "cmd_vel")] == routes[("safetywatchdog", "cmd_vel_in")] == "cmd_vel_unsafe"


def test_watchdog_thread_does_not_shadow_the_modules_event_loop() -> None:
    """DimOS's Module keeps its asyncio loop in `_loop`; a method of that name broke the watchdog under dimos run."""
    from dimos_worlds.safety.module import SafetyWatchdog

    assert "_loop" not in SafetyWatchdog.__dict__
