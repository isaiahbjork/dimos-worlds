"""The drone-dock blueprint: the pilot launches and lands through the covers, flies by velocity and goto, holds on the
safety latch; the DimOS module runs it in real time; the blueprint puts the safety watchdog between /cmd_vel and the
drone."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from dimos_worlds.dock.pilot import CMD_TIMEOUT_S, DockPilot, PilotError
from dimos_worlds.dock.sim import CLEAR_ALT_M, DockSim


@pytest.fixture()
def pilot() -> DockPilot:
    return DockPilot(DockSim())


def _flying(p: DockPilot) -> DockPilot:
    p.launch(4.0)
    p.run_until("flying", 30.0)
    return p


def test_launch_opens_climbs_and_closes_behind_the_drone(pilot: DockPilot) -> None:
    pilot.launch(4.0)
    assert pilot.phase == "opening"
    pilot.tick()
    assert pilot.sim.state().state == "opening" and pilot.pos[2] == pytest.approx(pilot.sim.pad_pos[2], abs=0.01)
    took = pilot.run_until("flying", 30.0)
    assert 5.0 < took < 12.0  # 5 s of cover travel, then the climb
    assert pilot.pos[2] - pilot.sim.pad_pos[2] == pytest.approx(4.0, abs=0.1)
    for _ in range(700):  # 7 s: the covers' travel
        pilot.tick()
    assert pilot.sim.state().state == "closed"
    assert pilot.events.count("covers closing behind the drone") == 1


def test_launch_refusals(pilot: DockPilot) -> None:
    with pytest.raises(PilotError):
        pilot.land()  # nothing to land
    with pytest.raises(PilotError):
        pilot.velocity(1.0, 0.0)  # parked
    pilot.stopped(True)
    with pytest.raises(PilotError, match="safety"):
        pilot.launch()
    pilot.stopped(False)
    pilot.launch(1.0)
    assert pilot.launch_alt >= CLEAR_ALT_M + 0.5  # always high enough for the covers to close under it


def test_velocity_is_a_dead_man_and_goto_reaches_the_point(pilot: DockPilot) -> None:
    _flying(pilot)
    x0 = pilot.pos.copy()
    for _ in range(100):  # 1 s at 2 m/s east, commands at 10 Hz
        if _ % 10 == 0:
            pilot.velocity(2.0, 0.0, 0.0)
        pilot.tick()
    assert pilot.pos[0] - x0[0] > 1.2
    for _ in range(int((CMD_TIMEOUT_S + 2.0) / 0.01)):  # no more commands: it stops and hovers
        pilot.tick()
    held = pilot.pos.copy()
    for _ in range(100):
        pilot.tick()
    assert np.linalg.norm(pilot.pos - held) < 0.05
    pilot.goto(3.0, -2.0, 5.0)
    for _ in range(800):
        pilot.tick()
    assert np.linalg.norm(pilot.pos - [3.0, -2.0, 5.0]) < 0.1
    pilot.goto(0.0, 0.0, -10.0)  # below the floor: clamped to 1 m above the pad
    for _ in range(800):
        pilot.tick()
    assert pilot.pos[2] >= pilot.sim.pad_pos[2] + 0.95


def test_safety_latch_holds_and_refuses_but_allows_landing(pilot: DockPilot) -> None:
    _flying(pilot)
    pilot.goto(6.0, 0.0, 5.0)
    for _ in range(50):
        pilot.tick()
    pilot.stopped(True)
    for _ in range(300):
        pilot.tick()
    held = pilot.pos.copy()
    for _ in range(200):
        pilot.tick()
    assert np.linalg.norm(pilot.pos - held) < 0.05 and held[0] < 5.0  # the goto was dropped
    for refused in (lambda: pilot.goto(1, 1, 5), lambda: pilot.velocity(1.0, 0.0)):
        with pytest.raises(PilotError, match="safety"):
            refused()
    pilot.velocity(0.0, 0.0)  # zero (what a latched watchdog sends) is fine
    pilot.land()
    pilot.run_until("parked", 60.0)


def test_land_through_the_covers(pilot: DockPilot) -> None:
    _flying(pilot)
    pilot.goto(4.0, 3.0, 5.0)
    for _ in range(600):
        pilot.tick()
    for _ in range(800):  # covers closed behind it by now
        pilot.tick()
    assert pilot.sim.state().state == "closed"
    pilot.land()
    pilot.run_until("holding", 30.0)
    assert pilot.sim.state().state in ("opening", "open")
    with pytest.raises(PilotError):
        pilot.close()  # the covers are in use
    pilot.run_until("parked", 60.0)
    for _ in range(700):
        pilot.tick()
    s = pilot.sim.state()
    assert s.state == "closed" and s.drone == "parked"
    assert np.linalg.norm(pilot.pos[:2] - pilot.sim.pad_pos[:2]) < 0.02
    assert pilot.sim.max_pen < 0.005 and not pilot.sim.bad_contacts


def test_parse_command() -> None:
    from dimos_worlds.dock.module import parse_command

    assert parse_command("launch 4") == ("launch", [4.0])
    assert parse_command("goto 1, 2, 3") == ("goto", [1.0, 2.0, 3.0])
    assert parse_command("LAND") == ("land", [])
    for bad in ("", "jump", "goto 1 2", "land 3"):
        with pytest.raises(ValueError):
            parse_command(bad)


def test_module_runs_a_launch_and_landing_in_real_time() -> None:
    """The DimOS module on its own thread (no coordinator), 10x real time, cameras off."""
    from dimos.msgs.geometry_msgs.Twist import Twist
    from dimos.msgs.geometry_msgs.Vector3 import Vector3

    from dimos_worlds.dock.module import DroneDock

    m = DroneDock(speed=10.0, feed_fps=0.0)
    status: list[str] = []
    states: list[dict] = []
    odom: list[float] = []
    m.dock_status.subscribe(lambda s: status.append(s.data))
    m.dock_state.subscribe(lambda s: states.append(json.loads(s.data)))
    m.odom.subscribe(lambda p: odom.append(p.position.z))
    try:
        m._start_sim()  # start() also subscribes the inputs, which needs a transport (the coordinator's)

        def wait(phase: str, timeout: float = 10.0) -> None:
            t = time.monotonic()
            while m.state().get("phase") != phase:
                assert time.monotonic() - t < timeout, (phase, m.state(), status[-5:])
                time.sleep(0.02)

        m.command("launch 4")
        wait("flying")
        m._cmds.put(("vel", Twist(linear=Vector3(1.0, 0.0, 0.0), angular=Vector3(0.0, 0.0, 0.0))))
        m._cmds.put(("stop", True))  # the watchdog latched
        time.sleep(0.2)
        m.command("goto 2 2 5")
        time.sleep(0.2)
        assert any("safety stop is latched" in s for s in status)
        m.command("land")
        wait("parked")
        assert "ok launch 4" in status and "ok land" in status
        assert max(odom) > 4.0 and states[-1]["phase"] == "parked"
    finally:
        m.stop()


def _cmd_routes(bp) -> dict:
    return {k: v for k, v in bp.remapping_map.items() if "cmd_vel" in k[1]}


def test_blueprint_routes_cmd_vel_through_the_watchdog() -> None:
    from dimos_worlds.dock.blueprint import drone_dock, drone_dock_blueprint

    for bp in (drone_dock, drone_dock_blueprint(safety=True)):
        dogs = [a for a in bp.blueprints if a.module.__name__ == "SafetyWatchdog"]
        assert len(dogs) == 1 and dogs[0].kwargs["robot"] == "drone"
        r = _cmd_routes(bp)
        assert r[("safetywatchdog", "cmd_vel_in")] == "cmd_vel"  # what senders publish
        assert r[("safetywatchdog", "cmd_vel")] == "drone_cmd_vel"  # what the dock listens to
        assert ("dronedock", "drone_cmd_vel") not in r
    bare = drone_dock_blueprint(safety=False)
    assert not [a for a in bare.blueprints if a.module.__name__ == "SafetyWatchdog"]
    assert _cmd_routes(bare)[("dronedock", "drone_cmd_vel")] == "cmd_vel"


def test_dock_and_watchdog_streams_meet() -> None:
    """autoconnect joins by name and type: the dock's odom feeds the watchdog, the watchdog's latch reaches the dock."""
    import typing

    from dimos_worlds.dock.module import DroneDock
    from dimos_worlds.safety.module import SafetyWatchdog

    d, w = typing.get_type_hints(DroneDock), typing.get_type_hints(SafetyWatchdog)
    assert typing.get_args(d["odom"]) == typing.get_args(w["odom"])
    assert typing.get_args(d["safety_stopped"]) == typing.get_args(w["safety_stopped"])
    assert typing.get_args(d["drone_cmd_vel"]) == typing.get_args(w["cmd_vel"])


def test_watchdog_counts_climb_rate_as_motion() -> None:
    from dimos_worlds.safety.watchdog import Watchdog

    w = Watchdog()
    w.odom(0.0)
    w.command(0.0, 0.0, 0.0, 0.0, vz=0.8)
    assert w.vz_out() == 0.8
    w.odom(0.6)
    w.step(0.6)
    assert w.latched and w.reason == "link_lost" and w.vz_out() == 0.0
