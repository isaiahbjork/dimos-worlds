"""The shared world: both robots walk on their own physics steps and see each other."""
from __future__ import annotations

import math

import pytest

from dimos_worlds.fleet.world import CTRL_DT, SharedWorld, warehouse_world
from dimos_worlds.warehouse import cell


@pytest.fixture(scope="module")
def world() -> SharedWorld:
    w = warehouse_world()
    yield w
    w.close()


def test_each_robot_keeps_its_policy_timestep(world: SharedWorld) -> None:
    go2, g1 = world.bodies["go2"], world.bodies["g1"]
    assert go2.model.opt.timestep == 0.005 and go2.n_substeps == 4
    assert g1.model.opt.timestep == 0.002 and g1.n_substeps == 10
    assert go2.n_substeps * go2.model.opt.timestep == pytest.approx(CTRL_DT)
    # the robots' own joints only: proxies are mocap bodies
    assert go2.model.nq == 19 and g1.model.nq == 36
    assert go2.model.nmocap > 0 and g1.model.nmocap > 0


def test_both_walk_and_proxies_follow(world: SharedWorld) -> None:
    start = world.poses()
    loc = cell.BY_ID["shelf-c"]
    r_g1 = world.walk_to("g1", *loc.stand_point(*loc.center), 0.5)
    r_go2 = world.walk_route("go2", [(9.0, 11.4, 0.0)], 0.6)
    world.step(int(20.0 / CTRL_DT))
    end = world.poses()
    assert math.dist(start["go2"][:2], end["go2"][:2]) > 3.0
    assert math.dist(start["g1"][:2], end["g1"][:2]) > 1.5
    assert world.bodies["g1"].falls == 0 and world.bodies["go2"].falls == 0
    assert r_go2.done and r_go2.error is None
    assert r_g1.done and r_g1.error is None
    # the Go2's pelvis proxy in the G1's model sits where the Go2 is (one tick late at most)
    g1 = world.bodies["g1"]
    mid, bid = world._proxy_src["g1"]["go2"][0]  # noqa: SLF001
    go2_xy = world.bodies["go2"].data.xpos[bid][:2]
    assert math.dist(g1.data.mocap_pos[mid][:2], go2_xy) < 0.05


def test_velocity_command_is_rate_limited(world: SharedWorld) -> None:
    world.stop("go2")
    world.step(50)
    world.set_velocity("go2", 0.8, 0.0, 0.0)
    world.tick()
    v = float(world.bodies["go2"].cmd.value[0])
    assert 0.0 < v <= world.bodies["go2"].spec.acc_vx * CTRL_DT + 1e-6
    world.stop("go2")


def test_g1_tracks_cmd_vel_as_a_velocity(world: SharedWorld) -> None:
    """Open loop the G1 policy walks ~1.6x the command, under-turns while walking and drifts sideways; tracked,
    a straight 0.3 m/s command walks ~0.3 m/s straight."""
    g1 = world.bodies["g1"]
    assert g1.spec.track_velocity and not world.bodies["go2"].spec.track_velocity
    g1.place(6.0, 11.0, 0.0)
    world.set_velocity("g1", 0.3, 0.0, 0.0)
    world.step(int(2.0 / CTRL_DT))
    x0, y0, _ = world.pose("g1")
    world.step(int(5.0 / CTRL_DT))
    x1, y1, yaw = world.pose("g1")
    world.set_velocity("g1", 0.0, 0.0, 0.0)
    assert (x1 - x0) / 5.0 == pytest.approx(0.3, abs=0.06)
    assert abs(y1 - y0) < 0.25 and abs(yaw) < 0.15
    assert g1.falls == 0


def test_g1_holds_its_pose_at_zero_command(world: SharedWorld) -> None:
    g1 = world.bodies["g1"]
    g1.place(6.0, 11.0, 0.0)
    world.set_velocity("g1", 0.3, 0.0, 0.0)
    world.step(int(3.0 / CTRL_DT))
    world.set_velocity("g1", 0.0, 0.0, 0.0)
    world.step(int(2.0 / CTRL_DT))
    x0, y0, _ = world.pose("g1")
    world.step(int(30.0 / CTRL_DT))
    x1, y1, _ = world.pose("g1")
    assert math.hypot(x1 - x0, y1 - y0) < 0.1 and g1.falls == 0


def test_go2_slow_in_place_turn_is_raised_to_one_that_turns(world: SharedWorld) -> None:
    go2 = world.bodies["go2"]
    go2.place(12.0, 11.4, 0.0)
    world.set_velocity("go2", 0.0, 0.0, 0.3)  # below the policy's in-place dead band
    world.step(int(4.0 / CTRL_DT))
    world.set_velocity("go2", 0.0, 0.0, 0.0)
    assert world.pose("go2")[2] > 0.8
    assert [c["args"]["wz"] for c in world.log.commands() if c["robot"] == "go2" and c["op"] == "velocity"][-2] == 0.3  # logged as sent
