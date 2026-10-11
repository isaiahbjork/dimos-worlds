"""The safety watchdog against a walking Go2 in the shared MuJoCo world, stepped on sim time.

The loop does what the DimOS module does, on sim time: odometry and IMU attitude (the body's orientation, as the
Go2's lowstate imu_state.rpy reports it) every 40 ms, a sender commanding 0.5 m/s forward every 0.1 s through the
watchdog, the watchdog's checks every tick, and zero velocity to the robot on every tick while it is latched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import pytest

from dimos_worlds.fleet.robots import go2
from dimos_worlds.fleet.world import CTRL_DT, warehouse_world
from dimos_worlds.safety.watchdog import Alert, Watchdog, roll_pitch

FWD = (0.5, 0.0, 0.0)


@dataclass
class Rig:
    guarded: bool = True
    world: object = None
    dog: Watchdog = field(default_factory=Watchdog)
    sending: bool = True  # the command link
    sensing: bool = True  # odometry and IMU
    push: tuple[float, float] | None = None  # (from, to) sim seconds: a roll torque on the trunk
    alerts: list[Alert] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.world = warehouse_world((go2(3.0, 11.4, 0.0),))
        self.body = self.world.bodies["go2"]
        self.dog.on_alert.append(self.alerts.append)

    @property
    def t(self) -> float:
        return self.world.sim_time

    def attitude(self) -> tuple[float, float]:
        w, x, y, z = (float(v) for v in self.body.data.qpos[3:7])
        return roll_pitch(x, y, z, w)

    def xy(self) -> tuple[float, float]:
        return float(self.body.data.qpos[0]), float(self.body.data.qpos[1])

    def run(self, seconds: float) -> None:
        for _ in range(round(seconds / CTRL_DT)):
            t = self.t
            self.body.data.xfrc_applied[1] = [0, 0, 0, 40.0, 0, 0] if self.push and self.push[0] <= t < self.push[1] \
                else 0.0
            n = self.world.ticks
            if self.sensing and n % 2 == 0:
                self.dog.odom(t)
                self.dog.imu(*self.attitude(), t)
            if self.sending and n % 5 == 0:
                out = self.dog.command(*FWD, t) if self.guarded else FWD
                self.world.set_velocity("go2", *out)
            if self.guarded:
                self.dog.step(t)
                if self.dog.latched:
                    self.world.set_velocity("go2", 0.0, 0.0, 0.0)
            self.world.tick()

    def close(self) -> None:
        self.world.close()


@pytest.fixture
def rig():
    r = Rig()
    yield r
    r.close()


def test_tipped_go2_is_stopped_latched_and_walks_again_only_after_a_reset(rig: Rig) -> None:
    rig.run(3.0)
    x0, _ = rig.xy()
    assert x0 > 3.8 and not rig.dog.latched  # walking, never tripped by its own gait
    rig.push = (3.0, 3.3)
    rig.run(0.4)
    (fall,) = rig.alerts
    assert fall.kind == "fall" and 3.0 < fall.t <= 3.3  # caught while still tipping, before it hits the floor
    assert rig.body.cmd.goal.tolist() == [0.0, 0.0, 0.0]
    ok, why = rig.dog.reset(rig.t)  # lying on its side
    assert not ok and "tilted" in why
    rig.run(4.0)  # the sim stands a fallen robot back up after 1 s; the sender keeps asking for 0.5 m/s
    assert rig.dog.latched and max(abs(a) for a in rig.attitude()) < math.radians(15)
    x1, y1 = rig.xy()
    rig.run(2.0)
    assert math.dist((x1, y1), rig.xy()) < 0.1  # latched: up again, but it does not walk
    ok, why = rig.dog.reset(rig.t)
    assert ok, why
    rig.run(3.0)
    assert math.dist((x1, y1), rig.xy()) > 0.8  # walks on the sender's next commands
    assert [a.kind for a in rig.alerts] == ["fall", "reset_refused", "reset"]


def test_cut_command_link_stops_the_walking_go2(rig: Rig) -> None:
    rig.run(3.0)
    rig.sending = False  # the link drops; the robot still holds the last command (0.5 m/s forward)
    rig.run(2.5)
    (lost,) = rig.alerts
    assert lost.kind == "link_lost" and 0.5 < lost.data["silent_s"] <= 0.54  # timeout + one tick
    x1, y1 = rig.xy()
    rig.run(2.0)
    assert math.dist((x1, y1), rig.xy()) < 0.1  # stopped
    rig.sending = True  # link back: still latched, nothing moves until a reset
    rig.run(1.0)
    assert math.dist((x1, y1), rig.xy()) < 0.1 and rig.dog.latched


def test_without_the_watchdog_a_cut_link_keeps_the_go2_walking() -> None:
    rig = Rig(guarded=False)
    try:
        rig.run(3.0)
        rig.sending = False
        rig.run(2.5)
        x1, y1 = rig.xy()
        rig.run(2.0)
        assert math.dist((x1, y1), rig.xy()) > 0.6  # the control: the last command keeps it going
    finally:
        rig.close()


def test_stale_odometry_stops_the_walking_go2(rig: Rig) -> None:
    rig.run(3.0)
    rig.sensing = False
    rig.run(1.0)
    (stale,) = rig.alerts
    assert stale.kind == "odom_stale" and 0.5 < stale.data["silent_s"] <= 0.54
    x1, y1 = rig.xy()
    rig.run(2.0)
    assert math.dist((x1, y1), rig.xy()) < 0.1
