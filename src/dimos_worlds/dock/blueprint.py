"""`dimos run dimos-worlds.drone-dock`: the Dock 3-class station and its M4D-class quadcopter, headless MuJoCo.

    dimos --viewer none run dimos-worlds.drone-dock
    dimos topic send /dock_command 'String("launch 4")'    # covers open, climb to 4 m, covers close behind it
    dimos topic send /dock_command 'String("goto 6 2 5")'
    dimos topic send /dock_command 'String("land")'        # back over the pad, covers open, land, covers close
    dimos topic echo /dock_state String

Velocity flight: send Twist on /cmd_vel (world frame, linear.z climbs, angular.z yaws), at 10 Hz or more while
moving. With the safety watchdog (the default) /cmd_vel goes through it to the drone (`drone_cmd_vel`): a fall
(tilt past 40 degrees), a sender that stops while asking for motion, or stale odometry latches a stop, and the dock
holds the drone in place (its `safety_stopped`) and refuses launch, goto and velocity until /safety_reset; landing
stays allowed. DIMOS_WORLDS_SAFETY=0 connects /cmd_vel straight to the drone.

Cameras: /dock_internal_image, /dock_external_image (fisheye), /drone_image, 1 frame per second each by default.
"""
from __future__ import annotations

from typing import Any

from dimos.core.coordination.blueprints import autoconnect

from dimos_worlds.dock.module import DroneDock
from dimos_worlds.safety.module import SafetyWatchdog, safety_default


def drone_dock_blueprint(safety: bool | None = None, **dock_config: Any) -> Any:
    """The dock module, behind a SafetyWatchdog unless safety is False (None: DIMOS_WORLDS_SAFETY)."""
    dock = DroneDock.blueprint(**dock_config)
    if not (safety_default() if safety is None else safety):
        return autoconnect(dock).remappings([(DroneDock, "drone_cmd_vel", "cmd_vel")])
    # cmd_vel (anyone) -> watchdog -> drone_cmd_vel -> dock; the dock's odom feeds the watchdog; its latch holds
    # the drone. The drone's attitude comes from odometry (no IMU stream).
    return autoconnect(dock, SafetyWatchdog.blueprint(robot="drone")).remappings([
        (SafetyWatchdog, "cmd_vel_in", "cmd_vel"),
        (SafetyWatchdog, "cmd_vel", "drone_cmd_vel"),
    ])


drone_dock = drone_dock_blueprint()
