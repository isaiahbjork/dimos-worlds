"""Traffic agents for fleet robots, and which of them settle at their goal after the planner reports arrival, and how (see traffic.SettleConfig).

DimOS's planner stops a robot within 0.2 m and 15-20 degrees of the goal, and accepts anything within 0.5 m when a
replan happens near the goal; the G1 then stood tens of degrees off. A robot whose cmd_vel is tracked as a velocity
(spec.track_velocity: the G1) can be walked the last centimetres and turned the last degrees, holonomically, at
<= 0.15 m/s. A robot without velocity tracking has dead bands below which small forward commands do nothing (the
Go2), so it only corrects its heading, with in-place turns that clear its turn dead band.
"""
from __future__ import annotations

import math

from dimos_worlds.coord.traffic import Agent, SettleConfig
from dimos_worlds.fleet.robots import RobotSpec


def settle_for(spec: RobotSpec) -> SettleConfig | None:
    if spec.comp.track_velocity:
        return SettleConfig(pos_tol=0.12, yaw_tol=math.radians(6.0), translate=True, max_v=0.15, max_wz=0.35)
    if spec.comp.min_pure_turn > 0:
        # heading only; in-place turns below min_pure_turn are raised to it by the world, so stop early enough
        return SettleConfig(yaw_tol=math.radians(8.0), translate=False, k_yaw=1.0, max_wz=0.5, min_wz=0.0)
    return None


def agents_for(specs: tuple[RobotSpec, ...]) -> list[Agent]:
    """Priority by list order (first = highest), radius from the spec, settling as above."""
    n = len(specs)
    return [Agent(id=s.id, radius=s.body_radius, priority=n - i, speed=0.5, settle=settle_for(s))
            for i, s in enumerate(specs)]
