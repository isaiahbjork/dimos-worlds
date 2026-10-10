"""DimOS's ReplanningAStarPlanner with a per-robot speed scale, for blueprints with one planner per robot.

DimOS reads the planner's speed scale from `GlobalConfig.nerf_speed`, one value for the whole run. The fleet needs
it per robot (the G1's planner at 0.4, see `RobotSpec.nav_speed`). Passing a modified copy of the global config at
import time would freeze it before `dimos run` applies the command line, so the CLI overrides would be lost. Here
the scale is module config (`nav_speed`), and the planner's config is derived when the module is built in its
worker, from the run's config (`self.config.g`, CLI overrides applied) with `nerf_speed` multiplied by `nav_speed`.
A run-wide `--nerf-speed` still slows every robot.
"""
from __future__ import annotations

from typing import Any

from dimos.utils.logging_config import setup_logger

try:
    from dimos.navigation.replanning_a_star.module import ReplanningAStarPlanner, ReplanningAStarPlannerConfig
except ModuleNotFoundError:  # DimOS main (after 0.0.14) moved it under navigation/go2/
    from dimos.navigation.go2.replanning_a_star.module import ReplanningAStarPlanner, ReplanningAStarPlannerConfig

logger = setup_logger()


def scaled_nerf_speed(run_nerf_speed: float, nav_speed: float) -> float:
    """The planner speed scale for one robot: the run's (DimOS applies it only below 1.0) times the robot's."""
    return min(run_nerf_speed, 1.0) * nav_speed


class FleetPlannerConfig(ReplanningAStarPlannerConfig):
    nav_speed: float = 1.0  # this robot's planner speed scale (RobotSpec.nav_speed)


class FleetPlanner(ReplanningAStarPlanner):
    """ReplanningAStarPlanner whose local planner runs at `nav_speed` x the run's speed."""

    config: FleetPlannerConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.config.nav_speed != 1.0:
            # The parent built its GlobalPlanner (no threads until start()) from the run's config plus robot size.
            # Rebuild it with this robot's speed; the LocalPlanner reads nerf_speed only in its constructor.
            base = self._planner._global_config
            speed = scaled_nerf_speed(self.config.g.nerf_speed, self.config.nav_speed)
            self._planner = type(self._planner)(base.model_copy(update={"nerf_speed": speed}))
            logger.info(f"{self.config.frame_id_prefix or 'planner'}: planner speed x{speed:g}")
