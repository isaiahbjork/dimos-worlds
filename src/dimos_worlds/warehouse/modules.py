"""`WarehousePriorMap`: the warehouse's known occupancy as `global_costmap`.

The warehouse is static and known, so the planner gets the cooked scene's footprint (walls, rack frames, decks,
conveyor, pallets, bollards), each rack's whole footprint (the frames alone leave gaps between beams), and the arm
cell's stay-out from `places.json`, instead of a lidar-built map. Everything outside the building is lethal.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid

from dimos_worlds.fleet import costmap
from dimos_worlds.fleet.places import load_places, paint_stays


def warehouse_prior() -> tuple[np.ndarray, costmap.Prior]:
    """(sealed grid, prior) for the cooked warehouse scene."""
    import mujoco

    from dimos_worlds.warehouse.scene import PLACES_JSON, SCENE_XML

    from dimos_worlds.warehouse.cell import LAYOUT

    return prior_for(costmap.prior_from_model(mujoco.MjModel.from_xml_path(str(SCENE_XML))),
                     load_places(PLACES_JSON), LAYOUT)


def prior_for(prior: costmap.Prior, places: list[dict], layout: dict) -> tuple[np.ndarray, costmap.Prior]:
    """Scene prior + rack footprints + stays, everything outside the building lethal."""
    grid = paint_stays(prior.grid, prior.origin_x, prior.origin_y, prior.resolution, places)
    h, w = grid.shape
    xs = prior.origin_x + (np.arange(w) + 0.5) * prior.resolution
    ys = prior.origin_y + (np.arange(h) + 0.5) * prior.resolution
    gx, gy = np.meshgrid(xs, ys)
    for shelf in layout["shelves"].values():
        x0, y0, x1, y1 = shelf["rect"]
        grid[(gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)] = 100
    bw, bh = layout["building"]["size"][:2]
    grid[(gx < 0) | (gx > bw) | (gy < 0) | (gy > bh)] = 100
    return costmap.seal_unknown(grid), prior


def grid_msg(grid: np.ndarray, prior: costmap.Prior, ts: float) -> OccupancyGrid:
    origin = Pose()
    origin.position.x = prior.origin_x
    origin.position.y = prior.origin_y
    origin.orientation.w = 1.0
    return OccupancyGrid(grid=grid, resolution=prior.resolution, origin=origin, frame_id="world", ts=ts)


class WarehousePriorMapConfig(ModuleConfig):
    publish_period_s: float = 1.0


class WarehousePriorMap(Module):
    config: WarehousePriorMapConfig
    global_costmap: Out[OccupancyGrid]

    @rpc
    def start(self) -> None:
        super().start()
        self._grid, self._prior = warehouse_prior()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._publish_loop, name="warehouse-prior-map", daemon=True)
        self._thread.start()

    def _publish_loop(self) -> None:
        while not self._stop_evt.is_set():
            self.global_costmap.publish(grid_msg(self._grid, self._prior, time.time()))
            self._stop_evt.wait(self.config.publish_period_s)

    @rpc
    def stop(self) -> None:
        if getattr(self, "_stop_evt", None) is not None:
            self._stop_evt.set()
        super().stop()
