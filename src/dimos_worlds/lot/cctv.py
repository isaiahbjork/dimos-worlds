"""Fixed CCTV cameras for the lot (optional sensor).

`CctvRenderer` renders one of the six pole cameras (cam1..cam6) from the standalone lot model with MuJoCo,
then runs `camera_model.isp`: barrel distortion, vignetting, bloom, seeded Poisson-Gaussian sensor noise,
tone curve, sharpening. Output is 1280x720 RGB uint8, like a 2 MP camera's sub-stream snapshot.

Same scene state + same seed -> same bytes. The seed is (camera, seed_parts...), so two captures with
different seed parts differ the way two real captures do.

`LotCctv` is the DimOS module: it publishes `cctv_image` (Image, frame_id = camera name) round-robin over
its cameras at `fps`, and follows `lot_event` so events show up in the frames.

MuJoCo frames are boxes and capsules with flat lighting: good for pipelines, timing and change detection
tests, not for judging what a real night camera would see. render/ has the Blender path for that.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.std_msgs.String import String
from dimos.utils.logging_config import setup_logger

from dimos_worlds.lot import camera_model as CM
from dimos_worlds.lot import events, layout
from dimos_worlds.lot.places import load_model

MUJOCO_EXPOSURE = 1.4  # MuJoCo's lighting is not physical; scale its linearised output to roughly match


class CctvRenderer:
    """Renders lot camera frames. Owns a standalone lot model unless one is passed in."""

    def __init__(self, model: Any = None, data: Any = None) -> None:
        import mujoco

        self.model = model if model is not None else load_model()
        self.data = data if data is not None else mujoco.MjData(self.model)
        if data is None:
            events.reset(self.model, self.data)
        self._renderer = mujoco.Renderer(self.model, CM.SRC_H, CM.SRC_W)

    def source(self, camera: str) -> np.ndarray:
        """Linear float32 pinhole source image (SRC_H x SRC_W x 3) from a lot camera."""
        self._renderer.update_scene(self.data, camera=camera)
        srgb = np.asarray(self._renderer.render(), dtype=np.float32) / 255.0
        return np.power(srgb, 2.2) * MUJOCO_EXPOSURE

    def frame(self, camera: str, seed_parts: tuple = (), mode: str = "night") -> np.ndarray:
        """1280x720 RGB uint8 as the camera would output it (before JPEG)."""
        if camera not in layout.CAMERAS:
            raise ValueError(f"unknown camera {camera!r}; known: {sorted(layout.CAMERAS)}")
        lens = layout.CAMERAS[camera][2]
        return CM.isp(self.source(camera), lens, mode, (camera, *seed_parts))

    def close(self) -> None:
        self._renderer.close()


def frame_from_source(src_linear: np.ndarray, camera: str, seed_parts: tuple = (), mode: str = "night") -> np.ndarray:
    """Camera model on a linear source render (from MuJoCo, Blender or anything else)."""
    return CM.isp(src_linear, layout.CAMERAS[camera][2], mode, (camera, *seed_parts))


# ---------------------------------------------------------------- DimOS module

logger = setup_logger()

class LotCctvConfig(ModuleConfig):
    cameras: list[str] = sorted(layout.CAMERAS)
    fps: float = 1.0  # total frames per second across all cameras
    mode: str = "night"  # "night" or "day" camera tuning
    seed: int = 0  # mixed into every frame's noise seed with the frame counter

class LotCctv(Module):
    """Six fixed lot cameras as one Image stream (frame_id = camera name)."""

    config: LotCctvConfig
    lot_event: In[String]
    cctv_image: Out[Image]

    _thread: threading.Thread | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._pending: list[str] = []
        self.lot_event.subscribe(self._on_event)
        self._thread = threading.Thread(target=self._render_loop, name="lot-cctv", daemon=True)
        self._thread.start()

    def _on_event(self, msg: Any) -> None:
        with self._lock:
            self._pending.append(str(getattr(msg, "data", msg)))

    def _render_loop(self) -> None:
        renderer = CctvRenderer()  # the GL context lives on this thread
        n = 0
        try:
            while not self._stop_evt.is_set():
                t0 = time.time()
                with self._lock:
                    pending, self._pending = self._pending, []
                for spec in pending:
                    try:
                        events.apply(renderer.model, renderer.data, spec)
                    except ValueError as exc:
                        logger.warning(f"cctv ignored event: {exc}")
                cam = self.config.cameras[n % len(self.config.cameras)]
                rgb = renderer.frame(cam, (self.config.seed, n), self.config.mode)
                self.cctv_image.publish(Image.from_numpy(rgb, format=ImageFormat.RGB, frame_id=cam, ts=time.time()))
                n += 1
                self._stop_evt.wait(max(0.0, 1.0 / self.config.fps - (time.time() - t0)))
        finally:
            renderer.close()

    @rpc
    def stop(self) -> None:
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=5.0)
        super().stop()
