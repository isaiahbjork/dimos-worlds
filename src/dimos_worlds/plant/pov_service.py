"""Plant render service: the Go2's front camera, a payload zoom camera and the plant's fixed CCTV cameras, all from
one warm Blender holding plant.blend (render/pov_render.py) and dimos_worlds.lot.camera_model's ISP.

    python -m dimos_worlds.plant.pov_service --port 8790 [--samples 16]          # serve
    python -m dimos_worlds.plant.pov_service --stills OUT [--events a,b] [--light day,night]   # fixed cameras, exit

    POST /v1/pov     {"camera": "go2", x, y, z, yaw, pitch, light?, events?, lamp?} -> image/jpeg: the Go2's head
                     camera (2.8 mm lens model, barrel distortion) at that pose
    POST /v1/view    {"pos": [x,y,z], "dir": [dx,dy,dz], "hfov_deg": 12, "w": 960, "h": 720, "light"?, "events"?,
                     "lamp"?: 0..10 (a head lamp at the camera)} -> image/jpeg, rectilinear (a payload or zoom camera)
    GET  /v1/health

Plant events (gauge_high:PG-102:8.9, valve_wrong:HV-202, ...) in a body win; otherwise the state file
(plant_state.load_events, DIMOS_WORLDS_PLANT_STATE) sets the needles, valves and leaks. Identical requests are cached.
Camera models are UNVERIFIED for any product. Light: the body's "light", else DIMOS_WORLDS_PLANT_LIGHT (default day).
Blender: DIMOS_WORLDS_BLENDER (default /opt/blender/blender). DIMOS_WORLDS_CYCLES_DEVICE=CUDA skips OptiX.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from dimos_worlds.lot import camera_model as CM
from dimos_worlds.plant import plant_state

HERE = Path(__file__).resolve().parent

BLEND = HERE / "assets" / "cache" / "plant.blend"
JPEG_QUALITY = 85
GO2_LENS_MM = 2.8
CACHE_SIZE = 256
EVENT_KINDS = ("gauge_high", "gauge_low", "motor_hot", "valve_wrong", "leak")


def blender_bin() -> str:
    return os.environ.get("DIMOS_WORLDS_BLENDER") or "/opt/blender/blender"


def plant_events(given=None) -> list[str]:
    """Plant events from the body when they are plant events, else the state file."""
    ev = [str(e) for e in (given or []) if str(e).split(":", 1)[0] in EVENT_KINDS]
    return ev or plant_state.load_events()


def light_for(req: dict) -> str:
    if req.get("light") in ("day", "night"):
        return req["light"]
    return os.environ.get("DIMOS_WORLDS_PLANT_LIGHT", "day")


def view_dir(yaw_deg: float, pitch_deg: float) -> list[float]:
    p, y = math.radians(pitch_deg), math.radians(yaw_deg)
    return [math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)]


class BlenderWorker:
    def __init__(self, samples: int) -> None:
        self.samples = samples
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.tmp = Path(tempfile.mkdtemp(prefix="plant-pov-"))
        self.n = 0
        self.log: list[str] = []

    def _start(self) -> None:
        if not BLEND.exists():
            raise RuntimeError(f"{BLEND} missing: run render/build_scene.py first (see README.md, Demo plant)")
        cmd = [blender_bin(), "-b", str(BLEND), "--disable-autoexec", "-P", str(HERE / "render" / "pov_render.py"),
               "--", "--samples", str(self.samples)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
        self._wait_for("POV_READY", 300)

    def _wait_for(self, prefix: str, timeout: float) -> str:
        assert self.proc and self.proc.stdout
        end = time.monotonic() + timeout
        for line in self.proc.stdout:
            line = line.rstrip()
            self.log = (self.log + [line])[-40:]
            if line.startswith(prefix) or line.startswith("POV_FAIL"):
                return line
            if time.monotonic() > end:
                break
        raise RuntimeError("blender worker stopped: " + " | ".join(self.log[-5:]))

    def render(self, job: dict) -> tuple[np.ndarray, float]:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            assert self.proc and self.proc.stdin
            self.n += 1
            rid = f"r{self.n}"
            out = self.tmp / f"{rid}.npy"
            self.proc.stdin.write(json.dumps({**job, "id": rid, "out": str(out)}) + "\n")
            self.proc.stdin.flush()
            line = self._wait_for(f"POV_DONE {rid}", 180)
            if not line.startswith("POV_DONE"):
                raise RuntimeError(line)
            img = np.load(out).astype(np.float32)
            out.unlink()
            return img, float(line.split()[2])


def auto_exposure(src: np.ndarray, target: float = 0.18, lo: float = 0.4, hi: float = 6.0) -> float:
    """A camera's AE, roughly: scale so the centre-weighted mean luminance lands near mid-grey (after the mode's gain).
    Keeps close-ups of a dial or a dark corner of the control room readable."""
    h, w = src.shape[:2]
    c = src[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    lum = float(np.mean(c @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)))
    return float(np.clip(target / max(lum * 0.55, 1e-4), lo, hi))


class PlantPov:
    def __init__(self, samples: int = 16) -> None:
        self.worker = BlenderWorker(samples)
        self.cache: OrderedDict[tuple, bytes] = OrderedDict()
        self.lock = threading.Lock()
        self.frames = 0
        self.total_ms = 0.0

    def _cached(self, key: tuple, make) -> tuple[bytes, str]:
        with self.lock:
            hit = self.cache.get(key)
            if hit is not None:
                self.cache.move_to_end(key)
                return hit, "hit"
        t0 = time.monotonic()
        jpeg = make()
        with self.lock:
            self.cache[key] = jpeg
            while len(self.cache) > CACHE_SIZE:
                self.cache.popitem(last=False)
            self.frames += 1
            self.total_ms += (time.monotonic() - t0) * 1e3
        return jpeg, "miss"

    def go2(self, req: dict) -> tuple[bytes, str]:
        x, y = float(req["x"]), float(req["y"])
        z = float(req.get("z", 0.4))
        yaw, pitch = float(req.get("yaw", 0.0)) % 360.0, float(req.get("pitch", 0.0))
        light, events, lamp = light_for(req), plant_events(req.get("events")), int(req.get("lamp") or 0)
        key = ("go2", round(x / 0.1), round(y / 0.1), round(z / 0.1), round(yaw / 1.5), round(pitch / 2), light,
               tuple(events), lamp > 0)

        def make() -> bytes:
            hfov = math.degrees(CM.source_fov(GO2_LENS_MM)[0])
            view = {"pos": [x, y, z], "dir": view_dir(yaw, pitch), "hfov_deg": hfov, "w": CM.SRC_W, "h": CM.SRC_H}
            src, _ = self.worker.render({"light": light, "events": events, "view": view, "lamp": lamp})
            seed = ("go2", round(x, 2), round(y, 2), round(yaw, 1), *events)
            return CM.to_jpeg(CM.isp(src, GO2_LENS_MM, light, seed, exposure=auto_exposure(src)), JPEG_QUALITY)

        return self._cached(key, make)

    def view(self, req: dict) -> tuple[bytes, str]:
        pos = [float(v) for v in req["pos"]]
        d = [float(v) for v in req["dir"]]
        hfov = float(req.get("hfov_deg", 12.0))
        w, h = int(req.get("w", 960)), int(req.get("h", 720))
        if not (1.0 <= hfov <= 120.0 and 64 <= w <= 2048 and 64 <= h <= 2048):
            raise ValueError("hfov 1-120 deg, w/h 64-2048")
        light, events, lamp = light_for(req), plant_events(req.get("events")), int(req.get("lamp") or 0)
        key = ("view", *(round(v, 3) for v in pos + d), hfov, w, h, light, tuple(events), lamp)

        def make() -> bytes:
            src, _ = self.worker.render({"light": light, "events": events, "lamp": lamp,
                                         "view": {"pos": pos, "dir": d, "hfov_deg": hfov, "w": w, "h": h}})
            seed = ("view", *(round(v, 2) for v in pos), *events)
            return CM.to_jpeg(CM.isp(src, None, light, seed, exposure=auto_exposure(src)), JPEG_QUALITY)

        return self._cached(key, make)

    def camera(self, cam: str, light: str, events: list[str]) -> bytes:
        lens = float(plant_state.layout()["cameras"][cam]["lens_mm"])

        def make() -> bytes:
            src, _ = self.worker.render({"light": light, "events": events, "view": {"camera": cam}})
            return CM.to_jpeg(CM.isp(src, lens, light, (cam, *events)), JPEG_QUALITY)

        return self._cached(("fixed", cam, light, tuple(events)), make)[0]

    def health(self) -> dict:
        w = self.worker
        return {"ok": w.proc is not None and w.proc.poll() is None, "samples": w.samples, "frames": self.frames,
                "mean_ms": round(self.total_ms / self.frames, 1) if self.frames else None, "cached": len(self.cache),
                "events": plant_state.load_events()}


def make_handler(svc: PlantPov):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:  # noqa: A002
            if os.environ.get("DIMOS_WORLDS_POV_LOG"):
                super().log_message(fmt, *args)

        def _send(self, code: int, data: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code: int, body: dict) -> None:
            self._send(code, json.dumps(body).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/v1/health":
                self._json(200, svc.health())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.rstrip("/")
            try:
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
                if path == "/v1/pov":
                    jpeg, cache = svc.go2(req)
                elif path == "/v1/view":
                    jpeg, cache = svc.view(req)
                else:
                    self._json(404, {"error": "not found"})
                    return
            except (KeyError, TypeError, ValueError) as e:
                self._json(422, {"error": str(e)[:300]})
                return
            except Exception as e:  # noqa: BLE001
                self._json(503, {"error": f"{type(e).__name__}: {e}"[:300]})
                return
            self._send(200, jpeg, "image/jpeg", {"X-Pov-Cache": cache})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--samples", type=int, default=int(os.environ.get("DIMOS_WORLDS_POV_SAMPLES", "16")))
    ap.add_argument("--stills", help="render every fixed camera into this directory and exit")
    ap.add_argument("--events", default="", help="comma-separated plant events for --stills (default: state file)")
    ap.add_argument("--light", default="day,night")
    args = ap.parse_args()
    svc = PlantPov(args.samples)
    if args.stills:
        out = Path(args.stills)
        out.mkdir(parents=True, exist_ok=True)
        events = [e for e in args.events.split(",") if e] or plant_state.load_events()
        for light in args.light.split(","):
            for cam in plant_state.layout()["cameras"]:
                t = time.monotonic()
                (out / f"{cam}-{light}.jpg").write_bytes(svc.camera(cam, light, events))
                print(f"STILL {cam} {light} {time.monotonic() - t:.1f}s", flush=True)
        return
    t = time.monotonic()
    svc.go2({"x": 5.0, "y": 23.6, "yaw": 0})
    print(f"plant_pov: warm in {time.monotonic() - t:.1f}s http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(svc)).serve_forever()


if __name__ == "__main__":
    main()
