"""Gauge-reading rounds in the plant: the gauge-rounds reader and agent driving the sim Go2.

    dimos run dimos-worlds.go2-plant &                                   # the Go2, DimOS nav, the plant's prior map
    python -m dimos_worlds.plant.pov_service --port 8790 &               # Blender frames of the plant (GPU)
    python -m dimos_worlds.plant.gauge_loop --out runs/day1 --events gauge_high:PG-102:8.9,gauge_low:TG-201:8

Needs gauge-rounds (an OpenCV 5 analog gauge reader with an active-perception agent) importable as `gaugerounds`.
For every gauge on the plant's round (layout.json routes.rounds order), gauge-rounds' `agent.inspect()` runs
unchanged with two stand-ins for its simulated world:

* `SimGo2` (the robot): `walk_to` publishes a `goal_request` to DimOS's ReplanningAStarPlanner and waits on `odom`
  until the dog stops, so the pose the policy then sees is where the dog actually stopped. `capture` asks the render
  service (/v1/view, Blender Cycles + camera model) for the camera's frame from that pose. Lamp = a head lamp in the
  render.
* `PlantSite` (the map): the plant's gauges as gauge-rounds Instruments (range, normal band, 225 deg start, 270 deg
  sweep, centre, face normal, dial size) with the sim truth for the current events as their value. The plant's dials
  carry no asset tags: the reader matches a dial to the map by position and size.

Modelling choices (UNVERIFIED on hardware): the camera is gauge-rounds' model (0.32 m up, 90 deg hfov at 1x,
1280x720, rectilinear) on an assumed pan-tilt-zoom head (layout.json robot.payload): it tilts up to 70 deg and zooms
optically up to 4x. --stock-camera uses the Go2's own camera instead (body pitch up to 35 deg, zoom is a crop). Pan and
tilt are render-only: the sim dog's body does not pitch, and the final turn toward the dial is not walked.

Writes to --out: steps.jsonl (every view: commanded and reached pose, reading, issues, decision, truth, error),
poses.jsonl (dog pose at 10 Hz), results.json (per gauge and summary), frames/<n>.jpg (raw) and
frames/<n>_ann.jpg (reader overlay).
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from dimos_worlds.plant import plant_state
from gaugerounds.agent import RulePolicy, inspect
from gaugerounds.spec import GaugeSpec, Registry
from gaugerounds.synth import Camera, Instrument, look_at
from gaugerounds.world import CAM_Z, PITCH_LIMIT, PTZ_TILT, RobotState, digital_zoom, optical

MIN_STANDOFF_M = 0.5   # the gauges sit on pump discharge pipes and tank nozzles: no closer than this to the face
# DimOS's planner calls a goal reached ~0.25 m out (measured on this sim: stops 0.15-0.3 m short), so a 0.2 m
# side-step sent as is never moves the dog. The goal is placed that far past the spot, along the move.
NAV_REACH_M = 0.25
# a standing dog is not a tripod: each capture is taken from a pose a few mm / a tenth of a degree off (seeded),
# which also means a retake is a new frame (the render service caches identical poses)
SWAY_M, SWAY_DEG = 0.004, 0.15


def _yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Go2Link:
    """The go2-plant blueprint's topics: `/goal_request` out (DimOS's planner), `/odom` in."""

    def __init__(self, wait_s: float = 3.0) -> None:
        from dimos.core.transport_factory import make_transport
        from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped

        self._PoseStamped = PoseStamped
        self._goal = make_transport("/goal_request", PoseStamped)
        self._odom = make_transport("/odom", PoseStamped)
        self._last: tuple[float, float, float] | None = None
        self._lock = threading.Lock()
        self._odom.subscribe(self._on_odom)
        self.stop = threading.Event()
        t0 = time.time()
        while self._last is None and time.time() - t0 < 30.0:
            time.sleep(0.1)
        if self._last is None:
            raise RuntimeError("no /odom: is `dimos run dimos-worlds.go2-plant` running?")
        time.sleep(wait_s)  # peer discovery before the first goal (see dimos_worlds.goal)

    def _on_odom(self, msg) -> None:
        with self._lock:
            self._last = (float(msg.position.x), float(msg.position.y), _yaw(msg.orientation))

    def pose(self) -> tuple[float, float, float]:
        with self._lock:
            assert self._last is not None
            return self._last

    def goto(self, leg: str, x: float, y: float, yaw_rad: float) -> str | None:
        self._goal.broadcast(None, self._PoseStamped(ts=time.time(), frame_id="world", position=[x, y, 0.0],
                                                     orientation=[0.0, 0.0, math.sin(yaw_rad / 2),
                                                                  math.cos(yaw_rad / 2)]))
        return None


class PoseLog(threading.Thread):
    def __init__(self, link: Go2Link, path: Path) -> None:
        super().__init__(daemon=True)
        self.link, self.fh, self.stop = link, path.open("a"), threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                x, y, yaw = self.link.pose()
                self.fh.write(json.dumps({"t": time.time(), "x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 4)})
                              + "\n")
                self.fh.flush()
            except Exception:  # noqa: BLE001
                pass
            self.stop.wait(0.1)


class PlantSite:
    """gauge-rounds' Plant interface over the sim plant's gauges."""

    def __init__(self, events: list[str]) -> None:
        truth = plant_state.truth(events)["gauges"]
        self.instruments: list[Instrument] = []
        self.meta: dict[str, dict] = {}
        lay = plant_state.layout()
        order = lay["routes"]["rounds"]
        gauges = [i for i in lay["instruments"] if i["kind"] == "gauge"]
        gauges.sort(key=lambda i: order.index(i["place"]) if i["place"] in order else 99)
        for g in gauges:
            lo, hi = g["range"]
            spec = GaugeSpec(lo, hi, g["unit"], tuple(g["normal"]), plant_state.START_DEG, plant_state.SWEEP_DEG, g["tag"])
            n = (float(g["facing"][0]), float(g["facing"][1]), 0.0)
            self.instruments.append(Instrument(g["tag"], spec, float(truth[g["tag"]]["value"]), tuple(g["pos"]), n,
                                               radius_m=g["dial_m"] / 2, tag_id=None))
            self.meta[g["tag"]] = g

    def registry(self) -> Registry:
        return Registry({})

    def by_id(self, iid: str) -> Instrument:
        return next(i for i in self.instruments if i.id == iid)

    def standpoint(self, inst: Instrument, dist: float, lateral: float = 0.0) -> tuple[float, float]:
        n = np.array(inst.normal[:2], float)
        n /= np.linalg.norm(n)
        side = np.array([-n[1], n[0]])
        p = np.array(inst.center[:2]) + n * max(dist, MIN_STANDOFF_M) - side * lateral
        return float(p[0]), float(p[1])

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return x, y


class SimGo2:
    """gauge-rounds' Robot interface on the sim Go2 + the plant render service."""

    def __init__(self, plant: PlantSite, link: Go2Link, pov_url: str, light: str, events: list[str], out: Path,
                 steps_log, ptz: bool = True) -> None:
        self.plant, self.link, self.light, self.events = plant, link, light, events
        self.pov = httpx.Client(timeout=300.0)
        self.pov_url = pov_url.rstrip("/")
        self.cam = Camera()
        self.backend = self  # gauge-rounds asks its world's backend whether the zoom is optical
        # the plant's payload (layout.json robot.payload): a pan-tilt-zoom head on the dog's back, so zoom is a lens
        # and the head tilts past the body's pitch. --stock-camera = the Go2's own camera (body pitch, crop zoom).
        self.optical_zoom = ptz
        self.tilt_limit = PTZ_TILT if ptz else PITCH_LIMIT
        x, y, yaw = link.pose()
        self.s = RobotState(x, y, math.degrees(yaw))
        self.target: Instrument | None = None
        self.out = out
        self.n = 0
        self.legs = 0
        self.log = steps_log
        self.last_walk: dict = {}
        self.rng = np.random.default_rng(7)

    def walk_to(self, x: float, y: float) -> dict:
        t0 = time.time()
        x0, y0, _ = self.link.pose()
        face = self.target.center if self.target else (x + 1, y, 0)
        yaw = math.atan2(face[1] - y, face[0] - x)
        self.legs += 1
        err = None
        dist = math.hypot(x - x0, y - y0)
        if dist > 0.08:
            over = NAV_REACH_M if dist < 1.0 else 0.0  # long walks: the route's last leg is not the straight line
            gx, gy = x + over * (x - x0) / dist, y + over * (y - y0) / dist
            err = self.link.goto(f"gl-{self.legs}", gx, gy, yaw)
            if err is None:
                err = self._wait(gx, gy)
        px, py, pyaw = self.link.pose()
        walked = self._walked_since(t0) or math.hypot(px - x0, py - y0)
        self.s.x, self.s.y, self.s.yaw_deg = px, py, math.degrees(pyaw)
        self.s.walked_m += walked
        self.last_walk = {"cmd": [round(x, 3), round(y, 3)], "reached": [round(px, 3), round(py, 3)],
                          "off_m": round(math.hypot(px - x, py - y), 3), "walked_m": round(walked, 2),
                          "s": round(time.time() - t0, 1), "error": err}
        self.log({"ev": "walk", "t0": t0, "t1": time.time(), **self.last_walk})
        return {"act": "walk_to", **self.last_walk}

    def _wait(self, gx: float, gy: float) -> str | None:
        """Until the dog has stopped: near the goal (nav's own arrival) or anywhere for 6 s (nav gave up short, e.g.
        a goal inside the plan's inflated obstacles: the round reads from where it stopped)."""
        x0, y0, _ = self.link.pose()
        limit = max(60.0, math.hypot(gx - x0, gy - y0) * 1.5 / 0.5 + 60.0)
        t0, still_since, last = time.time(), time.time(), (x0, y0)
        while time.time() - t0 < limit:
            time.sleep(0.25)
            px, py, _ = self.link.pose()
            if math.hypot(px - last[0], py - last[1]) > 0.02:
                last, still_since = (px, py), time.time()
                continue
            still = time.time() - still_since
            if math.hypot(px - gx, py - gy) < NAV_REACH_M + 0.15 and still > 1.0:
                return None
            if still > 6.0:
                return f"stopped {math.hypot(px - gx, py - gy) - NAV_REACH_M:.2f} m short"
        return "timeout"

    def _walked_since(self, t0: float) -> float:
        p = self.out / "poses.jsonl"
        pts = []
        for line in p.read_text().splitlines()[-3000:]:
            d = json.loads(line)
            if d["t"] >= t0:
                pts.append((d["x"], d["y"]))
        return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])))

    def aim_at(self, inst: Instrument) -> dict:
        p = look_at((self.s.x, self.s.y, CAM_Z), inst.center)
        clipped = abs(p.pitch_deg) > self.tilt_limit
        self.s.yaw_deg = p.yaw_deg
        self.s.pitch_deg = max(-self.tilt_limit, min(self.tilt_limit, p.pitch_deg))
        return {"act": "aim", "at": inst.id, "yaw": round(self.s.yaw_deg, 1), "pitch": round(self.s.pitch_deg, 1),
                "pitch_limited": clipped}

    def set_lamp(self, on: bool) -> dict:
        self.s.lamp = bool(on)
        self.log({"ev": "lamp", "t": time.time(), "on": self.s.lamp})
        return {"act": "lamp", "on": self.s.lamp}

    def set_zoom(self, z: float) -> dict:
        self.s.zoom = float(min(4.0, max(1.0, z)))
        return {"act": "zoom", "z": self.s.zoom}

    def capture(self) -> tuple[np.ndarray, list[dict]]:
        self.s.captures += 1
        jx, jy, jyaw, jp = self.rng.normal(0, [SWAY_M, SWAY_M, SWAY_DEG, SWAY_DEG])
        y, p = math.radians(self.s.yaw_deg + jyaw), math.radians(self.s.pitch_deg + jp)
        d = [math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)]
        t0 = time.time()
        cam = optical(self.cam, self.s.zoom) if self.optical_zoom and self.s.zoom > 1.0 else self.cam
        r = self.pov.post(self.pov_url + "/v1/view", json={
            "pos": [self.s.x + jx, self.s.y + jy, CAM_Z], "dir": d, "hfov_deg": cam.hfov_deg, "w": self.cam.width,
            "h": self.cam.height, "light": self.light, "events": self.events, "lamp": 10 if self.s.lamp else 0})
        r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if self.s.zoom > 1.0 and not self.optical_zoom:
            img, _ = digital_zoom(img, [], self.s.zoom)
        self.n += 1
        raw = self.out / "frames" / f"{self.n:03d}.jpg"
        if self.s.zoom > 1.0 and not self.optical_zoom:  # a crop: save what the reader got, losslessly
            cv2.imwrite(str(raw.with_suffix(".png")), img)
        else:
            raw.write_bytes(r.content)  # the exact bytes the reader decoded
        self.log({"ev": "capture", "t0": t0, "t1": time.time(), "n": self.n})
        return img, []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--events", default="")
    ap.add_argument("--light", default="day")
    ap.add_argument("--pov", default="http://127.0.0.1:8790")
    ap.add_argument("--only", default="", help="comma list of gauge tags (default: the whole round)")
    ap.add_argument("--stock-camera", action="store_true", help="the Go2's own camera instead of the PTZ payload")
    a = ap.parse_args()
    events = [e for e in a.events.split(",") if e]
    out = Path(a.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    plant_state.save_events(events)  # the render service draws the needles from the state file
    plant = PlantSite(events)
    link = Go2Link()
    poselog = PoseLog(link, out / "poses.jsonl")
    poselog.start()
    time.sleep(0.5)
    fh = (out / "steps.jsonl").open("a")

    def log(d: dict) -> None:
        fh.write(json.dumps(d, default=float) + "\n")
        fh.flush()

    robot = SimGo2(plant, link, a.pov, a.light, events, out, log, ptz=not a.stock_camera)
    if a.light == "night":
        robot.set_lamp(False)  # the policy turns it on when it cannot find the dial
    policy = RulePolicy()
    only = set(t for t in a.only.split(",") if t)
    rows = []
    t_start = time.time()
    log({"ev": "start", "t": t_start, "events": events, "light": a.light, "camera": "stock" if a.stock_camera else "ptz"})
    for inst in plant.instruments:
        if only and inst.id not in only:
            continue
        robot.target = inst
        n0 = robot.n
        log({"ev": "gauge_start", "t": time.time(), "tag": inst.id, "truth": inst.value,
             "spec": inst.spec.to_dict(), "label": plant.meta[inst.id]["label"]})
        trace: list[dict] = []
        o = inspect(robot, inst, policy, robot.cam, plant.registry(), trace, keep_frames=True)
        for k, tr in enumerate(trace):
            n = n0 + k + 1
            cv2.imwrite(str(out / "frames" / f"{n:03d}_ann.jpg"), tr["frame"], [cv2.IMWRITE_JPEG_QUALITY, 90])
            st = o.steps[k]
            log({"ev": "view", "tag": inst.id, "n": n, "view": tr["view"], "robot": tr["robot"],
                 "reading": tr["reading"], "decision": tr["decision"], "truth": inst.value,
                 "err_pct": st.truth_err_pct, "dial": tr["dial"]})
        err = o.err_pct(inst.spec)
        true_status = inst.spec.status(inst.value)
        row = {"tag": inst.id, "label": plant.meta[inst.id]["label"], "unit": inst.spec.unit, "truth": inst.value,
               "value": o.value, "err_pct": None if err is None else round(err, 2), "status": o.status,
               "true_status": true_status, "confidence": round(o.confidence, 3), "escalated": o.escalated,
               "confirmed": o.confirmed, "captures": o.captures, "walked_m": round(o.walked_m, 2),
               "silent_wrong": (not o.escalated) and err is not None and err > 2.0,
               "frames": list(range(n0 + 1, robot.n + 1))}
        rows.append(row)
        log({"ev": "gauge_done", "t": time.time(), **row})
        print(json.dumps(row), flush=True)
    poselog.stop.set()
    link.stop.set()
    answered = [r for r in rows if not r["escalated"]]
    faults = [r for r in rows if r["true_status"] != "normal"]
    summary = {
        "gauges": len(rows), "answered": len(answered), "escalated": len(rows) - len(answered),
        "within_2pct_of_answered": sum(1 for r in answered if r["err_pct"] is not None and r["err_pct"] <= 2.0),
        "silent_wrong": sum(1 for r in rows if r["silent_wrong"]),
        "faults": len(faults), "faults_called": sum(1 for r in faults if not r["escalated"] and r["status"] == r["true_status"]),
        "faults_escalated": sum(1 for r in faults if r["escalated"]),
        "false_alarms": sum(1 for r in rows if r["true_status"] == "normal" and not r["escalated"]
                            and r["status"] in ("high", "low")),
        "captures": sum(r["captures"] for r in rows), "minutes": round((time.time() - t_start) / 60, 1),
        "events": events, "light": a.light, "camera": "stock" if a.stock_camera else "ptz",
    }
    (out / "results.json").write_text(json.dumps({"summary": summary, "gauges": rows}, indent=1, default=float))
    log({"ev": "end", "t": time.time(), **summary})
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
