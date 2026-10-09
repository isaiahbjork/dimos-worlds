"""The lot's scene files and named places.

Places are fixed points of interest in the lot frame (metres, x east, y north, origin at the lot centre).
Each one names the fixed CCTV camera that covers it. `pose` is where events put the person; the optional
`robot_pose` is where a robot is sent instead (`Place.goal_xy`). `home` is where a robot starts: a free spot on the
asphalt west of row D. It is a sim choice, not part of the layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

SCENE_DIR = Path(__file__).resolve().parent / "scene"
LOT_XML = SCENE_DIR / "lot.xml"
PLACES_JSON = SCENE_DIR / "places.json"
SCENE_META = SCENE_DIR / "scene.meta.json"


@dataclass(frozen=True)
class Place:
    id: str
    name: str
    kind: str
    x: float
    y: float
    camera: str | None  # MJCF camera name ("cam1".."cam6"); None for home
    # Where a robot should stand to look at the place. Same as (x, y) unless the place itself hugs an
    # obstacle (the front gate's point is 0.35 m from the closed gate leaf).
    robot_x: float | None = None
    robot_y: float | None = None

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def goal_xy(self) -> tuple[float, float]:
        if self.robot_x is None or self.robot_y is None:
            return self.xy
        return (self.robot_x, self.robot_y)


HOME = Place(id="home", name="Home", kind="start", x=-16.0, y=-11.0, camera=None)


def load_places(path: Path = PLACES_JSON, *, with_home: bool = False) -> dict[str, Place]:
    """{place id: Place}, in file order. with_home adds the robot start point."""
    places: dict[str, Place] = {}
    if with_home:
        places[HOME.id] = HOME
    for r in json.loads(path.read_text()):
        places[r["id"]] = Place(
            id=r["id"],
            name=r["name"],
            kind=r["kind"],
            x=float(r["pose"]["x"]),
            y=float(r["pose"]["y"]),
            camera=r.get("camera"),
            robot_x=float(r["robot_pose"]["x"]) if "robot_pose" in r else None,
            robot_y=float(r["robot_pose"]["y"]) if "robot_pose" in r else None,
        )
    return places


def load_model(path: Path = LOT_XML):  # type: ignore[no-untyped-def]
    """The standalone lot as a MuJoCo model (no robot)."""
    import mujoco

    return mujoco.MjModel.from_xml_path(str(path))
