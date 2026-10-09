"""Download the lot's third-party assets into assets/cache/ (untracked).

    python3 assets/fetch.py                  # download everything in manifest.json (skips files already present)
    python3 assets/fetch.py --write-manifest # rebuild manifest.json from WANTED (asks Poly Haven for authors/licence)

Only permissive licences: Poly Haven (CC0) and the Blender Foundation human base meshes (CC0).
Every downloaded file lands under assets/cache/<asset id>/. Nothing under cache/ is committed
(.gitignore). Poly Haven files are checked against the md5 its API publishes.

Stdlib only, so it runs with any python3 >= 3.10, inside or outside the uv env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
MANIFEST = HERE / "manifest.json"
UA = {"User-Agent": "dimos-worlds-asset-fetch/1.0 (+https://polyhaven.com/license)"}

# (Poly Haven id, kind, resolution, what it is used for)
WANTED_POLYHAVEN = [
    ("worn_asphalt", "texture", "2k", "lot surface (worn asphalt with patches)"),
    ("asphalt_02", "texture", "2k", "drive lanes and street outside the fence"),
    ("concrete_floor_worn_001", "texture", "1k", "pads, curbs, wheel stops, sidewalk"),
    ("corrugated_iron_02", "texture", "1k", "service bay walls"),
    ("painted_plaster_wall", "texture", "1k", "office walls"),
    ("kloppenheim_02_puresky", "hdri", "2k", "night sky (stars, faint skyglow); low strength"),
    ("kloofendal_overcast_puresky", "hdri", "2k", "day-ish overcast sky for --light day"),
    ("modular_chainlink_fence", "model", "1k", "chain-link wire material (alpha + normal maps) on fence and gate fabric"),
    ("rollershutter_door", "model", "1k", "service bay roll-up doors"),
    ("metal_trash_can", "model", "1k", "trash can by the office door"),
    ("exterior_aircon_unit", "model", "1k", "rooftop / wall AC units on the office"),
    ("covered_car", "model", "1k", "covered vehicle in the back row"),
    ("utility_box_01", "model", "1k", "electrical cabinet by the service bay"),
    ("old_tyre", "model", "1k", "used tyres stacked by the service bay"),
    ("barrel_03", "model", "1k", "oil drums by the fuel tank"),
    ("fire_hydrant", "model", "1k", "hydrant outside the front fence"),
]

WANTED_ZIP = [
    {
        "id": "human_base_meshes",
        "name": "Human Base Meshes bundle v1.4.1",
        "kind": "bundle",
        "source": "https://www.blender.org/download/demo-files/#assets",
        "download": "https://download.blender.org/demo/asset-bundles/human-base-meshes/human-base-meshes-bundle-v1.4.1.zip",
        "license": "CC0-1.0",
        "author": "Blender Studio (Blender Foundation)",
        "use": "realistic male body mesh for the intruder; rigged and posed in render/build_scene.py",
    },
]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _json(url: str):
    return json.loads(_get(url))


def write_manifest() -> None:
    entries = []
    for pid, kind, res, use in WANTED_POLYHAVEN:
        info = _json(f"https://api.polyhaven.com/info/{pid}")
        entries.append(
            {
                "id": pid,
                "name": info.get("name", pid),
                "kind": kind,
                "source": f"https://polyhaven.com/a/{pid}",
                "license": "CC0-1.0",
                "license_url": "https://polyhaven.com/license",
                "author": ", ".join(info.get("authors", {}).keys()),
                "resolution": res,
                "use": use,
            }
        )
    entries.extend(WANTED_ZIP)
    MANIFEST.write_text(json.dumps({"assets": entries}, indent=2) + "\n")
    print(f"wrote {MANIFEST} ({len(entries)} assets)")


def _save(url: str, dest: Path, md5: str | None = None) -> None:
    if dest.is_file() and (md5 is None or hashlib.md5(dest.read_bytes()).hexdigest() == md5):
        return
    data = _get(url)
    if md5 and hashlib.md5(data).hexdigest() != md5:
        raise RuntimeError(f"md5 mismatch for {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"  {dest.relative_to(CACHE)} ({len(data) // 1024} KiB)")


def _safe_join(root: Path, rel: str) -> Path:
    out = (root / rel).resolve()
    if root.resolve() not in out.parents and out != root.resolve():
        raise RuntimeError(f"refusing path outside cache: {rel}")
    return out


def fetch_polyhaven(a: dict) -> None:
    pid, kind, res = a["id"], a["kind"], a["resolution"]
    root = CACHE / pid
    files = _json(f"https://api.polyhaven.com/files/{pid}")
    if kind == "hdri":
        f = files["hdri"][res]["hdr"]
        _save(f["url"], root / f"{pid}_{res}.hdr", f.get("md5"))
    elif kind == "texture":
        for key, name in (("Diffuse", "diff"), ("nor_gl", "nor_gl"), ("Rough", "rough")):
            f = files[key][res]["jpg"]
            _save(f["url"], root / f"{pid}_{name}_{res}.jpg", f.get("md5"))
    elif kind == "model":
        blend = files["blend"][res]["blend"]
        _save(blend["url"], root / f"{pid}_{res}.blend", blend.get("md5"))
        for rel, f in blend.get("include", {}).items():
            _save(f["url"], _safe_join(root, rel), f.get("md5"))
    else:
        raise ValueError(kind)


def fetch_zip(a: dict) -> None:
    root = CACHE / a["id"]
    marker = root / ".extracted"
    if marker.is_file():
        return
    zpath = root / "download.zip"
    try:
        _save(a["download"], zpath)
    except OSError as e:  # download.blender.org answers 403 to some cloud/datacenter IPs
        raise RuntimeError(f"{a['download']}: {e}. Download it by hand (any browser) to {zpath} and rerun.") from e
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = _safe_join(root, m.filename)
            if m.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(m))
    zpath.unlink()
    marker.write_text("ok\n")
    print(f"  extracted {a['id']}")


def fetch_all() -> None:
    assets = json.loads(MANIFEST.read_text())["assets"]
    CACHE.mkdir(exist_ok=True)
    for a in assets:
        print(a["id"])
        if a["source"].startswith("https://polyhaven.com/"):
            fetch_polyhaven(a)
        else:
            fetch_zip(a)
    print(f"done: {len(assets)} assets in {CACHE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write-manifest", action="store_true")
    args = p.parse_args()
    try:
        write_manifest() if args.write_manifest else fetch_all()
    except Exception as e:  # network or checksum failure: say which, non-zero exit
        print(f"fetch failed: {e}", file=sys.stderr)
        sys.exit(1)
