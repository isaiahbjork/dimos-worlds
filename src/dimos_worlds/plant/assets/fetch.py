"""Download the plant scene's third-party assets into assets/cache/ (untracked).

    python3 assets/fetch.py                  # download everything in manifest.json (skips files already present)
    python3 assets/fetch.py --write-manifest # rebuild manifest.json from WANTED (asks Poly Haven for authors/licence)

Only Poly Haven (CC0) textures and HDRIs; everything else in the scene is procedural (this repo). Every file lands under assets/cache/<asset id>/. Nothing under
cache/ is committed (.gitignore). Poly Haven files are checked against the md5 its
API publishes.

Stdlib only, so it runs with any python3 >= 3.10, inside or outside the uv env.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
MANIFEST = HERE / "manifest.json"
UA = {"User-Agent": "dimos-worlds-plant-fetch/1.0 (+https://polyhaven.com/license)"}

# (Poly Haven id, kind, resolution, what it is used for). All CC0 (https://polyhaven.com/license).
WANTED_POLYHAVEN = [
    ("concrete_floor_02", "texture", "2k", "process-area slab"),
    ("gravel_concrete_03", "texture", "1k", "yard outside the slab"),
    ("painted_concrete", "texture", "1k", "control room walls"),
    ("concrete_wall_008", "texture", "1k", "tank bund wall, plinths"),
    ("corrugated_iron", "texture", "1k", "motor shelter roof, control room roof"),
    ("green_metal_rust", "texture", "1k", "tank shells"),
    ("blue_metal_plate", "texture", "1k", "pump and motor casings"),
    ("rusty_metal_02", "texture", "1k", "pipe rack steel"),
    ("metal_plate", "texture", "1k", "pipes, valve bodies"),
    ("metal_grate_rusty", "texture", "1k", "grating on the pipe rack and stair treads"),
    ("painted_metal_shutter", "texture", "1k", "control room door"),
    ("kloofendal_overcast_puresky", "hdri", "1k", "day sky"),
    ("kloppenheim_02_puresky", "hdri", "1k", "night sky"),
]
MENAGERIE: list[dict] = []  # no robot meshes in this scene
MENAGERIE_REPO = "google-deepmind/mujoco_menagerie"
MENAGERIE_REF = "main"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(5):  # flaky links: retry incomplete reads
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (OSError, http.client.HTTPException):
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


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
    for m in MENAGERIE:
        base = f"https://github.com/{MENAGERIE_REPO}"
        entries.append(
            {
                "id": m["id"],
                "name": m["dir"],
                "kind": "robot",
                "source": f"{base}/tree/{MENAGERIE_REF}/{m['dir']}",
                "license": m["license"],
                "license_url": f"{base}/blob/{MENAGERIE_REF}/{m['dir']}/LICENSE",
                "author": m["author"],
                "use": m["use"],
                "menagerie_dir": m["dir"],
            }
        )
    entries.append(
        {
            "id": "procedural",
            "name": "Procedural geometry",
            "kind": "procedural",
            "source": "dimos_worlds/plant/render/",
            "license": "Apache-2.0 (this repository)",
            "author": "this repo",
            "use": "pumps, motors, tanks, pipe rack, pipes, valves, gauges (faces drawn here), control room, stairs, lights, walkway paint, signage",
        }
    )
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


def fetch_menagerie(a: dict) -> None:
    root = CACHE / a["id"]
    if (root / ".done").is_file():
        return
    d = a["menagerie_dir"]
    tree = _json(f"https://api.github.com/repos/{MENAGERIE_REPO}/git/trees/{MENAGERIE_REF}?recursive=1")
    for t in tree["tree"]:
        if t["type"] == "blob" and t["path"].startswith(d + "/"):
            dest = _safe_join(root, t["path"])
            if not dest.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(_get(f"https://raw.githubusercontent.com/{MENAGERIE_REPO}/{MENAGERIE_REF}/{t['path']}"))
    (root / ".done").write_text("ok\n")
    print(f"  fetched {d}")


def fetch_all() -> None:
    assets = json.loads(MANIFEST.read_text())["assets"]
    CACHE.mkdir(exist_ok=True)
    for a in assets:
        print(a["id"])
        if a["source"].startswith("https://polyhaven.com/"):
            fetch_polyhaven(a)
        elif a["kind"] == "robot":
            fetch_menagerie(a)
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
