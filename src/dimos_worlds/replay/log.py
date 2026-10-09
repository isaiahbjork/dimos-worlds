"""Run logs: every command a world received, with the tick it landed on, plus a state hash every few seconds.

One JSON object per line:

    {"t": "header", "format": 1, "world": {...}, "versions": {...}}     first line: how to rebuild the world
    {"t": "cmd", "tick": 812, "robot": "g1", "op": "walk_route", "args": {...}}
    {"t": "state", "tick": 1000, "hash": "9f2c..."}                      every STATE_EVERY_TICKS
    {"t": "event", "tick": 1403, "robot": "g1", "what": "fell", ...}     informational (falls, recoveries)

A command logged at tick N takes effect before tick N + 1 is stepped, in log order. The world steps on sim time only,
so re-issuing the commands at the same ticks reproduces the run bit for bit (dimos_worlds.replay.replay).
Floats are written with Python's repr, which round-trips exactly.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, TextIO

FORMAT = 1


class RunLog:
    """Collects log lines in memory and, given a path, appends them to a JSONL file as they happen."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.lines: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._fh: TextIO | None = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w")

    def write(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self.lines.append(entry)
            if self._fh is not None:
                self._fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    # ---- views
    def header(self) -> dict[str, Any] | None:
        return next((e for e in self.lines if e.get("t") == "header"), None)

    def commands(self) -> list[dict[str, Any]]:
        return [e for e in self.lines if e.get("t") == "cmd"]

    def states(self) -> dict[int, str]:
        return {int(e["tick"]): str(e["hash"]) for e in self.lines if e.get("t") == "state"}

    @classmethod
    def read(cls, path: str | Path) -> RunLog:
        log = cls()
        for raw in Path(path).read_text().splitlines():
            raw = raw.strip()
            if raw:
                log.lines.append(json.loads(raw))
        if log.header() is None:
            raise ValueError(f"{path}: no header line; not a dimos-worlds run log")
        fmt = log.header().get("format")  # type: ignore[union-attr]
        if fmt != FORMAT:
            raise ValueError(f"{path}: run log format {fmt}, this version reads {FORMAT}")
        return log
