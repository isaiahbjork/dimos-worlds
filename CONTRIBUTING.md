# Contributing

- Open an issue before large changes.
- Python 3.12, `uv pip install -e ".[dev]"`, `pytest` must pass. On headless Linux set `MUJOCO_GL=egl`.
- Keep runs deterministic: seed everything, no wall-clock in sim state. A change that alters a state hash needs a note in the PR.
- Do not commit heavy binaries (STL, GLB, MP4, USDZ) or fetched assets. Add them to `assets/manifest.json` with license and source URL.
- Do not vendor DimOS or Menagerie files. Import or fetch.
- Commit format: `<area>: <what changed>`.
- Contributions are licensed under Apache-2.0.
- Changes that belong in DimOS itself go to https://github.com/dimensionalOS/dimos (their CLA applies there, not here).
