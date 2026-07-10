# Changelog

## v0.1.3 — 2026-07-09

Polish from a clean-room stranger-install test (graded A-; all nits one-liners).

- `pyproject.toml` version now tracks the release tag (was stuck at 0.1.0).
- Docs: port-conflict troubleshooting (`BACKLOT_WEB_PORT` / `server.web_port`); zsh-safe
  install quoting (`pip install -e '.[dev]'`); Windows long-path note; explicit note that
  the studio serves without ComfyUI (generation is what needs it).

## v0.1.2 — 2026-07-09

### Added
- **CI (GitHub Actions)**: unit suite on push/PR to main, matrix
  ubuntu-latest + windows-latest × Python 3.11/3.12 (offline suite; ComfyUI/GPU/Blender
  tests self-skip on runners). CI badge in README.

### Fixed
- `faceswap.py` CUDA bootstrap: `os.add_dll_directory` is Windows-only — now also catches
  `AttributeError` so a Linux machine with the nvidia cu12 wheels installed doesn't crash
  at import.

## v0.1.1 — 2026-07-09

Fix round from two independent cold AI reviews of the public repo.

### Security
- **Path traversal**: `web/app.py::_stage_source` now resolves client-supplied
  `filename`/`subfolder` and rejects anything outside the ComfyUI output root
  (`..`, absolute paths, drive letters).
- LAN env overrides renamed `BACKLOT_WEB_HOST`/`BACKLOT_WEB_PORT` (were rename residue);
  binding beyond 127.0.0.1 now carries an explicit no-auth warning in code.

### Fixed
- **A/V drift in crossfade assembly**: `_concat_xfade` places each segment's audio
  ABSOLUTELY at its video start on the final timeline (`adelay` + `amix normalize=0`)
  instead of chaining `acrossfade` (which drifts when a clip's audio and video stream
  lengths differ — every talking clip). `_duration` now measures the VIDEO stream
  (frames/fps), never the container (= longest stream), which also fixes late xfade
  junctions. Verified with a synthetic a/v-mismatch test: junction audio lands at the
  video junction exactly (was +0.3s per junction in the failure case).
- **Cancel semantics**: cancelling a QUEUED job no longer sends a global `/interrupt`
  (which killed whatever job was RUNNING); queued jobs are dequeued via
  `POST /queue {"delete": [id]}`, only the currently-executing job is interrupted.
- **Blender under MCP stdio hosts**: the bpy subprocess no longer inherits the host's
  stdin (`stdin=DEVNULL`) — inheriting the MCP protocol pipe blocked Blender at startup.

### Changed
- Tailwind Play runtime vendored to `web/static/vendor/tailwind.js` — the studio is now
  fully offline/local (no CDN request, no supply-chain surface).
- Dev/live utility scripts moved `tests/` → `scripts/` (pytest suite stays in `tests/`).
- README: explicit **Required ComfyUI custom nodes** section (only the talking-head lanes
  need custom nodes; everything else is core ComfyUI).

### Planned
- CI (GitHub Actions): lint + unit suite on push.

## v0.1.0 — 2026-07-09

Initial public release: engine core (workflow-as-config registry, typed param injection,
ws job manager), MCP server, web studio (gallery/sessions/editing/storyboard/Director),
Blender bridge (headless bpy, GPU depth → Wan VACE), QC suite, multi-shot character
consistency pipeline, 17 generation capabilities. Verified: 105 unit tests + live MCP
handshake + live SDXL render + real headless Blender renders.
