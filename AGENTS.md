# AGENTS.md

Guidance for AI coding agents working in this repository. The README and CONTRIBUTING are written in Chinese (Traditional); this file summarizes the parts that matter for code changes.

## Project

Single-object KCF (Kernelized Correlation Filters) tracker in Python, ported from KCFcpp via uoip/KCFpy. Supported Python: `>=3.11,<3.13`. Runtime deps are only NumPy, Numba and `opencv-python`; don't add other OpenCV distributions or unrelated dependencies.

## Commands

Always go through uv with `--locked`:

```sh
uv sync --locked                                   # set up env
uv run --locked pre-commit install                 # once per checkout

# Pre-review checks (non-mutating) — run all before proposing a commit
uv run --locked ruff format --check src tests
uv run --locked ruff check src tests
uv run --locked pre-commit validate-config
uv run --locked pytest

# Single test / file
uv run --locked pytest tests/test_tracking.py::test_recovery_gives_up_after_max_lost_frames
uv run --locked pytest tests/test_tracking.py -k scale_filter

# Run the tracker (needs a display; first frame pauses for mouse ROI selection)
uv run --locked kcf-track --source samples/videos/sintel_20s.mp4 --use-hog --multiscale --use-color-model
uv run --locked kcf-debackground --source 0 --size 40 30

# Offline benchmark (VOT2016 data goes to benchmarks/data/, which is gitignored)
uv run --locked python benchmarks/fetch_vot.py --out benchmarks/data
uv run --locked python benchmarks/evaluate.py --data benchmarks/data/vot --video bag=benchmarks/data/pysot/bag.avi --configs hog_ms_color hog_ms_color_rec --json results.json

# Multi-platform check in headless Linux containers (linux/arm64 and linux/amd64, Python 3.11 and 3.12)
docker/run-tests.sh benchmarks/data
```

- Manage deps with `uv add` / `uv remove`; never hand-edit `uv.lock` or use `--frozen` to hide drift.
- Ruff: py311 target, line length 120, rules `E,F,I,UP,B,N,ASYNC,LOG`, `E501` ignored. Fix diagnostics; don't add broad suppressions or unsafe fixes.
- Never commit with `--no-verify`. If the pre-commit hooks modify files, review the diff and re-stage.
- Changing the supported Python range means updating `requires-python`, `.python-version` (if the default changes), the Ruff target and the lock, then running the tests on every supported version.

## Architecture

Code lives in `src/kcf_tracker/` (src layout). CLI entry points are `kcf-track` → `cli:cli` and `kcf-debackground` → `debackground:cli`.

**`tracker.KCFTracker`** is the core. Public API: `init(roi, image)` and `update(image)`, where `roi` is `[x, y, w, h]`. Every feature is an opt-in constructor flag, and defaults keep the original KCF behavior:
- `hog`: HOG features (`fhog.py`, Numba `@njit(cache=True)` kernels, cell size 4) or raw grayscale (CSK-style, cell size 1). The two modes use different `interp_factor`/`sigma`.
- Scale handling is one of: `multiscale` (3-scale search, step 1.05), `scale_filter` (fDSST-style 1-D scale filter in `scale_filter.py`, which may take a dict of `ScaleFilter` kwargs), or neither. `fixed_window`, `multiscale` and `scale_filter` all use a 96-px template.
- `use_color_model`: `color_model.ColorHistogramModel` (BGR 16 bins/channel, foreground vs. surrounding ring). Its likelihood map is fused into the KCF response (`color_merge_factor`) and can optionally soft-weight the HOG **training** samples only (`color_weight_floor`, 1 = off). Search features are never weighted, and grayscale features are never weighted because both caused drift in testing. Mutually exclusive with `use_debackground`.
- `use_debackground`: the legacy hard grid-based background mask (`debackground.py`, applied via `recttool.subwindow`).
- `recovery`: confidence-gated updates. Peak and APCE are compared to running means. `state` is `"tracking"`, `"uncertain"` (follow the peak, skip the model update) or `"lost"` (hold the box, re-detect over a 3×3 grid of centers plus color-likelihood peaks).

**Correlation math** is done in the Fourier domain: one batched `numpy.fft.rfft2` over all channels. `_alphaf`, `_prob` and `_tmpl_f` are stored as **half spectra** (`rfft2` layout), and `train()` keeps `_tmpl_f`/`_tmpl_sq` in sync with `_tmpl`. If you change the template, the cached spectrum must change with it. `fft.py` holds the older cv2 DFT helpers.

**Lost-target contract:** without `recovery`, `update()` returns `[0, 0, 0, 0]` when the peak is below `pv_thres` and keeps the last model. Callers (the CLI) stop drawing and wait for a new `init()`. With `recovery`, it keeps returning the held box until `max_lost_frames` is exceeded (or forever if `None`).

**`cli.py`** owns the mouse-selection state machine and the frame loop: left-drag to select, right-click to re-center the last box size, `q`/`Esc` to quit. A selection must be larger than 10 px on each side.

## Testing conventions

- Tests use synthetic images and monkeypatched GUI calls. They must never open a camera or need a display.
- When you fix a behavior, add a small test that reproduces the problem.
- Report real-video or interactive results separately from headless test results. Passing synthetic tests says nothing about real-video accuracy or FPS. Use `benchmarks/evaluate.py` for that. Its FPS counts only `init` and `update` time, and it reports OPE and VOT-style reset protocols.
- To compare changes to the tracker hot path, run the benchmark before and after.
- Numba compiles on first use, so the first HOG frame is slow. Don't count it when timing.
