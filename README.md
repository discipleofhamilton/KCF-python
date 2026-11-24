# KCF tracker in Python 3.11 / 3.12

English | [繁體中文](README_CH.md)

A Python implementation of KCF (Kernelized Correlation Filters) for single-object tracking.
Open a camera or video, select the target with the mouse, and the tracker updates the box every frame while showing FPS.
It supports grayscale or HOG features, multi-scale estimation, a color-histogram foreground model,
confidence-gated updates with re-detection, and an experimental background-removal step.

This project is derived from [uoip/KCFpy](https://github.com/uoip/KCFpy), which is based on
[KCFcpp](https://github.com/joaofaro/KCFcpp). The algorithm follows
[High-Speed Tracking with Kernelized Correlation Filters](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf).
See [LICENSE](LICENSE) for licensing.

## Scope and strengths

The project is built on classical computer vision, specifically **correlation filters**. The tracker learns online from the box
selected in the first frame, then computes correlations in the frequency domain with FFTs. It needs no pre-training,
dataset or model weights.

- **Lightweight and portable:** the only runtime dependencies are NumPy, Numba and OpenCV, all of which ship wheels for
  macOS, Linux (x86_64/aarch64) and Windows x86_64. No GPU is needed, and there are no model files to download or
  licenses to clear.
- **Real-time on CPU:** the recommended configuration runs at about 400 FPS for one target on an Apple M2 Pro, leaving
  headroom for decoding, display or several trackers.
- **Explainable and tunable:** every stage (HOG, color probability, correlation response, APCE confidence) is an
  inspectable intermediate result. `tracker.state` distinguishes tracking, low confidence and lost-and-searching, so
  calling code can make its own decisions.
- **Measurable:** an offline VOT2016 benchmark (including the bag video from the SiamRPN demo) lets each change be
  compared on the same accuracy and speed metrics.

### Future direction

Deep-learning Siamese trackers (such as NanoTrack and VitTrack, built into OpenCV) are more accurate on the same benchmark,
but they need model weights, depend on a changing OpenCV DNN API, and some weights have no stated license.
This version stays centered on correlation filters. The Siamese results are kept in the
[benchmark notes](benchmarks/README.md) as a baseline for a next-generation design (for example, deep features or a
Siamese network for re-detection).

## Setup

The toolchain (uv, Ruff, pre-commit and pytest) follows the
[development tools handbook](https://ionetworks.slite.page/p/p5chrq_Z79).
Supported Python is `>=3.11,<3.13`. The default request is 3.11, and tests run on 3.11 and 3.12.
Python 3.13+ is not yet supported; widening the range requires updating the lock and testing every supported version.

Install uv following the [official guide](https://docs.astral.sh/uv/getting-started/installation/), then run from the
project root:

```sh
uv sync --locked
uv run --locked python --version
uv run --locked kcf-track --help
uv run --locked pre-commit install
```

The only runtime dependencies are NumPy, Numba and OpenCV. Use `opencv-python`, which has GUI window support, and do not
install other OpenCV distributions in the same environment. Linux may also need `libgl1`, `libglib2.0-0` and a working
display. Camera access needs OS permission. The first use of HOG or background removal may pause briefly while Numba
compiles.

`pyproject.toml` declares the dependencies, and `uv.lock` is used to rebuild the environment.

## Usage

```sh
# Default camera
uv run --locked kcf-track

# A specific camera, with HOG and multi-scale
uv run --locked kcf-track --source 2 --use-hog --multiscale

# Video with background removal
uv run --locked kcf-track --source path/to/video.mp4 --use-hog --use-debackground

# Color-histogram foreground model (recommended over --use-debackground)
uv run --locked kcf-track --source path/to/video.mp4 --use-hog --multiscale --use-color-model

# Recommended: HOG + multi-scale + color model + confidence gating and re-detection
uv run --locked kcf-track --source path/to/video.mp4 --use-hog --multiscale --use-color-model --recovery

# Also available through the package entry point
uv run --locked python -m kcf_tracker --source path/to/video.mp4
```

Controls: the first frame is paused. Left-drag to select the target, then press any key to start playback; `q` or `Esc`
quits. During playback you can left-drag a new selection, or right-click to move the last selection's size to a new
center. The selection must be wider and taller than 10 pixels. When confidence is too low the tracker stops drawing the
box and you need to select the target again. With `--recovery` it keeps searching instead: a yellow box means tracking,
orange means low confidence (the model is not updated), and a red box labeled "searching" means the target was lost and
is being re-detected.

| Option | Description |
| --- | --- |
| `--source` | Camera index or video path; default `0` |
| `--use-hog` | Use HOG features; grayscale by default |
| `--fixed-window` | Fix the template's longest side to about 96 pixels, still aligned to the feature grid |
| `--multiscale` | Compare the current size with the two neighboring scales; implies a fixed template size |
| `--scale-filter` | Experimental: estimate scale with an fDSST-style scale filter; mutually exclusive with `--multiscale` |
| `--recovery` | Judge confidence from APCE and peak value, skip model updates when it is low, and re-detect over a wider area instead of stopping when the target is lost |
| `--use-debackground` | Apply a background mask before feature extraction; off by default |
| `--use-color-model` | Enable the color-histogram foreground model; mutually exclusive with `--use-debackground` |
| `--color-merge` | Share of the color response in the fused response; `0` disables fusion; default `0.3` |
| `--color-weight-floor` | Soft weight of pure background in HOG features (e.g. `0.6`); `1` disables; default `1` |

The color model (similar to DAT/Staple) learns a foreground histogram from the target box and a background histogram from
a ring around it (BGR, 16 bins per channel). This gives a per-pixel `P(foreground | color)`, updated every frame with a
learning rate of 0.04. The probability map is used in two ways:

- **Response fusion:** the map is box-averaged over the target size to form a color response, which is blended linearly
  with the KCF response according to `--color-merge`.
- **Soft weighting** (optional): applied only to HOG training samples, not to the search region, which avoids the false
  edges a hard mask creates.

In a synthetic deformation test (real video background, target rotating and stretching), HOG with fusion reached a mean
IoU of 0.63, plain HOG 0.06 and `--use-debackground` 0.20. Soft weighting helped slightly on its own but hurt when combined
with fusion, so it is off by default. Grayscale features are never soft-weighted because it caused noticeable drift in
testing.

### Performance and accuracy

Correlations are batched in the frequency domain (one FFT over all HOG channels, with the template spectrum cached). The
tracking results are unchanged and the speed is about 1.7× the previous version.
On 28 VOT2016 sequences (including the bag video from the SiamRPN demo), the recommended configuration improved mean IoU
from 0.339 to 0.389, reduced reset failures from 54 to 49, and increased speed from 236 to 401 FPS (Apple M2 Pro).
See the [benchmark notes](benchmarks/README.md) for the evaluation method, the comparison of optimization approaches, and
platform verification.

Automatic object classification, image-folder sequences and multi-object tracking are not supported.

Background removal and foreground extraction can also be run on their own:

```sh
uv run --locked kcf-debackground --source 0 --size 40 30
uv run --locked kcf-debackground --use-custombitwise
uv run --locked python -m kcf_tracker.foreground_filter ./input_image.jpg
```

Background removal splits the image into a rows × columns grid, quantizes each cell's mean color, and grows the background
from the border through 4-connected cells of the same color. The grid never exceeds the image size. The mask is upscaled
and applied to the original image, preserving foreground texture, the `uint8` type and the 0–255 range.
The algorithm assumes the border is background, so a target touching the border or similar in color to the background may
be removed by mistake.

## Code structure

The project uses the standard src layout. Code lives in `src/kcf_tracker/`, and setuptools installs the package and CLI
entry points.

| File | Responsibility |
| --- | --- |
| `src/kcf_tracker/cli.py` | CLI, mouse-selection state and frame loop |
| `src/kcf_tracker/tracker.py` | `KCFTracker.init(roi, image)`, `update(image)` and model updates |
| `src/kcf_tracker/fhog.py` | Numba-accelerated HOG features |
| `src/kcf_tracker/fft.py` | Batched `rfft2`/`irfft2` and legacy OpenCV DFT helpers |
| `src/kcf_tracker/scale_filter.py` | fDSST-style scale filter (`--scale-filter`) |
| `src/kcf_tracker/recttool.py` | Region cropping, border padding and background-mask integration |
| `src/kcf_tracker/color_model.py` | Color-histogram foreground model, soft weights and color response |
| `src/kcf_tracker/debackground.py` | Color grid, background connectivity and comparison demo |
| `src/kcf_tracker/foreground_filter.py` | Standalone Otsu / largest-contour foreground extraction |
| `src/kcf_tracker/utils.py` | Mask operations, distance and timing helpers |
| `tests/` | Synthetic-image tracking, feature and CLI regression tests |
| `benchmarks/` | VOT2016 download, OPE/reset evaluation and results |
| `docker/` | Multi-platform (linux/amd64, linux/arm64) test image and scripts |

`roi` is `[x, y, width, height]`.
When confidence is too low, `update()` returns `[0, 0, 0, 0]` and keeps the last model; the caller should stop tracking
and wait for a new `init()`.
With `recovery=True` it instead keeps returning the last position and re-detects until `max_lost_frames` is exceeded
(never gives up by default); see `tracker.state` for the current state.
Import it in Python with `from kcf_tracker.tracker import KCFTracker`.
Code that imported the old root-level modules must switch to the `kcf_tracker` package path; the CLI names are unchanged.

## Development and verification

```sh
uv sync --locked
uv run --locked ruff check --fix src tests
uv run --locked ruff format src tests
git diff

# Pre-commit quality checks; these do not modify source files
uv run --locked ruff format --check src tests
uv run --locked ruff check src tests
uv run --locked pre-commit validate-config
uv run --locked pytest
```

Ruff targets Python 3.11 syntax, a 120-character line length and the `E,F,I,UP,B,N,ASYNC,LOG` rules, ignoring `E501`.
Checks cover `src tests`, i.e. the package modules and the tests.
pre-commit runs the locked Ruff as a system hook. Each checkout must install the hook, and uv must be on the PATH used by
Git and your IDE. If a hook modifies files, review the diff, then re-stage and commit; do not bypass checks with
`--no-verify`.
Tests use synthetic images and stubbed GUI calls, so they never open a camera or need a display.

Add or remove dependencies with `uv add` / `uv remove`, and review both `pyproject.toml` and `uv.lock` afterwards.
Do not edit the lock by hand or use `--frozen` to hide a mismatch between the declarations and the lock.
See [CONTRIBUTING.md](CONTRIBUTING.md) for file responsibilities and the review process.

### Multi-platform testing with Docker

`docker/Dockerfile` builds a headless Linux image that installs dependencies from `uv.lock` and runs tests and benchmarks
on amd64 and arm64. Windows and macOS can run the Linux containers through Docker Desktop; non-native architectures run
under emulation and are slower.

```sh
docker build -f docker/Dockerfile -t kcf-python .
docker run --rm kcf-python                              # runs pytest
docker build -f docker/Dockerfile --platform linux/amd64 --build-arg PYTHON_VERSION=3.12 -t kcf-python:amd64-py312 .

# Test linux/arm64 and linux/amd64 × Python 3.11 and 3.12 in turn; optionally pass the fetch_vot.py data directory for a short benchmark
docker/run-tests.sh benchmarks/data
PLATFORMS=linux/amd64 PYTHONS=3.12 docker/run-tests.sh
```

The containers have no display, so run the interactive `kcf-track` on your own machine.

## Limitations

KCF is suited to short-term tracking. `--recovery` adds confidence gating and local re-detection, but its search range is
limited, so a target that leaves the frame or stays occluded for a long time may not be recovered. Fast-moving targets
whose appearance keeps changing can still drift while confidence is high.
Passing synthetic-image tests does not mean real-video accuracy or FPS meets any particular standard; background removal
also needs to be evaluated per scene.
