#!/usr/bin/env sh
# Build the image and run Ruff, pytest and a short benchmark for each platform and Python version.
# Usage: docker/run-tests.sh [benchmark-data-dir]   (data from benchmarks/fetch_vot.py; optional)
# Non-native platforms run under emulation (QEMU/Rosetta) and are slower.
set -eu
cd "$(dirname "$0")/.."
DATA="${1:-}"
PLATFORMS="${PLATFORMS:-linux/arm64 linux/amd64}"
PYTHONS="${PYTHONS:-3.11 3.12}"
for platform in $PLATFORMS; do
    for python in $PYTHONS; do
        tag="kcf-python:$(echo "$platform" | tr / -)-py$python"
        echo "== $platform, Python $python"
        docker build -q -f docker/Dockerfile --platform "$platform" --build-arg PYTHON_VERSION="$python" -t "$tag" . >/dev/null
        docker run --rm --platform "$platform" "$tag" sh -c \
            'python -c "import platform, numpy, cv2, numba; print(platform.machine(), platform.python_version(), numpy.__version__, cv2.__version__, numba.__version__)" &&
             ruff check src tests benchmarks && pytest -q'
        if [ -n "$DATA" ]; then
            docker run --rm --platform "$platform" -v "$(cd "$DATA" && pwd)":/data:ro "$tag" \
                python benchmarks/evaluate.py --data /data/vot --video bag=/data/pysot/bag.avi \
                --sequences bag sphere --configs hog_ms_color_rec --protocols ope
        fi
    done
done
