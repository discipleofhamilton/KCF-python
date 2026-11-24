# Headless Linux image for tests and benchmarks on amd64 or arm64.
#   docker build -f docker/Dockerfile -t kcf-python .
#   docker build -f docker/Dockerfile --platform linux/amd64 --build-arg PYTHON_VERSION=3.12 -t kcf-python:amd64-py312 .
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# opencv-python needs libGL and GLib at import time, even without a display.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv

# Use the image's interpreter (not .python-version), so PYTHON_VERSION selects the version tested.
ENV UV_LINK_MODE=copy \
    UV_PYTHON=/usr/local/bin/python \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    NUMBA_CACHE_DIR=/tmp/numba-cache

WORKDIR /app
# Install locked dependencies first so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project
COPY . .
RUN uv sync --locked

# The locked environment is on PATH, so commands run without re-syncing.
CMD ["pytest", "-q"]
