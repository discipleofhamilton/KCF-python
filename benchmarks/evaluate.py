"""Offline accuracy and speed benchmark on VOT-style sequences.

Each sequence directory holds ``groundtruth.txt`` (VOT polygons or x,y,w,h boxes) and either the
frames ``%08d.jpg`` or a video named by ``--video name=path`` (for example the pysot ``bag.avi``,
which is the VOT2016 ``bag`` sequence at the same size).

Two protocols are reported:

* **OPE** (one-pass): initialise once on frame 1 and never reset. A lost target (the tracker
  returns an empty box) scores IoU 0. Reports mean IoU, success AUC and precision at 20 px.
* **Reset** (VOT-style supervised): when IoU drops to 0, count a failure and re-initialise from
  ground truth 5 frames later. Reports failures and the mean IoU of tracked frames.

FPS counts only ``tracker.update`` and ``tracker.init`` time, not decoding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from kcf_tracker.tracker import KCFTracker

CONFIGS = {
    "gray": dict(hog=False, fixed_window=True, multiscale=False),
    "hog": dict(hog=True, fixed_window=True, multiscale=False),
    "hog_ms": dict(hog=True, fixed_window=True, multiscale=True),
    "hog_ms_color": dict(hog=True, fixed_window=True, multiscale=True, use_color_model=True),
    "hog_sf": dict(hog=True, fixed_window=True, scale_filter=True),
    "hog_sf_color": dict(hog=True, fixed_window=True, scale_filter=True, use_color_model=True),
    "sf_c_n13": dict(hog=True, scale_filter=dict(n_scales=13, n_interp=25), use_color_model=True),
    "sf_c_lr015": dict(hog=True, scale_filter=dict(learning_rate=0.015), use_color_model=True),
    "sf_c_a256": dict(hog=True, scale_filter=dict(max_area=256), use_color_model=True),
    "sf_c_lr04": dict(hog=True, scale_filter=dict(learning_rate=0.04), use_color_model=True),
    "hog_sf2": dict(hog=True, fixed_window=True, scale_filter=True, scale_filter_reuse=False),
    "hog_sf2_color": dict(
        hog=True, fixed_window=True, scale_filter=True, scale_filter_reuse=False, use_color_model=True
    ),
}


def axis_aligned(row):
    """pysot's area-preserving axis-aligned box from a VOT polygon, as x, y, w, h."""
    v = np.asarray(row, dtype=np.float64)
    if v.size == 4:
        return v
    xs, ys = v[0::2], v[1::2]
    cx, cy = xs.mean(), ys.mean()
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    a1 = np.linalg.norm(v[0:2] - v[2:4]) * np.linalg.norm(v[2:4] - v[4:6])
    a2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(a1 / a2) if a2 > 0 else 1.0
    w, h = s * (x2 - x1) + 1, s * (y2 - y1) + 1
    return np.array([cx - w / 2, cy - h / 2, w, h])


def load_groundtruth(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        values = [float(v) for v in line.replace("\t", ",").split(",") if v.strip()]
        rows.append(axis_aligned(values) if values and not np.isnan(values).any() else np.full(4, np.nan))
    return np.array(rows)


def iou(a, b):
    if a[2] <= 0 or a[3] <= 0 or np.isnan(b).any():
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def center_error(a, b):
    if a[2] <= 0 or a[3] <= 0:
        return np.inf
    return float(np.hypot(a[0] + a[2] / 2 - b[0] - b[2] / 2, a[1] + a[3] / 2 - b[1] - b[3] / 2))


def load_frames(seq_dir, video=None):
    if video:
        cap = cv2.VideoCapture(str(video))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return frames
    files = sorted(Path(seq_dir).glob("*.jpg"))
    return [cv2.imread(str(f)) for f in files]


def run_ope(make_tracker, frames, gt):
    tracker = make_tracker()
    elapsed = 0.0
    t = perf_counter()
    tracker.init(list(gt[0]), frames[0])
    elapsed += perf_counter() - t
    ious, errors = [1.0], [0.0]
    lost, processed = False, 1
    for frame, truth in zip(frames[1:], gt[1:], strict=True):
        if lost:
            box = np.zeros(4)
        else:
            t = perf_counter()
            box = np.asarray(tracker.update(frame), dtype=np.float64)
            elapsed += perf_counter() - t
            processed += 1
            # The tracker stops drawing when it loses confidence; the CLI then waits for a new box.
            lost = box[2] <= 0 or box[3] <= 0
        ious.append(iou(box, truth))
        errors.append(center_error(box, truth))
    return np.array(ious), np.array(errors), processed / elapsed, processed


def run_reset(make_tracker, frames, gt, skip=5, burn_in=10):
    """VOT supervised protocol; accuracy excludes ``burn_in`` frames after each (re)init."""
    failures, overlaps = 0, []
    i, n = 0, len(frames)
    while i < n:
        if np.isnan(gt[i]).any():
            i += 1
            continue
        tracker = make_tracker()
        tracker.init(list(gt[i]), frames[i])
        start = i
        i += 1
        while i < n:
            box = np.asarray(tracker.update(frames[i]), dtype=np.float64)
            o = iou(box, gt[i])
            if o <= 0 and not np.isnan(gt[i]).any():
                failures += 1
                i += skip
                break
            if i - start > burn_in:
                overlaps.append(o)
            i += 1
    return failures, float(np.mean(overlaps)) if overlaps else 0.0


def success_auc(ious):
    thresholds = np.linspace(0, 1, 21)
    return float(np.mean([(ious > t).mean() for t in thresholds]))


def evaluate(config, sequences, data_dir, videos, protocols):
    params = CONFIGS[config] if isinstance(config, str) else config
    results = {}
    for name in sequences:
        seq_dir = Path(data_dir) / name
        gt = load_groundtruth(seq_dir / "groundtruth.txt")
        frames = load_frames(seq_dir, videos.get(name))
        n = min(len(frames), len(gt))
        frames, gt = frames[:n], gt[:n]
        row = {"frames": n}
        if "ope" in protocols:
            ious, errors, fps, processed = run_ope(lambda: KCFTracker(**params), frames, gt)
            row.update(
                processed=processed,
                miou=float(ious.mean()),
                auc=success_auc(ious),
                prec20=float((errors <= 20).mean()),
                fps=fps,
            )
        if "reset" in protocols:
            failures, acc = run_reset(lambda: KCFTracker(**params), frames, gt)
            row.update(failures=failures, accuracy=acc)
        results[name] = row
        print(
            f"  {name:12s} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
            file=sys.stderr,
            flush=True,
        )
    return results


def summarize(results):
    rows = list(results.values())
    out = {}
    for key in ("miou", "auc", "prec20", "accuracy"):
        if key in rows[0]:
            out[key] = float(np.mean([r[key] for r in rows]))
    if "failures" in rows[0]:
        out["failures"] = int(sum(r["failures"] for r in rows))
    if "fps" in rows[0]:
        frames = sum(r["processed"] for r in rows)
        out["fps"] = frames / sum(r["processed"] / r["fps"] for r in rows)  # frame-weighted
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Directory with one sub-directory per sequence")
    parser.add_argument("--sequences", nargs="*", help="Sequence names (default: all sub-directories)")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS), help=f"Any of {list(CONFIGS)}")
    parser.add_argument("--video", action="append", default=[], help="name=path to read a sequence from a video")
    parser.add_argument("--protocols", nargs="*", default=["ope", "reset"], choices=["ope", "reset"])
    parser.add_argument("--json", help="Write per-sequence results to this file")
    args = parser.parse_args(argv)
    videos = dict(v.split("=", 1) for v in args.video)
    sequences = args.sequences or sorted(p.name for p in Path(args.data).iterdir() if (p / "groundtruth.txt").exists())

    # Warm up Numba so JIT compilation is not counted as tracking time.
    warm = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    KCFTracker(hog=True).init([40, 40, 40, 40], warm)

    report = {}
    for config in args.configs:
        print(config, file=sys.stderr, flush=True)
        results = evaluate(config, sequences, args.data, videos, args.protocols)
        report[config] = {"summary": summarize(results), "sequences": results}
        print(config, json.dumps({k: round(v, 4) for k, v in report[config]["summary"].items()}), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
