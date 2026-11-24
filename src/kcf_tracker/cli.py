"""Interactive single-object tracking from a camera or video."""

import argparse
from time import perf_counter

import cv2

from .tracker import KCFTracker


class Selection:
    """Mouse selection and requests to initialize a new target."""

    def __init__(self):
        self.selecting = False
        self.on_tracking = False
        self.pending = False
        self.start = (0, 0)
        self.cursor = (0, 0)
        self.roi = [0, 0, 0, 0]

    def draw_boundingbox(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.on_tracking = False
            self.pending = False
            self.start = self.cursor = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            width, height = abs(x - self.start[0]), abs(y - self.start[1])
            if width > 10 and height > 10:
                self.roi = [min(x, self.start[0]), min(y, self.start[1]), width, height]
                self.pending = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.on_tracking = False
            width, height = self.roi[2:]
            if width > 0 and height > 0:
                self.roi = [x - width // 2, y - height // 2, width, height]
                self.pending = True


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index or video path (default: 0)")
    parser.add_argument("--use-hog", action="store_true", help="Use HOG rather than grayscale features")
    parser.add_argument("--fixed-window", action="store_true", help="Use a fixed template size")
    parser.add_argument("--multiscale", action="store_true", help="Estimate target scale changes")
    background = parser.add_mutually_exclusive_group()
    background.add_argument(
        "--use-debackground", action="store_true", help="Suppress background before feature extraction"
    )
    background.add_argument(
        "--use-color-model",
        action="store_true",
        help="Weight features and fuse responses with a color-histogram foreground model",
    )
    parser.add_argument(
        "--color-weight-floor",
        type=float,
        default=1.0,
        help="HOG feature weight of pure background with --use-color-model, e.g. 0.6; 1 disables (default: 1)",
    )
    parser.add_argument(
        "--color-merge",
        type=float,
        default=0.3,
        help="Share of the color response in the fused response; 0 disables fusion (default: 0.3)",
    )
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="Skip model updates on low confidence and re-detect a lost target instead of stopping",
    )
    args = parser.parse_args(argv)
    for name in ("color_weight_floor", "color_merge"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def main(args):
    source = int(args.source) if args.source.isdigit() else args.source
    interval = 1 if isinstance(source, int) else 30
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")
        tracker = KCFTracker(
            hog=args.use_hog,
            fixed_window=args.fixed_window,
            multiscale=args.multiscale,
            use_debackground=args.use_debackground,
            use_color_model=args.use_color_model,
            color_weight_floor=args.color_weight_floor,
            color_merge_factor=args.color_merge,
            recovery=args.recovery,
        )
        colors = {"tracking": (0, 255, 255), "uncertain": (0, 165, 255), "lost": (0, 0, 255)}
        selection = Selection()
        duration = 0.01
        first_frame = True
        cv2.namedWindow("tracking")
        cv2.setMouseCallback("tracking", selection.draw_boundingbox)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if first_frame:
                cv2.imshow("tracking", frame)
                key = cv2.waitKey(0) & 0xFF
                if key in (27, ord("q")):
                    break
                first_frame = False
            if selection.selecting:
                cv2.rectangle(frame, selection.start, selection.cursor, (0, 255, 255), 1)
            elif selection.pending:
                tracker.init(selection.roi, frame)
                selection.pending = False
                selection.on_tracking = True
            elif selection.on_tracking:
                start = perf_counter()
                box = tracker.update(frame)
                duration = 0.8 * duration + 0.2 * (perf_counter() - start)
                if box[2] <= 0 or box[3] <= 0:
                    selection.on_tracking = False
                else:
                    x, y, width, height = map(int, box)
                    color = colors.get(tracker.state, colors["tracking"])
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1)
                    if tracker.state == "lost":
                        cv2.putText(frame, "searching", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(frame, f"FPS: {1 / duration:.1f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("tracking", frame)
            if cv2.waitKey(interval) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def cli():
    main(parse_arguments())


if __name__ == "__main__":
    cli()
