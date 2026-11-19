"""Color-grid background suppression and an interactive comparison demo."""

import argparse

import cv2
import numba as nb
import numpy as np

from utils import Timer, bitwise, get_euclidean_dist

COLOR_NAMES = (
    np.array(
        [
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
            [128, 0, 0],
            [128, 128, 0],
            [0, 128, 0],
            [0, 128, 128],
            [0, 0, 128],
            [128, 0, 128],
            [0, 0, 0],
            [128, 128, 128],
            [192, 192, 192],
            [255, 255, 255],
        ],
        dtype=np.float32,
    )
    / 255
)


@nb.njit
def find_similar_colorname(cell):
    mean = np.zeros(3, dtype=np.float32)
    for channel in range(3):
        mean[channel] = np.mean(cell[:, :, channel])
    min_dist = np.inf
    index = 0
    for i in range(len(COLOR_NAMES)):
        distance = get_euclidean_dist(COLOR_NAMES[i], mean)
        if distance < min_dist:
            min_dist = distance
            index = i
    return COLOR_NAMES[index]


@nb.njit
def get_grids(output_size, image):
    """Quantize cell means; clamp the grid to the image to avoid empty cells."""
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("Expected a nonempty BGR image")
    if len(output_size) != 2 or output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("Grid size must contain two positive integers")
    height, width, _ = image.shape
    rows = min(int(output_size[0]), height)
    columns = min(int(output_size[1]), width)
    cells = np.empty((rows, columns, 3), dtype=np.float32)
    for row in range(rows):
        for column in range(columns):
            cell = image[
                row * height // rows : (row + 1) * height // rows,
                column * width // columns : (column + 1) * width // columns,
            ]
            cells[row, column] = find_similar_colorname(cell)
    return cells


@nb.njit
def connect_background(colornames):
    """Flood equal-color cells from every boundary using four-way connectivity."""
    height, width, _ = colornames.shape
    background = np.zeros((height, width), dtype=np.uint8)
    stack = np.empty((height * width, 2), dtype=np.int64)
    count = 0
    for row in range(height):
        for column in range(width):
            if row == 0 or row == height - 1 or column == 0 or column == width - 1:
                background[row, column] = 1
                stack[count] = row, column
                count += 1
    while count:
        count -= 1
        row, column = stack[count]
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y, x = row + dy, column + dx
            if (
                0 <= y < height
                and 0 <= x < width
                and not background[y, x]
                and np.array_equal(colornames[y, x], colornames[row, column])
            ):
                background[y, x] = 1
                stack[count] = y, x
                count += 1
    return 1 - background


def show_colornames(cn_unit=100):
    colors = np.repeat(COLOR_NAMES[np.newaxis], cn_unit, axis=0)
    cv2.imshow("color names", np.repeat(colors, cn_unit, axis=1))


def debackground(output_size, image, show_timer=False):
    """Suppress background while preserving the input image's dtype and range."""
    if image.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR image")
    timer = Timer()
    bins = get_grids(np.asarray(output_size), image.astype(np.float32) / 255)
    mask = connect_background(bins).astype(np.uint8)
    # Apply the coarse mask at full resolution to preserve texture for HOG.
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    result = cv2.bitwise_and(image, image, mask=mask)
    if show_timer:
        print(f"debackground: {timer.time() * 1000:.3f}ms")
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--show-colornames", action="store_true")
    parser.add_argument("--size", type=int, nargs=2, default=[40, 30], metavar=("ROWS", "COLUMNS"))
    parser.add_argument("--use-custombitwise", action="store_true")
    args = parser.parse_args(argv)
    if min(args.size) <= 0:
        parser.error("--size values must be positive")
    return args


def main(args):
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")
        if args.show_colornames:
            show_colornames()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.use_custombitwise:
                bins = get_grids(np.asarray(args.size), frame.astype(np.float32) / 255)
                mask = connect_background(bins).astype(np.uint8)
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                result = bitwise(frame, mask)
            else:
                result = debackground(args.size, frame, show_timer=True)
            cv2.imshow("res", np.concatenate((frame, result), axis=1))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27, 32):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def cli():
    main(parse_arguments())


if __name__ == "__main__":
    cli()
