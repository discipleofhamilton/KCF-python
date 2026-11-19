"""Standalone foreground extraction using Otsu thresholding and contours."""

import argparse

import cv2
import numpy as np


def filter_foreground(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], 0, 255, -1)
    return cv2.bitwise_and(image, image, mask=mask)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", default="input_image.jpg")
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    try:
        cv2.imshow("Filtered Image", filter_foreground(image))
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
