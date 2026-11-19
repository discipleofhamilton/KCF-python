import cv2
import numpy as np

from .debackground import debackground


def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, limit):
    if rect[0] + rect[2] > limit[0] + limit[2]:
        rect[2] = limit[0] + limit[2] - rect[0]
    if rect[1] + rect[3] > limit[1] + limit[3]:
        rect[3] = limit[1] + limit[3] - rect[1]
    if rect[0] < limit[0]:
        rect[2] -= limit[0] - rect[0]
        rect[0] = limit[0]
    if rect[1] < limit[1]:
        rect[3] -= limit[1] - rect[1]
        rect[1] = limit[1]
    if rect[2] < 0:
        rect[2] = 0
    if rect[3] < 0:
        rect[3] = 0
    return rect


def get_border(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert np.all(np.array(res) >= 0)
    return res


def subwindow(img, window, use_debackground: bool = False, border_type=cv2.BORDER_CONSTANT):
    window = [int(x) for x in window]
    cut_window = window.copy()
    limit(cut_window, [0, 0, img.shape[1], img.shape[0]])  # modify cut_window
    if cut_window[2] <= 0 or cut_window[3] <= 0:
        raise ValueError("Window must overlap the image and have positive dimensions")
    border = get_border(window, cut_window)
    res = img[cut_window[1] : cut_window[1] + cut_window[3], cut_window[0] : cut_window[0] + cut_window[2]]

    if use_debackground:
        res = debackground(output_size=np.array([40, 30]), image=res)

    if border != [0, 0, 0, 0]:
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], border_type)
    return res
