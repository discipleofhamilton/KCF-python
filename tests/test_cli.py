from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from kcf_tracker import cli
from kcf_tracker.debackground import parse_arguments as parse_background_arguments


def test_main_tracks_frames_and_releases_capture(monkeypatch):
    image = np.random.default_rng(4).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    capture = Mock(spec=cv2.VideoCapture)
    capture.isOpened.return_value = True
    capture.read.side_effect = [(True, image.copy()) for _ in range(3)] + [(False, None)]
    selection = None
    labels = []

    def set_callback(name, callback):
        nonlocal selection
        selection = callback

    def wait_key(delay):
        if delay == 0:
            selection(cv2.EVENT_LBUTTONDOWN, 30, 30, 0, None)
            selection(cv2.EVENT_LBUTTONUP, 70, 70, 0, None)
        return -1

    monkeypatch.setattr(cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(cv2, "setMouseCallback", set_callback)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "putText", lambda image, text, *args: labels.append(text))
    cli.main(cli.parse_arguments([]))
    capture.release.assert_called_once()
    assert len(labels) == 2
    assert all(label.startswith("FPS:") for label in labels)


def test_main_releases_unavailable_source(monkeypatch):
    capture = Mock(spec=cv2.VideoCapture)
    capture.isOpened.return_value = False
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: capture)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(RuntimeError, match="Cannot open source"):
        cli.main(cli.parse_arguments(["--source", "missing.avi"]))
    capture.release.assert_called_once()


def test_selection_supports_reverse_drag_and_repositioning():
    selection = cli.Selection()
    selection.draw_boundingbox(cv2.EVENT_LBUTTONDOWN, 70, 60, 0, None)
    selection.draw_boundingbox(cv2.EVENT_LBUTTONUP, 20, 10, 0, None)
    assert selection.roi == [20, 10, 50, 50]
    assert selection.pending
    selection.draw_boundingbox(cv2.EVENT_RBUTTONDOWN, 100, 100, 0, None)
    assert selection.roi == [75, 75, 50, 50]
    selection.draw_boundingbox(cv2.EVENT_LBUTTONDOWN, 0, 0, 0, None)
    selection.draw_boundingbox(cv2.EVENT_LBUTTONUP, 2, 2, 0, None)
    assert not selection.pending


def test_background_cli_requires_two_positive_grid_dimensions():
    assert parse_background_arguments(["--size", "8", "6"]).size == [8, 6]
    with pytest.raises(SystemExit):
        parse_background_arguments(["--size", "0", "6"])


def test_tracking_cli_color_model_options():
    args = cli.parse_arguments(["--use-color-model", "--color-merge", "0.5", "--color-weight-floor", "1"])
    assert args.use_color_model and args.color_merge == 0.5 and args.color_weight_floor == 1.0
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--use-color-model", "--use-debackground"])
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--color-merge", "2"])


def test_tracking_cli_recovery_option():
    assert cli.parse_arguments(["--recovery"]).recovery
    assert not cli.parse_arguments([]).recovery
