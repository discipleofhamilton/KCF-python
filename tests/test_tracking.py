import numpy as np
import pytest

from kcftracker import KCFTracker


@pytest.mark.parametrize("hog", [False, True])
@pytest.mark.parametrize("multiscale", [False, True])
@pytest.mark.parametrize("use_debackground", [False, True])
def test_tracks_translated_target(hog, multiscale, use_debackground):
    rng = np.random.default_rng(42)
    target = rng.integers(40, 240, (48, 48, 3), dtype=np.uint8)
    tracker = KCFTracker(hog=hog, fixed_window=True, multiscale=multiscale, use_debackground=use_debackground)
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    frame[60:108, 80:128] = target
    tracker.init([80, 60, 48, 48], frame)
    for shift in (2, 4, 6):
        frame = np.zeros_like(frame)
        frame[60:108, 80 + shift : 128 + shift] = target
        x, y, width, height = tracker.update(frame)
        assert x == pytest.approx(80 + shift, abs=3)
        assert y == pytest.approx(60, abs=3)
        assert width == pytest.approx(48, abs=5)
        assert height == pytest.approx(48, abs=5)


def test_low_confidence_preserves_model_and_allows_reinitialization(monkeypatch):
    image = np.random.default_rng(1).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    tracker = KCFTracker(multiscale=True)
    tracker.init([20, 20, 40, 40], image)
    template = tracker._tmpl.copy()
    roi = tracker._roi.copy()
    scale = tracker._scale
    responses = iter([([0, 0], 0.01), ([0, 0], 0.02), ([0, 0], 0.03)])
    with monkeypatch.context() as patch:
        patch.setattr(tracker, "detect", lambda *args: next(responses))
        assert tracker.update(image) == [0, 0, 0, 0]
    np.testing.assert_array_equal(tracker._tmpl, template)
    assert tracker._roi == roi
    assert tracker._scale == scale
    assert tracker.update(image) == pytest.approx(roi, abs=1)
    tracker.init([30, 30, 30, 30], image)
    assert tracker.update(image) == pytest.approx([30, 30, 30, 30], abs=1)


@pytest.mark.parametrize("roi", [[0, 0, 0, 10], [0, 0, 10, -1], [0, 0, 10], [np.nan, 0, 10, 10]])
def test_invalid_roi(roi):
    with pytest.raises(ValueError, match="ROI"):
        KCFTracker().init(roi, np.zeros((100, 100, 3), dtype=np.uint8))


def test_update_requires_initialization():
    with pytest.raises(RuntimeError, match="Initialize"):
        KCFTracker().update(np.zeros((100, 100, 3), dtype=np.uint8))
