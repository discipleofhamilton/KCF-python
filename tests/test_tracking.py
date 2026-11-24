import numpy as np
import pytest

from kcf_tracker.tracker import KCFTracker


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


def _reference_correlation(tracker, x1, x2):
    """The original per-channel cv2.dft gaussian correlation, kept as an oracle."""
    import cv2

    from kcf_tracker import fft

    rows, cols, channels = tracker.size_patch
    c = np.zeros((rows, cols), np.float64)
    for i in range(channels):
        a = cv2.dft(np.float64(x1[i].reshape(rows, cols)), flags=cv2.DFT_COMPLEX_OUTPUT)
        b = cv2.dft(np.float64(x2[i].reshape(rows, cols)), flags=cv2.DFT_COMPLEX_OUTPUT)
        c += cv2.dft(cv2.mulSpectrums(a, b, 0, conjB=True), flags=cv2.DFT_INVERSE | cv2.DFT_SCALE)[:, :, 0]
    c = fft.rearrange(c)
    d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (rows * cols * channels)
    return np.exp(-np.maximum(d, 0) / (tracker.sigma * tracker.sigma))


def test_fourier_domain_correlation_matches_per_channel_reference():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (160, 160, 3), dtype=np.uint8)
    tracker = KCFTracker(hog=True)
    tracker.init([50, 50, 48, 40], image)
    x = tracker.get_features(np.roll(image, 3, axis=1), 0)
    np.testing.assert_allclose(
        tracker.gaussian_correlation(x, tracker._tmpl), _reference_correlation(tracker, x, tracker._tmpl), atol=1e-5
    )
    # The cached template spectrum must equal a fresh transform after a training update.
    tracker.train(x, 0.3)
    np.testing.assert_allclose(tracker._tmpl_f, tracker._spectrum(tracker._tmpl), rtol=1e-4, atol=1e-3)
