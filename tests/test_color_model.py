import numpy as np
import pytest

from kcf_tracker.color_model import ColorHistogramModel
from kcf_tracker.tracker import KCFTracker

RED, BLUE, GREEN = (0, 0, 220), (220, 0, 0), (0, 200, 0)


def scene(target_xy=(80, 60), size=40, shape=(180, 240)):
    image = np.full((*shape, 3), BLUE, dtype=np.uint8)
    x, y = target_xy
    image[y : y + size, x : x + size] = RED
    return image


def test_likelihood_separates_target_from_surround():
    model = ColorHistogramModel()
    image = scene()
    model.init(image, [80, 60, 40, 40])
    probability = model.likelihood(image)
    assert probability.dtype == np.float32
    assert probability[80, 100] > 0.95  # inside the target
    assert probability[10, 10] < 0.05  # background
    # A color seen in neither region is neutral.
    assert model.likelihood(np.full((4, 4, 3), GREEN, dtype=np.uint8))[0, 0] == pytest.approx(0.5)


def test_update_blends_with_learning_rate():
    model = ColorHistogramModel(learning_rate=0.5)
    model.init(scene(), [80, 60, 40, 40])
    # The target turns green while the background stays blue.
    changed = scene()
    changed[60:100, 80:120] = GREEN
    model.update(changed, [80, 60, 40, 40])
    patch = np.full((1, 2, 3), RED, dtype=np.uint8)
    patch[0, 1] = GREEN
    red, green = model.likelihood(patch)[0]
    assert red == pytest.approx(1.0)  # still unseen in the background
    assert green == pytest.approx(1.0)


def test_spatial_weight_and_response_shapes_and_ranges():
    model = ColorHistogramModel()
    image = scene()
    model.init(image, [80, 60, 40, 40])
    patch = image[40:120, 60:140]  # 80x80 with the target centred
    weight = model.spatial_weight(patch, (18, 18), border=4, floor=0.3)
    assert weight.shape == (18, 18)
    assert weight.min() >= 0.3 - 1e-6 and weight.max() <= 1.0 + 1e-6
    assert weight[9, 9] > 0.9 and weight[0, 0] < 0.4
    response = model.response(patch, (40, 40), (80, 80))
    assert response.shape == (80, 80)
    peak = np.unravel_index(np.argmax(response), response.shape)
    assert abs(peak[0] - 40) <= 1 and abs(peak[1] - 40) <= 1


@pytest.mark.parametrize("kwargs", [{"bins": 12}, {"bins": 0}, {"learning_rate": 0}, {"surround_factor": 1}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ColorHistogramModel(**kwargs)


def test_requires_bgr_uint8_and_initialization():
    model = ColorHistogramModel()
    with pytest.raises(RuntimeError, match="Initialize"):
        model.likelihood(scene())
    with pytest.raises(ValueError, match="BGR"):
        model.init(np.zeros((50, 50), dtype=np.uint8), [10, 10, 10, 10])


@pytest.mark.parametrize("hog,floor", [(False, 1.0), (True, 1.0), (True, 0.3)])
@pytest.mark.parametrize("multiscale", [False, True])
def test_color_model_tracker_follows_target_on_clutter(hog, floor, multiscale):
    rng = np.random.default_rng(7)
    background = rng.integers(0, 120, (180, 240, 3), dtype=np.uint8)
    background[..., 2] //= 4  # keep red out of the background
    target = rng.integers(0, 60, (48, 48, 3), dtype=np.uint8)
    target[..., 2] = rng.integers(180, 255, (48, 48), dtype=np.uint8)
    tracker = KCFTracker(hog=hog, multiscale=multiscale, use_color_model=True, color_weight_floor=floor)
    frame = background.copy()
    frame[60:108, 80:128] = target
    tracker.init([80, 60, 48, 48], frame)
    for shift in (3, 6, 9, 12):
        frame = background.copy()
        frame[60 + shift // 3 : 108 + shift // 3, 80 + shift : 128 + shift] = target
        x, y, width, height = tracker.update(frame)
        assert x == pytest.approx(80 + shift, abs=3)
        assert y == pytest.approx(60 + shift // 3, abs=3)
        assert width == pytest.approx(48, abs=5)


def test_tracker_rejects_conflicting_background_options():
    with pytest.raises(ValueError, match="mutually exclusive"):
        KCFTracker(use_debackground=True, use_color_model=True)
    with pytest.raises(ValueError, match="color_merge_factor"):
        KCFTracker(use_color_model=True, color_merge_factor=1.5)
