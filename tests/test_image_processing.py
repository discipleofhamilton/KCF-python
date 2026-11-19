import cv2
import numpy as np
import pytest

from kcf_tracker import fft, fhog
from kcf_tracker.debackground import connect_background, debackground, get_grids
from kcf_tracker.foreground_filter import filter_foreground
from kcf_tracker.recttool import subwindow
from kcf_tracker.utils import bitwise


def test_complex_arithmetic_matches_numpy():
    rng = np.random.default_rng(2)
    left = rng.random((4, 6, 2), dtype=np.float32)
    right = rng.random((4, 6, 2), dtype=np.float32) + 0.1
    a = left[:, :, 0] + 1j * left[:, :, 1]
    b = right[:, :, 0] + 1j * right[:, :, 1]
    for result, expected in [
        (fft.complex_division(left, right), a / b),
        (fft.complex_multiplication(left, right), a * b),
    ]:
        np.testing.assert_allclose(result[:, :, 0] + 1j * result[:, :, 1], expected, atol=1e-6)


def test_hog_histograms_preserve_fractional_and_large_values():
    image = np.random.default_rng(3).integers(0, 255, (48, 48, 3), dtype=np.uint8)
    features = fhog.get_feature_maps(image, 4, {})["map"]
    assert features.dtype == np.float32
    assert features.max() > 255
    assert np.any(features != features.astype(np.uint8))


@pytest.mark.parametrize("shape", [(5, 7), (43, 37), (80, 60)])
def test_grid_handles_small_and_uneven_images(shape):
    image = np.ones((*shape, 3), dtype=np.float32)
    cells = get_grids(np.array([40, 30]), image)
    assert cells.shape == (min(40, shape[0]), min(30, shape[1]), 3)
    np.testing.assert_array_equal(cells, np.ones_like(cells))


@pytest.mark.parametrize("size", [[0, 30], [-1, 30], [40]])
def test_grid_rejects_invalid_sizes(size):
    with pytest.raises(ValueError, match="Grid size"):
        get_grids(np.array(size), np.zeros((10, 10, 3), dtype=np.float32))


def test_background_flood_reaches_cells_from_bottom():
    cells = np.zeros((7, 7, 3), dtype=np.float32)
    cells[1:6, 1:6] = [1, 0, 0]
    cells[2:7, 3] = 0
    mask = connect_background(cells)
    assert mask[2, 3] == 0
    assert mask[3, 2] == 1
    assert not mask[0].any()


def test_background_filter_preserves_foreground_pixels():
    image = np.zeros((80, 60, 3), dtype=np.uint8)
    image[20:60, 20:40] = [73, 190, 210]
    result = debackground([40, 30], image)
    assert result.dtype == image.dtype
    assert result.shape == image.shape
    np.testing.assert_array_equal(result[20:60, 20:40], image[20:60, 20:40])
    np.testing.assert_array_equal(result[:20], 0)


def test_subwindow_replicates_image_border():
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    result = subwindow(image, [-2, -1, 6, 4], border_type=cv2.BORDER_REPLICATE)
    expected = cv2.copyMakeBorder(image[:3, :4], 1, 0, 2, 0, cv2.BORDER_REPLICATE)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(4, 6), (4, 6, 3)])
def test_custom_mask_matches_opencv(shape):
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[1:3, 2:4] = 1
    np.testing.assert_array_equal(bitwise(image, mask), cv2.bitwise_and(image, image, mask=mask))


def test_foreground_filter_handles_no_contours():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    np.testing.assert_array_equal(filter_foreground(image), image)
