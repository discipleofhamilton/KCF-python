"""Color-histogram foreground model in the style of DAT and Staple.

A foreground histogram is learned from the target box and a background histogram from the ring
around it. Every pixel then gets ``P(foreground | color)``. The likelihood map is used two ways:

* as a soft spatial weight on the KCF features, in place of the hard debackground mask, so the
  features have no artificial edges;
* as a color response that is fused with the KCF correlation response, as in Staple.
"""

import cv2
import numpy as np


class ColorHistogramModel:
    def __init__(
        self,
        bins: int = 16,
        learning_rate: float = 0.04,
        surround_factor: float = 2.0,
        foreground_shrink: float = 0.8,
    ):
        if bins <= 0 or 256 % bins or bins & (bins - 1):
            raise ValueError("bins must be a power of two that divides 256")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if surround_factor <= 1.0:
            raise ValueError("surround_factor must be greater than 1")
        if not 0.0 < foreground_shrink <= 1.0:
            raise ValueError("foreground_shrink must be in (0, 1]")
        self.bins = bins
        self.learning_rate = learning_rate
        self.surround_factor = surround_factor
        self.foreground_shrink = foreground_shrink
        self._shift = 8 - (bins.bit_length() - 1)
        self._foreground = None  # normalized histogram, shape (bins**3,)
        self._background = None
        self._lut = None  # P(foreground | color), shape (bins, bins, bins)

    @property
    def initialized(self) -> bool:
        return self._lut is not None

    def _quantize(self, image):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected a uint8 BGR image")
        return image >> self._shift

    def _flat_index(self, quantized):
        # bins**3 <= 4096 for the default 16 bins, so uint16 halves the memory traffic of int32.
        dtype = np.uint16 if self.bins**3 <= 1 << 16 else np.int32
        q = quantized.astype(dtype)
        return (q[..., 0] * dtype(self.bins) + q[..., 1]) * dtype(self.bins) + q[..., 2]

    def _counts(self, flat_index):
        return np.bincount(flat_index.ravel(), minlength=self.bins**3)

    def _normalize(self, counts):
        total = int(counts.sum())
        if total == 0:
            return None
        return counts.astype(np.float32) / total

    @staticmethod
    def _box(cx, cy, width, height, shape):
        """Clip a box centred at (cx, cy) to the image; return integer x0, y0, x1, y1."""
        x0 = int(np.clip(round(cx - width / 2), 0, shape[1]))
        y0 = int(np.clip(round(cy - height / 2), 0, shape[0]))
        x1 = int(np.clip(round(cx + width / 2), 0, shape[1]))
        y1 = int(np.clip(round(cy + height / 2), 0, shape[0]))
        return x0, y0, x1, y1

    def _region_histograms(self, image, roi):
        x, y, width, height = roi
        cx, cy = x + width / 2, y + height / 2
        sx0, sy0, sx1, sy1 = self._box(cx, cy, width * self.surround_factor, height * self.surround_factor, image.shape)
        if sx1 <= sx0 or sy1 <= sy0:
            return None, None
        flat = self._flat_index(self._quantize(image[sy0:sy1, sx0:sx1]))

        # Both regions are rectangles, so slice them instead of building boolean masks, and get
        # the background (surround minus target box) by subtracting integer counts.
        tx0, ty0, tx1, ty1 = self._box(cx, cy, width, height, image.shape)
        target = self._counts(flat[ty0 - sy0 : ty1 - sy0, tx0 - sx0 : tx1 - sx0])

        # Box corners usually contain background, so learn the foreground from a shrunken box.
        ix0, iy0, ix1, iy1 = self._box(
            cx, cy, width * self.foreground_shrink, height * self.foreground_shrink, image.shape
        )
        inner = self._counts(flat[iy0 - sy0 : iy1 - sy0, ix0 - sx0 : ix1 - sx0])

        return self._normalize(inner), self._normalize(self._counts(flat) - target)

    def _refresh_lut(self):
        foreground, background = self._foreground, self._background
        total = foreground + background
        # Colors seen in neither region carry no evidence: give them the neutral prior 0.5.
        ratio = np.where(total > 1e-12, foreground / np.maximum(total, 1e-12), 0.5)
        self._lut = ratio.astype(np.float32).reshape(self.bins, self.bins, self.bins)

    def init(self, image, roi):
        foreground, background = self._region_histograms(image, roi)
        if foreground is None:
            raise ValueError("ROI must overlap the image")
        self._foreground = foreground
        # A target that fills the whole surround gives no background sample; assume nothing.
        self._background = background if background is not None else np.zeros_like(foreground)
        self._refresh_lut()

    def update(self, image, roi):
        if not self.initialized:
            self.init(image, roi)
            return
        foreground, background = self._region_histograms(image, roi)
        rate = self.learning_rate
        if foreground is not None:
            self._foreground = (1 - rate) * self._foreground + rate * foreground
        if background is not None:
            self._background = (1 - rate) * self._background + rate * background
        self._refresh_lut()

    def likelihood(self, patch):
        """Return P(foreground | color) per pixel as float32 in [0, 1]."""
        if not self.initialized:
            raise RuntimeError("Initialize the color model before querying it")
        q = self._quantize(patch)
        return self._lut[q[..., 0], q[..., 1], q[..., 2]]

    @staticmethod
    def to_grid(values, shape, border=0):
        """Resample a patch-sized map onto a feature grid of ``shape`` (rows, columns).

        ``border`` drops that many pixels per side first, matching the HOG cells that
        ``fhog.normalize_and_truncate`` removes.
        """
        if border:
            values = values[border:-border, border:-border]
        if values.shape[:2] == tuple(shape):
            return values.astype(np.float32, copy=False)
        return cv2.resize(values, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)

    def spatial_weight(self, patch, shape, border=0, floor=0.3):
        """Soft foreground weight on the feature grid: ``floor`` for background, 1 for foreground."""
        probability = cv2.GaussianBlur(self.likelihood(patch), (5, 5), 0)
        grid = self.to_grid(probability, shape, border)
        return (floor + (1.0 - floor) * grid).astype(np.float32)

    def response(self, patch, target_size, shape, border=0):
        """Mean foreground probability in a target-sized box centred on every position.

        ``target_size`` is (width, height) in patch pixels. The response is centred like the
        rearranged KCF response, so the two can be fused directly.
        """
        width = int(np.clip(round(target_size[0]), 1, patch.shape[1]))
        height = int(np.clip(round(target_size[1]), 1, patch.shape[0]))
        averaged = cv2.boxFilter(
            self.likelihood(patch), -1, (width, height), normalize=True, borderType=cv2.BORDER_REPLICATE
        )
        return self.to_grid(averaged, shape, border)
