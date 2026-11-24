"""fDSST-style discriminative scale filter (Danelljan et al., TPAMI 2017).

Translation is estimated once at the current scale by the KCF filter; this 1-D correlation filter
then estimates the scale change. Each scale sample is the target box (no padding) cropped at one of
``n_scales`` relative sizes, resized to a small fixed size and described by HOG, so a sample is one
feature column and a frame's samples form a ``(d, n_scales)`` matrix. The filter correlates along
the scale axis only. As in fDSST the features are compressed with a QR (PCA-like) basis to
``n_scales`` dimensions and the response is interpolated in the Fourier domain from ``n_scales`` to
``n_interp`` scales, which gives a fine step at a fraction of the cost of extracting every scale.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import fhog


def _hog(patch, cell_size):
    mapp = {"size_x": 0, "size_y": 0, "num_features": 0, "map": 0}
    mapp = fhog.get_feature_maps(patch, cell_size, mapp)
    mapp = fhog.normalize_and_truncate(mapp, 0.2)
    mapp = fhog.pca_feature_maps(mapp)
    return mapp["map"]


def _resize_dft(spectrum, length):
    """Zero-pad (or truncate) a 1-D spectrum along axis -1, as fDSST's ``resizeDFT``."""
    n = spectrum.shape[-1]
    m = min(n, length)
    mids, mide = (m + 1) // 2, (m - 1) // 2
    out = np.zeros(spectrum.shape[:-1] + (length,), np.complex128)
    out[..., :mids] = spectrum[..., :mids]
    if mide:
        out[..., -mide:] = spectrum[..., -mide:]
    return out * (length / n)


class ScaleFilter:
    def __init__(
        self,
        n_scales: int = 17,
        n_interp: int = 33,
        scale_step: float = 1.02,
        sigma_factor: float = 1.0 / 16,
        learning_rate: float = 0.025,
        lambdar: float = 1e-2,
        max_area: float = 512.0,
        cell_size: int = 4,
    ):
        self.n_scales = n_scales
        self.n_interp = n_interp
        self.learning_rate = learning_rate
        self.lambdar = lambdar
        self.max_area = max_area
        self.cell_size = cell_size

        half = (n_scales - 1) // 2
        exps = np.arange(-half, n_scales - half) * n_interp / n_scales
        self.factors = scale_step**exps  # sample order: smallest to largest
        sigma = n_interp * sigma_factor
        labels = np.exp(-0.5 * np.roll(exps, -half) ** 2 / sigma**2)  # peak at index 0
        self._yf = np.fft.fft(labels)
        ihalf = (n_interp - 1) // 2
        self.interp_factors = scale_step ** np.roll(np.arange(-ihalf, n_interp - ihalf), -ihalf)
        self._window = np.hanning(n_scales + 2)[1:-1]  # strictly positive, like MATLAB's hann(n)

        self._model_sz = (0, 0)
        self._num = None  # running mean of samples, (d, n_scales)
        self._sf_num = None  # (n_scales, n_scales) compressed numerator spectrum
        self._sf_den = None  # (n_scales,)
        self._basis = None  # (d, n_scales)

    def _samples(self, image, center, target_sz):
        """HOG of the target box at every relative scale, one column per scale.

        The resized samples are stacked vertically and described by one HOG call, which costs about
        a fifth of 17 separate calls. Each sample keeps a one-cell ring that is discarded, so the
        surviving cells are normalised only by cells of their own sample; the ring absorbs the
        gradient and soft-binning bleed between neighbouring samples.
        """
        mw, mh = self._model_sz
        cs = self.cell_size
        border = 2 * cs
        pw, ph = mw + border, mh + border
        stack = np.empty((ph * self.n_scales, pw, 3), np.uint8)
        for k, f in enumerate(self.factors):
            w = max(int(round(target_sz[0] * f * pw / mw)), 2)
            h = max(int(round(target_sz[1] * f * ph / mh)), 2)
            x0, y0 = int(np.floor(center[0] - w / 2)), int(np.floor(center[1] - h / 2))
            patch = _subwindow(image, x0, y0, w, h)
            stack[k * ph : (k + 1) * ph] = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_LINEAR)
        hog = _hog(stack, cs)  # cells (rows, cols, 31) with the stack's outer ring dropped
        rows, cols = ph // cs, mw // cs
        hog = hog.reshape(self.n_scales * rows - 2, cols, -1)
        hog = np.concatenate([hog[:1] * 0, hog, hog[:1] * 0])  # restore the dropped rows
        hog = hog.reshape(self.n_scales, rows, cols, -1)[:, 1:-1]
        return hog.reshape(self.n_scales, -1).T.astype(np.float64) * self._window

    def init(self, image, center, target_sz):
        w, h = target_sz
        k = np.sqrt(self.max_area / (w * h)) if w * h > self.max_area else 1.0
        cs = self.cell_size
        # At least 3x3 HOG cells survive the border so the descriptor keeps some layout.
        self._model_sz = (max(int(w * k) // cs, 3) * cs, max(int(h * k) // cs, 3) * cs)
        self._num = None
        self.update(image, center, target_sz)

    def estimate(self, image, center, target_sz):
        """Relative scale change of the target centred at ``center``."""
        xs = self._samples(image, center, target_sz)
        xsf = np.fft.fft(self._basis.T @ xs, axis=1)
        response_f = np.sum(self._sf_num * xsf, axis=0) / (self._sf_den + self.lambdar)
        response = np.real(np.fft.ifft(_resize_dft(response_f, self.n_interp)))
        i = int(np.argmax(response))
        # Parabolic sub-sample refinement on the circular response, in units of interp steps.
        left, right = response[i - 1], response[(i + 1) % self.n_interp]
        div = 2 * response[i] - left - right
        delta = 0.0 if abs(div) < 1e-12 else float(np.clip(0.5 * (right - left) / div, -0.5, 0.5))
        step_log = np.log(self.interp_factors[1])
        return float(self.interp_factors[i] * np.exp(delta * step_log)), xs

    def update(self, image, center, target_sz, xs=None):
        if xs is None:
            xs = self._samples(image, center, target_sz)
        lr = 1.0 if self._num is None else self.learning_rate
        self._num = xs if self._num is None else (1 - lr) * self._num + lr * xs
        self._basis, _ = np.linalg.qr(self._num)
        den_basis, _ = np.linalg.qr(xs)
        sf_proj = np.fft.fft(self._basis.T @ self._num, axis=1)
        sf_num = self._yf * np.conj(sf_proj)
        xsf = np.fft.fft(den_basis.T @ xs, axis=1)
        sf_den = np.sum(np.real(xsf * np.conj(xsf)), axis=0)
        self._sf_num = sf_num  # fDSST recomputes the numerator from the averaged samples
        self._sf_den = sf_den if self._sf_den is None or lr == 1.0 else (1 - lr) * self._sf_den + lr * sf_den


def _subwindow(image, x0, y0, w, h):
    """Crop with replicated borders for any part of the box outside the image."""
    ih, iw = image.shape[:2]
    x1, y1 = x0 + w, y0 + h
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, iw), min(y1, ih)
    if cx0 >= cx1 or cy0 >= cy1:  # entirely outside: use the nearest edge pixel
        cx0, cy0 = min(max(x0, 0), iw - 1), min(max(y0, 0), ih - 1)
        cx1, cy1 = cx0 + 1, cy0 + 1
    patch = image[cy0:cy1, cx0:cx1]
    if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
        patch = cv2.copyMakeBorder(
            patch,
            max(cy0 - y0, 0),
            max(y1 - cy1, 0),
            max(cx0 - x0, 0),
            max(x1 - cx1, 0),
            cv2.BORDER_REPLICATE,
        )
    if patch.ndim == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    return patch
