import cv2
import numpy as np

from . import fft, fhog, recttool
from .color_model import ColorHistogramModel


# KCF tracker
class KCFTracker:
    def __init__(
        self,
        hog=False,
        fixed_window=True,
        multiscale=False,
        pv_thres=0.2,
        use_debackground: bool = False,
        use_color_model: bool = False,
        color_weight_floor: float = 1.0,
        color_merge_factor: float = 0.3,
    ):
        if use_debackground and use_color_model:
            raise ValueError("use_debackground and use_color_model are mutually exclusive")
        if not 0.0 <= color_weight_floor <= 1.0:
            raise ValueError("color_weight_floor must be in [0, 1]")
        if not 0.0 <= color_merge_factor <= 1.0:
            raise ValueError("color_merge_factor must be in [0, 1]")
        self.lambdar = 0.0001  # regularization
        self.padding = 2.5  # extra area surrounding the target
        self.output_sigma_factor = 0.125  # bandwidth of gaussian target
        self.pv_thres = pv_thres  # peak value threshold to stop updating on wrong object feature

        if hog:  # HOG feature
            # VOT
            self.interp_factor = 0.012  # linear interpolation factor for adaptation
            self.sigma = 0.6  # gaussian kernel bandwidth
            # TPAMI   #interp_factor = 0.02   #sigma = 0.5
            self.cell_size = 4  # HOG cell size
            self._hogfeatures = True
        else:  # raw gray-scale image # aka CSK tracker
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self._hogfeatures = False

        if multiscale:
            self.template_size = 96  # template size
            self.scale_step = 1.05  # scale step for multi-scale estimation
            self.scale_weight = 0.96  # to downweight detection scores of other scales for added stability
        elif fixed_window:
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        self._use_debackground = use_debackground
        # Color-histogram foreground model: soft feature weight plus Staple-style response fusion.
        self._color = ColorHistogramModel() if use_color_model else None
        # HOG weight of pure background; 1 (default) disables weighting. In a synthetic deformation
        # benchmark the weight helped alone but hurt when combined with fusion, so it is opt-in.
        self.color_weight_floor = color_weight_floor
        self.color_merge_factor = color_merge_factor  # share of the color response; 0 disables fusion

        self._tmpl_sz = [0, 0]  # cv::Size, [width,height]  #[int,int]
        self._roi = [0.0, 0.0, 0.0, 0.0]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self.size_patch = [0, 0, 0]  # [int,int,int]
        self._scale = 1.0  # float
        self._alphaf = None  # complex half spectrum (size_patch[0], size_patch[1] // 2 + 1)
        self._prob = None  # complex half spectrum (size_patch[0], size_patch[1] // 2 + 1)
        self._tmpl_f = None  # spectrum of self._tmpl, kept in sync by train()
        self._tmpl_sq = 0.0  # squared norm of self._tmpl
        self._tmpl = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
        self.hann = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])

    def sub_pixel_peak(self, left, center, right):
        divisor = 2 * center - right - left  # float
        return 0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor

    def create_hanning_mats(self):
        hann2t, hann1t = np.ogrid[0 : self.size_patch[0], 0 : self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t

        if self._hogfeatures:
            hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
            self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
        else:
            self.hann = hann2d
        self.hann = self.hann.astype(np.float32)

    def create_gaussian_peak(self, sizey, sizex):
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fft.rfft2(res.astype(np.float32))

    def _spatial_shape(self):
        return (self.size_patch[0], self.size_patch[1])

    def _spectrum(self, x):
        """Batched half-spectrum of a feature map: all HOG channels go through one FFT call."""
        if self._hogfeatures:
            x = x.reshape((self.size_patch[2], self.size_patch[0], self.size_patch[1]))
        return fft.rfft2(x)

    def _kernel(self, xf, xx, zf=None, zz=None):
        """Gaussian kernel k(x, z) from precomputed spectra and squared norms.

        Cross-power spectra are summed over channels in the Fourier domain, so a whole
        multi-channel correlation costs one inverse FFT. ``zf=None`` means the
        auto-correlation k(x, x), which needs only |X|^2.
        """
        if zf is None:
            cross = xf.real * xf.real + xf.imag * xf.imag
            zz = xx
        else:
            cross = xf * zf.conj()
        if cross.ndim == 3:
            cross = cross.sum(axis=0)
        c = fft.rearrange(fft.irfft2(cross, self._spatial_shape()).astype(np.float32))
        d = (xx + zz - 2.0 * c) / (self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        d = d * (d >= 0)
        return np.exp(-d / (self.sigma * self.sigma))

    def gaussian_correlation(self, x1, x2):
        """Gaussian kernel between two spatial feature maps (reference/compatibility API)."""
        return self._kernel(self._spectrum(x1), float(np.sum(x1 * x1)), self._spectrum(x2), float(np.sum(x2 * x2)))

    def get_features(self, image, inithann, scale_adjust=1.0):
        return self._features(image, inithann, scale_adjust)[0]

    def _feature_border(self):
        """Patch pixels per side that the HOG grid drops (see fhog.normalize_and_truncate)."""
        return self.cell_size if self._hogfeatures else 0

    def _target_size_in_patch(self, scale_adjust):
        factor = 1.0 / (self._scale * scale_adjust)
        return self._roi[2] * factor, self._roi[3] * factor

    def _features(self, image, inithann, scale_adjust=1.0, weighted=False):
        """Return the windowed feature map and the template-sized patch it came from.

        ``weighted`` multiplies in the soft color weight computed from this same patch. Only
        training samples are weighted, as in CSR-DCF: the filter learns mostly from the target,
        while search features stay unweighted. Weighting the search too would apply the previous
        frame's mask to both sides, and the matching mask patterns would bias the peak toward zero
        motion.

        The weight applies to HOG only. Raw grayscale features are signed intensities: scaling the
        template's background without scaling the search's changes how the patch looks, and in
        testing the box drifted by more than 20 px. Grayscale tracking uses response fusion alone.
        """
        extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
        cx = self._roi[0] + self._roi[2] / 2  # float
        cy = self._roi[1] + self._roi[3] / 2  # float

        if inithann:
            padded_w = self._roi[2] * self.padding
            padded_h = self._roi[3] * self.padding

            if self.template_size > 1:
                if padded_w >= padded_h:
                    self._scale = padded_w / float(self.template_size)
                else:
                    self._scale = padded_h / float(self.template_size)
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.0

            if self._hogfeatures:
                self._tmpl_sz[0] = (
                    int(self._tmpl_sz[0]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
                )
                self._tmpl_sz[1] = (
                    int(self._tmpl_sz[1]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
                )
            else:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0])
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        z = recttool.subwindow(
            image, extracted_roi, use_debackground=self._use_debackground, border_type=cv2.BORDER_REPLICATE
        )
        if z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]:
            z = cv2.resize(z, tuple(self._tmpl_sz))

        if self._hogfeatures:
            mapp = {"size_x": 0, "size_y": 0, "num_features": 0, "map": 0}
            mapp = fhog.get_feature_maps(z, self.cell_size, mapp)
            mapp = fhog.normalize_and_truncate(mapp, 0.2)
            mapp = fhog.pca_feature_maps(mapp)
            self.size_patch = map(int, [mapp["size_y"], mapp["size_x"], mapp["num_features"]])
            self.size_patch = list(self.size_patch)
            features_map = (
                mapp["map"].reshape((self.size_patch[0] * self.size_patch[1], self.size_patch[2])).T
            )  # (size_patch[2], size_patch[0]*size_patch[1])
        else:
            if z.ndim == 3 and z.shape[2] == 3:
                features_map = cv2.cvtColor(
                    z, cv2.COLOR_BGR2GRAY
                )  # z:(size_patch[0], size_patch[1], 3)  features_map:(size_patch[0], size_patch[1])   #np.int8  #0~255
            elif z.ndim == 2:
                features_map = z  # (size_patch[0], size_patch[1]) #np.int8  #0~255
            features_map = features_map.astype(np.float32) / 255.0 - 0.5
            self.size_patch = [z.shape[0], z.shape[1], 1]

        if inithann:
            self.create_hanning_mats()  # create_hanning_mats need size_patch

        window = self.hann
        if weighted and self._hogfeatures and self._color is not None and self.color_weight_floor < 1.0:
            grid = (self.size_patch[0], self.size_patch[1])
            weight = self._color.spatial_weight(z, grid, self._feature_border(), self.color_weight_floor)
            window = window * weight.reshape(1, -1)
        return window * features_map, z

    def detect(self, z, x, patch=None, scale_adjust=1.0):
        if z is self._tmpl:  # the template's spectrum and norm are cached by train()
            zf, zz = self._tmpl_f, self._tmpl_sq
        else:
            zf, zz = self._spectrum(z), float(np.sum(z * z))
        k = self._kernel(self._spectrum(x), float(np.sum(x * x)), zf, zz)
        res = fft.irfft2(self._alphaf * fft.rfft2(k), k.shape).astype(np.float32)

        if patch is not None and self._color is not None and self.color_merge_factor > 0:
            color = self._color.response(
                patch, self._target_size_in_patch(scale_adjust), res.shape, self._feature_border()
            )
            res = (1 - self.color_merge_factor) * res + self.color_merge_factor * color

        _, pv, _, pi = cv2.minMaxLoc(res)  # pv:float  pi:tuple of int
        p = [float(pi[0]), float(pi[1])]  # cv::Point2f, [x,y]  #[float,float]

        if pi[0] > 0 and pi[0] < res.shape[1] - 1:
            p[0] += self.sub_pixel_peak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])
        if pi[1] > 0 and pi[1] < res.shape[0] - 1:
            p[1] += self.sub_pixel_peak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        p[0] -= res.shape[1] / 2.0
        p[1] -= res.shape[0] / 2.0

        return p, pv

    def _detect_at_scale(self, image, scale_adjust):
        features, patch = self._features(image, 0, scale_adjust)
        if self._color is None:
            return self.detect(self._tmpl, features)
        return self.detect(self._tmpl, features, patch, scale_adjust)

    def train(self, x, train_interp_factor):
        xf = self._spectrum(x)
        k = self._kernel(xf, float(np.sum(x * x)))
        # The original packed-complex code added lambdar to both the real and imaginary parts;
        # kept for bit-for-bit comparable behaviour.
        alphaf = self._prob / (fft.rfft2(k) + self.lambdar * (1 + 1j))

        a = train_interp_factor
        self._tmpl = (1 - a) * self._tmpl + a * x
        # The DFT is linear, so the template spectrum follows the same interpolation as the template.
        self._tmpl_f = (1 - a) * self._tmpl_f + a * xf
        self._tmpl_sq = float(np.sum(self._tmpl * self._tmpl))
        self._alphaf = (1 - a) * self._alphaf + a * alphaf

    def init(self, roi, image):
        if len(roi) != 4 or not np.all(np.isfinite(roi)) or roi[2] <= 0 or roi[3] <= 0:
            raise ValueError("ROI must be finite [x, y, width, height] with positive dimensions")
        self._roi = list(map(float, roi))
        if self._color is not None:
            self._color.init(image, self._roi)  # discards any previous target's histograms
        self._tmpl, _ = self._features(image, 1, weighted=True)
        self._prob = self.create_gaussian_peak(self.size_patch[0], self.size_patch[1])
        self._alphaf = np.zeros_like(self._prob)
        self._tmpl_f = np.zeros_like(self._spectrum(self._tmpl))
        self.train(self._tmpl, 1.0)

    def update(self, image):
        if self._tmpl is None:
            raise RuntimeError("Initialize the tracker before updating it")
        if self._roi[0] + self._roi[2] <= 0:
            self._roi[0] = -self._roi[2] + 1
        if self._roi[1] + self._roi[3] <= 0:
            self._roi[1] = -self._roi[3] + 1
        if self._roi[0] >= image.shape[1] - 1:
            self._roi[0] = image.shape[1] - 2
        if self._roi[1] >= image.shape[0] - 1:
            self._roi[1] = image.shape[0] - 2

        cx = self._roi[0] + self._roi[2] / 2.0
        cy = self._roi[1] + self._roi[3] / 2.0

        loc, peak_value = self._detect_at_scale(image, 1.0)
        scale_adjust = 1.0

        if self.scale_step != 1:
            # Test at a smaller _scale
            new_loc1, new_peak_value1 = self._detect_at_scale(image, 1.0 / self.scale_step)
            # Test at a bigger _scale
            new_loc2, new_peak_value2 = self._detect_at_scale(image, self.scale_step)

            if self.scale_weight * new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2:
                loc = new_loc1
                peak_value = new_peak_value1
                scale_adjust = 1.0 / self.scale_step
            elif self.scale_weight * new_peak_value2 > peak_value:
                loc = new_loc2
                peak_value = new_peak_value2
                scale_adjust = self.scale_step

        if peak_value < self.pv_thres:
            return [0.0, 0.0, 0.0, 0.0]

        self._scale *= scale_adjust
        self._roi[2] *= scale_adjust
        self._roi[3] *= scale_adjust
        self._roi[0] = cx - self._roi[2] / 2.0 + loc[0] * self.cell_size * self._scale
        self._roi[1] = cy - self._roi[3] / 2.0 + loc[1] * self.cell_size * self._scale

        if self._roi[0] >= image.shape[1] - 1:
            self._roi[0] = image.shape[1] - 1
        if self._roi[1] >= image.shape[0] - 1:
            self._roi[1] = image.shape[0] - 1
        if self._roi[0] + self._roi[2] <= 0:
            self._roi[0] = -self._roi[2] + 2
        if self._roi[1] + self._roi[3] <= 0:
            self._roi[1] = -self._roi[3] + 2
        assert self._roi[2] > 0 and self._roi[3] > 0

        if self._color is not None:
            self._color.update(image, self._roi)
        x, _ = self._features(image, 0, 1.0, weighted=True)
        self.train(x, self.interp_factor)

        return self._roi
