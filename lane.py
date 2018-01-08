import numpy as np
import scipy.ndimage
import edges
import utils
import cv2

def perform_checks(params):
    # Standard limits for anomaly detection
    LIMITS = dict(
        curverad_diff = (0, np.inf),
        curverad_abs = (0, np.inf),
        weights_sum = (-np.inf, np.inf),
        curverad_consistency = (-np.inf, np.inf),
    )
    passed = True
    for k,v in params.items():
        if not np.isnan(v) and not (LIMITS[k][0] <= v <= LIMITS[k][1]):
            print("Check violated %r" % k)
            passed = False
    return passed


class Line(object):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Define window size for centroid search
    width = 80
    height = 80

    def __init__(self, straighten, xoffset=0, alpha=0.9, max_missed=10):
        self.straighten = straighten
        self.xoffset = 0
        self.alpha = alpha
        self.max_missed = max_missed
        self.reset()

    def find(self, mask, margin=50, fit_degree=2, polyfit=None):
        """Find a lane line in the image mask.
        Returns:
          x, y, w:  pixel coordinates and weights of found centroids
          polyfit:  polynomial coefficients interpolating centroids pixel coordinates
          curverad: curvature radius

        All coordinates are referred to the top-down view defined by self.straighten.
        """
        x, y, w = edges.find_window_centroids(mask,
                                              window_width=self.width,
                                              window_height=self.height,
                                              margin=margin,
                                              polyfit=polyfit,
                                              xoffset=self.xoffset,
        )
        if x is None:
            return x, y, w, None, [np.nan]*2
        x += self.xoffset

        coeffs_px = utils.polyfit(y, x, deg=fit_degree, robust=True)
        coeffs_m = utils.polyfit(y*self.ym_per_pix, x*self.xm_per_pix, deg=fit_degree, robust=True)
        y_eval = (mask.shape[0] - 1) * self.ym_per_pix
        curverad = (1 + np.polyval(np.polyder(coeffs_m), y_eval)**2)**1.5 \
                   / np.polyval(np.polyder(coeffs_m, 2), y_eval)
        position = np.fmax(mask.shape[1] - np.polyval(coeffs_px, mask.shape[0]),
                           np.polyval(coeffs_px, mask.shape[0]) - self.xoffset
                   ) * self.xm_per_pix

        return x, y, w, coeffs_px, curverad, position

    def reset(self):
        self.centroids = (np.array([np.nan]),)*3
        self.polyfit = None
        self.curverad = np.nan
        self.position = np.nan
        self.missed = 0
        self.x, self.y, self.topdown_x = (np.array([np.nan]),)*3
        self.found = False
        self.mask = None
        self.old = (self.centroids, self.polyfit, self.curverad, self.position,
                    self.found, self.mask, self.missed, self.x, self.y, self.topdown_x)
        self.checks = {}

    def step_forward(self, mask, centroids, polyfit, curverad, position, found=True):
        self.old = (self.centroids, self.polyfit, self.curverad, self.position,
                    self.found, self.mask, self.missed, self.x, self.y, self.topdown_x)
        self.centroids = centroids
        self.polyfit = polyfit
        self.curverad = curverad
        self.position = position
        self.found = found
        self.mask = mask
        return self.current(mask)

    def step_back(self, mask):
        print("Stepping back")
        self.centroids, self.polyfit, self.curverad, self.position, self.found, \
            self.mask, self.missed, self.x, self.y, self.topdown_x = self.old
        self.missed += 1
        return self.current(mask)

    def current(self, mask):
        if not self.found:
            return self.x, self.y
        shape = self.straighten.output_shape or mask.shape
        self.topdown_y = np.arange(shape[0])
        self.topdown_x = np.polyval(self.polyfit, self.topdown_y)
        self.x, self.y = self.straighten.inverse(self.topdown_x, self.topdown_y)
        return self.x, self.y

    def update(self, mask, *args, **kwargs):
        """Find line and update current state.
        See self.find for kwargs explaination."""
        old = None
        if self.missed >= self.max_missed:
            old = self.old #backup, in case the following find fails
            self.reset()
        polyfit = self.polyfit
        if self.mask is not None:
            mask = mask*self.alpha + self.mask*(1-self.alpha)
        x, y, w, coeffs, curverad, position = self.find(mask, polyfit=polyfit, *args, **kwargs)
        self.compute_checks(x, y, w, coeffs, curverad, position)
        if perform_checks(self.checks):
            self.step_forward(mask, (x, y, w), coeffs, curverad, position, found=x is not None)
        else:
            if old is None:
                self.missed += 1
            else:
                self.old = old
                self.step_back(mask)

        return self.x, self.y

    def compute_checks(self, x, y, w, coeffs, curverad, position):
        if x is None:
            self.checks = dict(
                curverad_diff = np.nan,
                curverad_abs  = np.nan,
                weights_sum   = np.nan,
            )
        else:
            self.checks = dict(
                curverad_abs  = np.abs(curverad) / 2,
                weights_sum   = w.sum(),
            )

    def plot_centroids(self, image, topdown_view=True):
        if not self.found:
            return utils.fillrects(image, [], color=(0,0,0), alpha=0)
        if topdown_view:
            centroids = zip(*self.centroids[:2])
        else:
            centroids = zip(*self.straighten.inverse(*self.centroids[:2]))
        return utils.fillrects(image,
                               [edges.get_rect_from_centroid(c, self.width, self.height)
                                for c in centroids],
                               color=(0, 255, 0),
                               alpha=0.3)

    def draw_line(self, image, topdown_view=True):
        if not self.found:
            return image
        shape = self.straighten.output_shape or image.shape
        y = np.arange(shape[0])
        x = np.polyval(self.polyfit, y)
        if not topdown_view:
            x, y = self.straighten.inverse(x, y)
        poly = list(zip(x, y))
        return utils.polyline(image, poly, color=(255, 0, 0), alpha=0.6, thickness=5)

    def draw_mask(self, image, topdown_view=True):
        mask = self.mask.astype(float)
        if not topdown_view:
            mask = self.straighten.inverse(mask)
        mask = ((mask - mask.min()) / (mask.max() - mask.min())).astype('uint8')
        canvas = np.copy(image)
        y, x = mask.nonzero()
        x += self.xoffset
        canvas[y, x, :] = (255, 255, 255)
        return cv2.addWeighted(image, 0.5, canvas, 0.5, 0)

class Lane(object):
    def __init__(self, straighten, alpha=0.5, max_missed=10, xm_per_pix=3.7/700, ym_per_pix=30/720, **kwargs):
        self.straighten = straighten
        self.left_line = Line(straighten, 0, alpha, max_missed)
        self.rght_line = Line(straighten, 0, alpha, max_missed)
        self.left_line.xm_per_pix = self.rght_line.xm_per_pix = xm_per_pix
        self.left_line.ym_per_pix = self.rght_line.ym_per_pix = ym_per_pix
        self.kwargs = kwargs

    def reset(self):
        self.left_line.reset()
        self.rght_line.reset()

    def preprocess_image(self, image):
        img = self.straighten(image)
        mask = edges.threshold(img, **self.kwargs)

        mask = scipy.ndimage.binary_opening(mask, np.ones((9,9)))
        mask = scipy.ndimage.binary_closing(mask, np.ones((9,9)))

        return mask

    def _apply(self, func, image, *args, **kwargs):
        mask = self.preprocess_image(image)
        offset = mask.shape[1]//2
        self.rght_line.xoffset = offset
        left_result = getattr(self.left_line, func)(mask[:, :offset], *args, **kwargs)
        rght_result = getattr(self.rght_line, func)(mask[:, offset:], *args, **kwargs)
        return left_result, rght_result

    def find(self, image, margin=50, fit_degree=2):
        return self._apply('find', image, margin, fit_degree)

    def update(self, image, margin=50, fit_degree=2):
        result = self._apply('update', image, margin, fit_degree)
        self.compute_checks()
        if not perform_checks(self.checks):
            result = self.left_line.step_back(image.shape), \
                     self.rght_line.step_back(image.shape)
        return result

    def compute_checks(self):
        self.checks = dict(
            curverad_consistency =
                np.abs(self.left_line.curverad -
                       self.rght_line.curverad),
        )

    @property
    def all_checks(self):
        checks = self.checks.copy()
        for k,v in self.left_line.checks.items():
            checks['left_' + k] = v
        for k,v in self.rght_line.checks.items():
            checks['rght_' + k] = v
        return checks

    def plot_centroids(self, image, topdown_view=False):
        image = self.left_line.plot_centroids(image, topdown_view)
        image = self.rght_line.plot_centroids(image, topdown_view)
        return image

    def draw_lines(self, image, topdown_view=False):
        image = self.left_line.draw_line(image, topdown_view)
        image = self.rght_line.draw_line(image, topdown_view)
        return image

    def draw_lane(self, image, topdown_view=False):
        image = self.draw_lines(image, topdown_view)
        if topdown_view:
            x = self.left_line.topdown_x.tolist()[::-1] + \
                self.rght_line.topdown_x.tolist()
            y = self.left_line.topdown_y.tolist()[::-1] + \
                self.rght_line.topdown_y.tolist()
        else:
            x = self.left_line.x.tolist()[::-1] + \
                self.rght_line.x.tolist()
            y = self.left_line.y.tolist()[::-1] + \
                self.rght_line.y.tolist()
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            return image
        poly = list(zip(x, y))
        image = utils.polyline(image, poly, color=(0, 255, 0), filled=True, alpha=0.2)
        return image

    def draw_mask(self, image, topdown_view=True):
        image = self.left_line.draw_mask(image, topdown_view)
        image = self.rght_line.draw_mask(image, topdown_view)
        return image

    def draw_info(self, image):
        curvature = (self.left_line.curverad, self.rght_line.curverad)
        curvature = curvature[np.argmin(np.abs(curvature))]
        if abs(curvature) >= 20000:
            curvature = np.inf
        position = (self.left_line.position - self.rght_line.position) / 2
        text = "Curvature: %4.1f km" % (-9.9)
        text_params = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1)
        text_size = cv2.getTextSize(text, thickness=10, **text_params)
        for color,size in [((255,255,255), 10), ((0,0,0), 2)]:
            text = "Curvature: %4.1f km" % (curvature/1000)
            image = cv2.putText(image, text, (10, text_size[0][1] + 10),
                                color=color, thickness=size,
                                lineType=cv2.LINE_AA, **text_params)
            text = " Position: %4.1f m" % (position)
            image = cv2.putText(image, text, (text_size[0][0] + 10, text_size[0][1] + 10),
                                color=color, thickness=size,
                                lineType=cv2.LINE_AA, **text_params)
        return image
