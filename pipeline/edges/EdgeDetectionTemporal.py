import cv2
import numpy as np
from typing import Optional

from pipeline.edges.EdgeDetectionBase import EdgeDetectionBase
from pipeline.non_line_suppression import non_line_suppression


class EdgeDetectionTemporal(EdgeDetectionBase):
    """
    Obtains edge proposals using a temporal smoothing approach.
    """
    def __init__(self, mask: Optional[np.ndarray] = None, detect_lines: bool = False):
        super().__init__(detect_lines)
        self._mask = mask
        self._previous_edges = None
        self._previous_grays_slow = None
        self._previous_grays_fast = None
        self._edges_filtered = None
        self._alpha_slow = 0.1
        self._alpha_fast = 0.8
        self._alpha_edge = 0.4
        self._median_blur = 5
        self._smooth_close_kernel = np.ones((7, 7), np.uint8)
        self._dog_small_size = (5, 5)
        self._dog_big_size = (9, 9)
        self._edge_gaussian_size = (5, 5)
        self._edge_canny_lo = 64
        self._edge_canny_hi = 100
        self._edge_blackhat_kernel = np.ones((17, 17), np.uint8)
        self._edge_erode_kernel = np.ones((2, 2), np.uint8)
        self._area_lo = 300
        self._area_hi = 600

        self.detect_lines = detect_lines
        self.hough_line_support = 50
        self.hough_line_length = 50
        self.hough_line_gap = 5

        self.reset()

    def reset(self):
        self._previous_edges = None
        self._previous_grays_slow = None
        self._previous_grays_fast = None
        self._edges_filtered = None

    @property
    def edges_filtered(self):
        return self._edges_filtered

    def filter(self, img: np.ndarray, is_lab: bool=False) -> np.ndarray:
        """
        Filters the specified image.
        :param img: The image to obtain masks from.
        :param is_lab: If False, the image is assumed to be BGR and will be converted to L*a*b*;
                       if True, the image is assumed to be L*a*b* already.
        :return: The pre-filtered image.
        """
        gray = cv2.GaussianBlur(img, (3, 3), 5)

        # Equalize for edge detection
        equalized = np.ma.masked_equal(gray, 0)
        slice_height = 5
        for y in range(0, gray.shape[0], slice_height):
            top, bottom = y, y + slice_height
            equalized[top:bottom, ...] = cv2.equalizeHist(equalized[top:bottom, ...])
        gray = np.ma.filled(equalized, 0)
        gray = cv2.blur(gray, (10, 3))

        if self._previous_grays_slow is None:
            temporally_smoothed_slow = gray
            temporally_smoothed_fast = gray
        else:
            temporally_smoothed_slow = self._alpha_slow * gray + (1 - self._alpha_slow) * self._previous_grays_slow
            temporally_smoothed_fast = self._alpha_fast * gray + (1 - self._alpha_fast) * self._previous_grays_fast

        # For edge detection we're going to need an integral image.
        temporally_smoothed_slow_8 = self.float2uint8(temporally_smoothed_slow, 1)
        temporally_smoothed_fast_8 = self.float2uint8(temporally_smoothed_fast, 1)

        # The reflections of the dashboard can be found mostly in vertical edges.
        ts_edges_y = np.sqrt((cv2.Scharr(temporally_smoothed_slow_8, cv2.CV_32F, 0, 1) / 255.)**2)
        dashboard_mask = 1 - ts_edges_y
        dashboard_mask = cv2.medianBlur(dashboard_mask, self._median_blur)
        dashboard_mask = np.clip(dashboard_mask, 0, 1)

        # Apply difference of gaussian edge detection.
        inp_8 = cv2.morphologyEx(temporally_smoothed_fast_8, cv2.MORPH_CLOSE, self._smooth_close_kernel)
        inp = np.float32(inp_8) / 255.
        dog = cv2.GaussianBlur(inp, self._dog_small_size, 5) - cv2.GaussianBlur(inp, self._dog_big_size, 9)
        dog = self.rescale(np.clip(dog, 0, 1)) * dashboard_mask

        # Since the non-line suppression later attempts to only keep edges
        # of lo-hi-lo intensity, we need to take care not to trigger
        # on the lo-hi-lo-hi-lo waves created by DoG. For this,
        # we suppress every response we know to be low from the input image.
        dog[inp < 0.5] = 0

        # Obtain new edges.
        if self._previous_edges is None:
            self._previous_edges = np.zeros_like(dog)

        edges_filtered = self._alpha_edge * dog + (1 - self._alpha_edge) * self._previous_edges
        edges_filtered = self.rescale(edges_filtered)
        edges_filtered = cv2.GaussianBlur(edges_filtered, self._edge_gaussian_size, 5)

        # Run canny on the pre-filtered edges
        edges_filtered_8 = self.float2uint8(edges_filtered)
        edges_canny_8 = cv2.Canny(edges_filtered_8, self._edge_canny_lo, self._edge_canny_hi)

        # Suppress all edges that are not lo-hi-lo.
        edges_canny_8 = non_line_suppression(inp, edges_canny_8,
                                             threshold=0.8)
        edge_lines = edges_canny_8
        if self.detect_lines:
            lines = cv2.HoughLinesP(edges_canny_8, 1, np.pi / 90, self.hough_line_support,
                                    minLineLength=self.hough_line_length, maxLineGap=self.hough_line_gap)
            if lines is not None:
                edge_lines = np.zeros_like(edges_canny_8)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(edge_lines, (x1, y1), (x2, y2), 255, 2)
                    edge_lines = edges_canny_8 // 2 | edge_lines

        # Carry the current state on to the next time stamp
        self._previous_grays_fast = temporally_smoothed_fast
        self._previous_grays_slow = temporally_smoothed_slow
        self._previous_edges = edges_filtered

        self._edges_filtered = edges_filtered_8
        return np.float32(edge_lines * self._mask) / 255.

    @staticmethod
    def float2uint8(img: np.ndarray, scale: float=255) -> np.ndarray:
        return np.clip(img * scale, 0, 255).astype(np.uint8)

    @staticmethod
    def uint82float(img: np.ndarray, scale: float=255) -> np.ndarray:
        return np.float32(img) / scale

    @staticmethod
    def rescale(img: np.ndarray) -> np.ndarray:
        min_, max_ = img.min(), img.max()
        return (img - min_) / (max_ - min_)
