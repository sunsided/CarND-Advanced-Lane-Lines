import cv2
import numpy as np
from typing import Optional


class EdgeDetectionTemporal:
    """
    Obtains edge proposals using a temporal smoothing approach.
    """
    def __init__(self, mask: Optional[np.ndarray]=None):
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
        self._dog_small_size = (9, 9)
        self._dog_big_size = (17, 17)
        self._edge_gaussian_size = (5, 5)
        self._edge_canny_lo = 64
        self._edge_canny_hi = 100
        self._edge_blackhat_kernel = np.ones((13, 13), np.uint8)
        self._edge_erode_kernel = np.ones((2, 2), np.uint8)
        self._area_lo = 300
        self._area_hi = 600
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
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) if not is_lab else img
        gray = cv2.GaussianBlur(lab[..., 0], (3, 3), 5)

        # Equalize for edge detection
        equalized = np.ma.masked_equal(gray, 0)
        slice_height = 5
        for y in range(0, gray.shape[0], slice_height):
            top, bottom = y, y + slice_height
            equalized[top:bottom, ...] = cv2.equalizeHist(equalized[top:bottom, ...])
        gray = np.ma.filled(equalized, 0)

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

        # Obtain new edges.
        if self._previous_edges is None:
            self._previous_edges = np.zeros_like(dog)

        edges_filtered = self._alpha_edge * dog + (1 - self._alpha_edge) * self._previous_edges
        edges_filtered = self.rescale(edges_filtered)
        edges_filtered = cv2.GaussianBlur(edges_filtered, self._edge_gaussian_size, 5)

        # Run canny on the pre-filtered edges
        edges_filtered_8 = self.float2uint8(edges_filtered)
        edges_canny_8 = cv2.Canny(edges_filtered_8, self._edge_canny_lo, self._edge_canny_hi)

        # We perform blob detection; for this, we close nearby contours.
        edges_contours_8 = cv2.morphologyEx(edges_canny_8, cv2.MORPH_BLACKHAT, self._edge_blackhat_kernel)
        edges_contours_8 = cv2.morphologyEx(edges_contours_8, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
        m2, contours, hierarchy = cv2.findContours(edges_contours_8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contours below a specified area will be discarded.
        # If the area is big enough, we count the contour as "good", otherwise
        # we're just going to use it as an indicator.
        good_contours = []
        ok_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._area_lo:
                continue
            if area > self._area_hi:
                good_contours.append(cnt)
            else:
                ok_contours.append(cnt)

        filled = edges_canny_8 // 2
        cv2.drawContours(filled, ok_contours, -1, 64, cv2.FILLED, cv2.LINE_4)
        cv2.drawContours(filled, good_contours, -1, 255, cv2.FILLED, cv2.LINE_4)

        # Carry the current state on to the next time stamp
        self._previous_grays_fast = temporally_smoothed_fast
        self._previous_grays_slow = temporally_smoothed_slow
        self._previous_edges = edges_filtered

        self._edges_filtered = edges_filtered_8
        return np.float32(filled * self._mask) / 255.

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
