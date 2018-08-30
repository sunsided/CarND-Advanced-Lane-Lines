import cv2
import numpy as np
from typing import Tuple


class LaneColorMasking:
    """
    Obtains masks for yellow and white lane markings from a color image.
    """
    def __init__(self, light_cutoff: float=.92, blue_threshold: int=30, luminance_kernel_width: int=127):
        self.light_cutoff = light_cutoff
        self.blue_threshold = blue_threshold
        self._lc_kernel_size = luminance_kernel_width
        self.hough_line_support = 10
        self.hough_line_length = 20
        self.hough_line_gap = 5
        self.canny_lo = 64
        self.canny_hi = 180
        self.detect_lines = False

    def process(self, img: np.ndarray, is_lab: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes the specified image.
        :param img: The image to obtain masks from.
        :param is_lab: If False, the image is assumed to be BGR and will be converted to L*a*b*;
                       if True, the image is assumed to be L*a*b* already.
        :return: A tuple consisting of the white and yellow lane mask.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) if not is_lab else img
        l_mask, b_mask = self._lab_mask2(lab)
        if self.detect_lines:
            l_mask = self._hough(l_mask)
            b_mask = self._hough(b_mask)
        return l_mask, b_mask

    def _hough(self, mask: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(mask, self.canny_lo, self.canny_hi)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 90, self.hough_line_support,
                                minLineLength=self.hough_line_length, maxLineGap=self.hough_line_gap)
        if lines is None:
            return mask
        edge_lines = np.zeros_like(edges)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(edge_lines, (x1, y1), (x2, y2), 255, 2)
        return edge_lines

    def _lab_mask2(self, lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtains the masks.
        :param lab: The L*a*b* image.
        :return: A tuple consisting of the white and yellow lane mask.
        """
        li = self._luminance_constancy_lab(lab[..., 0])
        if self._lc_kernel_size > 0:
            li = cv2.equalizeHist(li)
        b = lab[..., 2]
        if self._lc_kernel_size > 0:
            b = cv2.equalizeHist(np.uint8(b * 256 - 127))

        l_mask = np.zeros(shape=li.shape[:2], dtype=np.uint8)
        l_mask[li >= self.light_cutoff * li.max()] = 255
        b_mask = np.zeros_like(l_mask)
        b_mask[b >= self.blue_threshold] = 255

        return l_mask, b_mask

    def _luminance_constancy_lab(self, channel: np.ndarray) -> np.ndarray:
        """
        Performs local luminance equalization.
        :param channel: The channel to operate on.
        :return: The adjusted channel.
        """
        if self._lc_kernel_size == 0:
            return channel
        channel = np.float32(channel / 255.)
        blurred = cv2.GaussianBlur(channel, (self._lc_kernel_size, self._lc_kernel_size), 0)
        adjusted = channel / (blurred + 0.001)
        vmin, vmax = adjusted.min(), adjusted.max()
        adjusted = (adjusted - vmin) / (vmax - vmin)
        return np.clip(adjusted * 255, 0, 255).astype(np.uint8)
