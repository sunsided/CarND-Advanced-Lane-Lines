import cv2
import numpy as np
from typing import Tuple, Optional


class EdgeDetection:
    """
    Obtains edges for for further processing.
    """
    def __init__(self, lane_width: int=4, kernel_width: int=11,
                 mask: Optional[np.ndarray]=None,
                 morphological_filtering: bool=False, detect_lines: bool=True):
        """
        Initializes a new instance of the EdgeDetection class.
        :param lane_width: The expected width of the lane in pixels.
        :param kernel_width: The width of the kernel to use.
        """
        self._roi_mask = mask
        self._kernel_width = lane_width
        self._kernel_pad = kernel_width - lane_width
        self._morphological_filtering = morphological_filtering
        self.detect_lines = detect_lines
        self.hough_line_support = 60
        self.hough_line_length = 20
        self.hough_line_gap = 5
        self.canny_lo = 64
        self.canny_hi = 180
        self.filter_threshold = .08
        self._kernel = self._build_kernel(self._kernel_width, self._kernel_pad)

    def filter(self, img: np.ndarray, is_lab: bool=False) -> np.ndarray:
        """
        Filters the specified image.
        :param img: The image to obtain masks from.
        :param is_lab: If False, the image is assumed to be BGR and will be converted to L*a*b*;
                       if True, the image is assumed to be L*a*b* already.
        :return: The pre-filtered image.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) if not is_lab else img
        li = lab[..., 0]

        # Suppress non-lanes. In essence, this is an edge detection kernel that attempts
        # to enforce large areas of uniform lightness, followed by a confined lightness bump.
        # This both serves the purpose of detecting edges and suppressing uninteresting candidates.
        filtered = cv2.filter2D(np.float32(li) / 255., cv2.CV_32F, self._kernel)
        filtered[filtered < self.filter_threshold] = 0
        filtered = np.power(filtered, 1/4)
        if self._roi_mask is not None:
            filtered *= self._roi_mask
        return filtered

    def detect(self, img: np.ndarray, is_lab: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes the specified image.
        :param img: The image to obtain masks from.
        :param is_lab: If False, the image is assumed to be BGR and will be converted to L*a*b*;
                       if True, the image is assumed to be L*a*b* already.
        :return: An image containing the detected edges.
        """
        filtered = self.filter(img, is_lab)

        # Detect edges
        edges = cv2.Canny(np.uint8(filtered * 255), self.canny_lo, self.canny_hi)
        if self._morphological_filtering:
            edges = cv2.morphologyEx(edges, cv2.MORPH_BLACKHAT, np.ones(shape=(5, 5)))
            edges = cv2.medianBlur(edges, 3)

        if self.detect_lines:
            lines = cv2.HoughLinesP(edges, 1, np.pi / 90, self.hough_line_support,
                                    minLineLength=self.hough_line_length, maxLineGap=self.hough_line_gap)
            edge_lines = np.zeros_like(edges)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(edge_lines, (x1, y1), (x2, y2), 255, 2)
        else:
            edge_lines = edges

        return edge_lines

    @staticmethod
    def _build_kernel(width: int = 3, pad: int = 6):
        ksize = (width, width)
        psize = (pad, pad)
        kernel = np.float32(np.pad(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize), psize, 'constant'))
        kernel[kernel < 1] = -1 / np.sum(kernel < 1)
        kernel[kernel == 1] = 1 / np.sum(kernel == 1)
        return kernel