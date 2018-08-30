import cv2
import numpy as np
from typing import Optional

from pipeline.edges.EdgeDetectionBase import EdgeDetectionBase
from pipeline.swt import apply_swt, get_edges, get_gradients


class EdgeDetectionSWT(EdgeDetectionBase):
    """
    Obtains edges for for further processing using a Stroke Width Transform.
    """
    def __init__(self, mask: Optional[np.ndarray] = None, blur_size: int = 7, blur_strength: int = 5,
                 min_length: int = 3, max_length: int = 24, canny_lo: int = 8, canny_hi: int = 16,
                 edge_response: float = 0, out_blur_size: int = 3):
        """
        Initializes a new instance of the EdgeDetection class.
        :param mask: The ROI mask.
        :param blur_size: The window size of the pre-Canny blur.
        :param blur_strength: The strength (sigma) of the pre-Canny blur.
        :param min_length: The minimum required stroke length.
        :param max_length: The maximum allowed stroke length.
        :param canny_lo: The lower threshold for Canny edges.
        :param canny_hi: The upper threshold for Canny edges.
        :param edge_response: The minimum required edge response.
        :param out_blur_size: The final median blur window size.
        """
        super().__init__(morphological_filtering=False, detect_lines=False)
        self.mask = mask
        self.blur_size = blur_size
        self.blur_strength = blur_strength
        self.min_length = min_length
        self.max_length = max_length
        self.canny_lo = canny_lo
        self.canny_hi = canny_hi
        self.edge_response = edge_response
        self.median_window_size = out_blur_size

    def filter(self, img: np.ndarray) -> np.ndarray:
        """
        Filters the specified image.
        :param img: The image to obtain masks from.
        :return: The pre-filtered image.
        """
        li = img.copy()

        if self.blur_size > 0 and self.blur_strength > 0:
            blurred = cv2.GaussianBlur(li, (self.blur_size, self.blur_size), self.blur_strength)
        else:
            blurred = li

        edges = get_edges(blurred, self.canny_lo, self.canny_hi, 3) * self.mask
        gradients = get_gradients(blurred)
        swt = apply_swt(blurred, edges, gradients, edge_response=self.edge_response,
                        min_length=self.min_length, max_length=self.max_length)
        result = cv2.medianBlur(swt, self.median_window_size) if self.median_window_size > 0 else swt
        return np.float32(result) / 255.
