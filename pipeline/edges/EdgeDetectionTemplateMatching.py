import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from typing import Optional

from pipeline.edges.EdgeDetectionBase import EdgeDetectionBase


class EdgeDetectionTemplateMatching(EdgeDetectionBase):
    """
    Obtains edges for for further processing.
    """

    def __init__(self, path: str, workers: int = 8, mask: Optional[np.ndarray] = None, detect_lines: bool = False):
        """
        Initializes a new instance of the EdgeDetection class.
        """
        super().__init__(detect_lines, detect_lines=detect_lines)
        self._negatives = [np.float32(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) / 255
                           for path in glob.glob(os.path.join(path, '**', 'negative-*.png'), recursive=True)]
        self._positives = [np.float32(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) / 255
                           for path in glob.glob(os.path.join(path, '**', 'positive-*.png'), recursive=True)]
        self._roi_mask = mask
        self._pe = ThreadPoolExecutor(max_workers=workers)
        self._mode = cv2.TM_CCOEFF

    def filter(self, img: np.ndarray) -> np.ndarray:
        """
        Filters the specified image.
        :param img: The image to obtain masks from.
        :return: The pre-filtered image.
        """
        gray = img

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        mode = self._mode

        def process(template):
            m = cv2.matchTemplate(gray, template, mode)
            m[m < 0] = 0
            return m

        pos_matched = self._pe.map(process, self._positives)
        neg_matched = self._pe.map(process, self._negatives)

        pos_sum = np.zeros_like(gray)
        for result in pos_matched:
            pos_sum[8:745 + 8, 8:285 + 8] += result
        pos_sum /= len(self._positives)

        neg_sum = np.zeros_like(gray)
        for result in neg_matched:
            neg_sum[8:745 + 8, 8:285 + 8] += result
        neg_sum /= len(self._negatives)

        mask = (1 - neg_sum) * pos_sum

        mask[mask < 0] = 0
        mask = cv2.normalize(mask, 1, cv2.NORM_MINMAX)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask[mask < 0.05] = 0
        mask = cv2.normalize(mask, 1, cv2.NORM_MINMAX)

        if self._roi_mask is not None:
            mask *= self._roi_mask
        return mask
