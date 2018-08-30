import cv2
import numpy as np

from pipeline.non_line_suppression import non_line_suppression
from pipeline.swt import swt_process_pixel


class EdgeDetectionBase:
    def __init__(self, morphological_filtering: bool=False, detect_lines: bool=True):
        self.detect_lines = detect_lines
        self.hough_line_support = 60
        self.hough_line_length = 20
        self.hough_line_gap = 5
        self.canny_lo = 64
        self.canny_hi = 180
        self._morphological_filtering = morphological_filtering
        self._stroke_filter = False
        pass

    def filter(self, img: np.ndarray) -> np.ndarray:
        """
        Filters the specified image.
        :param img: The image to obtain masks from.
        :return: The pre-filtered image.
        """
        raise NotImplemented

    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Processes the specified image.
        :param img: The image to obtain masks from.
        :return: An image containing the detected edges.
        """
        filtered = self.filter(img)

        # Detect edges
        edges = cv2.Canny(np.uint8(filtered * 255), self.canny_lo, self.canny_hi)

        # Obtain the gradients for filtering.
        # We only require local gradients, so obtaining them only when required would make sense.
        dx = cv2.Scharr(filtered, cv2.CV_32F, 1, 0)
        dy = cv2.Scharr(filtered, cv2.CV_32F, 0, 1)

        # Now the stroke width transform already detects lo-hi-lo edges for us, but it is an extremely slow
        # implementation I did.
        if self._stroke_filter:
            gradients = (dx, dy)
            for y in range(0, edges.shape[0]):
                for x in range(0, edges.shape[1]):
                    if edges[y, x] == 0:
                        continue
                    ray = swt_process_pixel((x, y), edges, gradients, min_length=5, max_length=20)
                    if ray is None:
                        edges[y, x] = 0
        else:
            edges = non_line_suppression(filtered, edges, dx, dy)

        if self._morphological_filtering:
            edges = cv2.morphologyEx(edges, cv2.MORPH_BLACKHAT, np.ones(shape=(5, 5)))
            edges = cv2.medianBlur(edges, 3)

        if self.detect_lines:
            lines = cv2.HoughLinesP(edges, 1, np.pi / 90, self.hough_line_support,
                                    minLineLength=self.hough_line_length, maxLineGap=self.hough_line_gap)
            edge_lines = np.zeros_like(edges)
            if lines is None:
                return edge_lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(edge_lines, (x1, y1), (x2, y2), 255, 2)
        else:
            edge_lines = edges

        return edge_lines
