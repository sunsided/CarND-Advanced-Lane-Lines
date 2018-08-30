import cv2
import numpy as np
from typing import Optional
from pipeline.swt import swt_process_pixel
from pipeline.non_line_suppression import non_line_suppression


class EdgeDetectionNaive:
    """
    Obtains edges for for further processing by finding edges and then
    sieving out those edges that do not have a corresponding counterpart
    in a given distance. The result is locations where hi-lo-hi
    crossings can be found.
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
        self.filter_threshold = .03
        self._stroke_filter = False
        self._kernel = self._build_kernel(self._kernel_width, self._kernel_pad)

    def filter(self, img: np.ndarray) -> np.ndarray:
        """
        Filters the specified image.
        :param img: The image to obtain masks from.
        :return: The pre-filtered image.
        """
        li = cv2.blur(img, (5, 5))

        canny_8 = cv2.Canny(li, 16, 32) * self._roi_mask
        output = np.zeros(canny_8.shape[:2], np.float32)

        ys, xs = np.nonzero(canny_8)
        h, w = canny_8.shape[:2]
        max_width = 25
        min_width = 3
        threshold = 0.95
        for i in range(len(ys)):
            x, y = xs[i], ys[i]
            if x == 0 or (x + 1) == w:
                continue
            vl = li[y, x-1]
            vc = li[y, x+1]
            if vc == 0:
                continue
            left_ratio = vl / vc
            if left_ratio > threshold:
                continue
            for x2 in range(x+2, min(x + max_width, w - 2)):
                if canny_8[y, x2] == 0:
                    continue
                if x + min_width >= x2:
                    continue
                vr = li[y, x2 + 1]
                right_ratio = vr / vc
                if right_ratio < threshold:
                    output[y, x] = left_ratio
                    output[y, x2] = right_ratio
                    output[y, (x + x2) // 2] = (left_ratio + right_ratio) / 2
                break

        return output

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

    @staticmethod
    def _build_kernel(width: int = 3, pad: int = 6):
        ksize = (width, width)
        psize = (pad, pad)
        kernel = np.float32(np.pad(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize), psize, 'constant'))
        kernel[kernel < 1] = -1 / np.sum(kernel < 1)
        kernel[kernel == 1] = 1 / np.sum(kernel == 1)
        return kernel
