import cv2
import numpy as np
from typing import Optional


def non_line_suppression(img: np.ndarray, edges: np.ndarray,
                         dx: Optional[np.ndarray]=None,
                         dy: Optional[np.ndarray]=None,
                         threshold: float=.7,
                         sum_window: int=10,
                         close_window: int=2):
    if dx is None:
        dx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
    if dy is None:
        dy = cv2.Scharr(img, cv2.CV_32F, 0, 1)

    # Alternatively, here's another trick: If we obtain the direction of each edge in the image,
    # then, within a certain area, we expect to find an opposing edge (i.e. the opposite direction).
    # The edge might be less strong, so we only look at the actual angle.
    directions = np.arctan2(dy, dx)
    amplitude = dx ** 2 + dy ** 2
    directions[amplitude == 0] = 0

    # By applying a box blur we sum up all pixel values in the given window. Since opposing edges
    # have (approximately) opposite values, areas of opposing values will have a low response.
    # We use this to build a mask that suppresses areas of high response, as they cannot contain
    # lo-hi-lo edges.
    filtered_directions = np.abs(cv2.boxFilter(directions, cv2.CV_32F, (sum_window, sum_window), normalize=True, borderType=cv2.BORDER_CONSTANT))
    filtered_directions = 1 - filtered_directions / filtered_directions.max()

    _, suppression = cv2.threshold(np.abs(filtered_directions), threshold, 1, cv2.THRESH_BINARY)
    edges_suppressed = edges * suppression.astype(np.uint8)
    edges_suppressed = cv2.morphologyEx(edges_suppressed, cv2.MORPH_CLOSE, np.ones((close_window, close_window), np.uint8))
    return edges_suppressed