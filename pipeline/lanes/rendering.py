from enum import Enum

import cv2
import numpy as np

from .tracks import Track
from .types import Rects


class LaneColor(Enum):
    Valid = (1, 0.5, 0.1)
    Cached = (0.75, 0.1, 1)
    Warning = (0.05, 0.1, 1)


def render_rects(canvas: np.ndarray, rects: Rects, alpha: float):
    for rect in rects:
        cv2.rectangle(canvas, tuple(rect[:2]), tuple(rect[2:]), color=(0, 0, alpha), thickness=1)


def render_lane(canvas: np.ndarray, track: Track, color: LaneColor=LaneColor.Valid):
    if len(track.rects) == 0:
        return
    h, w = canvas.shape[:2]
    highest_rect = track.rects[-1][1]
    highest_rect = min(h // 2, highest_rect) if highest_rect is not None else h // 2
    ys = np.linspace(h - 1, highest_rect, h - highest_rect)
    xs = np.polyval(track.fit, ys)
    pts = np.int32([(x, y) for (x, y) in zip(xs, ys)])
    cv2.polylines(canvas, [pts], False, color=color.value, thickness=4, lineType=cv2.LINE_AA)
