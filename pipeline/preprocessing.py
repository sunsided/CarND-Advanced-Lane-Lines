import cv2
import numpy as np
from typing import Optional, Union

from pipeline.edges import *


def detect_lane_pixels(frame: np.ndarray, edg: Optional[Union[EdgeDetectionConv, EdgeDetectionNaive]],
                       tmp: Optional[Union[EdgeDetectionTemporal, EdgeDetectionSWT]],
                       lcm: Optional[LaneColorMasking],
                       threshold: float=.25) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    mask_sum = np.ones(lab.shape[:2], np.float32)
    if lcm is not None:
        mask_l, mask_b = lcm.process(lab, is_lab=True)
        alpha = 0.
        mask = alpha * np.float32((mask_l | mask_b) / 255.) + (1. - alpha)
        mask_sum += mask_l.astype(np.float32) / 256 / 2
        mask_sum += mask_b.astype(np.float32) / 256 / 2
    else:
        mask = np.ones_like(mask_sum)

    if edg is not None:
        edges_static = edg.detect(lab, is_lab=True).astype(np.float32) / 255.
        mask_sum += edges_static * mask

    if tmp is not None:
        edges_temporal = tmp.filter(lab, is_lab=True)
        mask_sum += edges_temporal * mask

    scaled = (mask_sum - mask_sum.min()) / (mask_sum.max() - mask_sum.min())

    scaled[scaled < threshold] = 0
    return scaled
