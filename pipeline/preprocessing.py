import cv2
import numpy as np
from typing import Optional, Union, Tuple

from pipeline.edges import *


def lab_enhance_yellow(img: np.ndarray, normalize: bool=False, power: float=2, ypower: float=2) \
        -> Tuple[np.ndarray, np.ndarray]:
    img = (np.float32(img) / 255) if normalize else img

    # NOTE That the next statement is using RGB even though the input image
    # is required to be BGR! This is due to the development setup in the original
    # Jupyter notebook. Changing the channel order will change meaning of the L*a*b*
    # channels; this could be updated in the future. For now, it works as it is.
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    rg = (127 - lab[..., 1]) / 256
    yb = (127 - lab[..., 2]) / 256

    yb[yb < 0.4] = 0  # suppress blue
    yb[rg < 0.2] = 0  # suppress red
    yb[rg > 0.8] = 0  # suppress green

    if ypower != 1:
        yb = yb ** ypower

    # Normalize "yellow channel" to 0..1
    cv2.normalize(yb, yb, 1, norm_type=cv2.NORM_MINMAX)

    # OpenCV L*a*b*'s L is 0..100
    gray = lab[..., 0] / 100
    if power != 1:
        gray = gray ** power

    # Whatever is brighter is our pixel value.
    mixed = cv2.max(gray, yb)
    cv2.normalize(mixed, mixed, 1, norm_type=cv2.NORM_MINMAX)

    lab[..., 0] = mixed * 100

    # NOTE: As above, we're fixing the channels now.
    lab = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), cv2.COLOR_BGR2LAB)
    return mixed, lab


def detect_lane_pixels(frame: np.ndarray, gray: np.ndarray, edg: Optional[Union[EdgeDetectionConv, EdgeDetectionNaive]],
                       tmp: Optional[Union[EdgeDetectionTemporal, EdgeDetectionSWT, EdgeDetectionTemplateMatching]],
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
        edges_static = edg.detect(gray).astype(np.float32) / 255.
        mask_sum += edges_static * mask

    if tmp is not None:
        edges_temporal = tmp.filter(gray)
        mask_sum += edges_temporal * mask

    scaled = mask_sum
    cv2.normalize(scaled, scaled, 1, norm_type=cv2.NORM_MINMAX)

    scaled[scaled < threshold] = 0
    return scaled


def detect_lane_pixels_2(frame: np.ndarray, gray: np.ndarray, edg: Optional[Union[EdgeDetectionConv, EdgeDetectionNaive]],
                       tmp: Optional[Union[EdgeDetectionTemporal, EdgeDetectionSWT, EdgeDetectionTemplateMatching]],
                       lcm: Optional[LaneColorMasking],
                       threshold: float=.05) -> np.ndarray:
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
        edges_static = edg.detect(gray).astype(np.float32) / 255.
        mask_sum *= edges_static * mask

    if tmp is not None:
        edges_temporal = tmp.filter(gray)
        mask_sum *= edges_temporal * mask

    scaled = mask_sum
    cv2.normalize(scaled, scaled, 1, norm_type=cv2.NORM_MINMAX)

    scaled[scaled < threshold] = 0
    return scaled