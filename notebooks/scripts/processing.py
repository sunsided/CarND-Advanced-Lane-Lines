import sys
sys.path.append("..")
import os
import cv2
import numpy as np
from pipeline import CameraCalibration, BirdsEyeView, ImageSection, Point

_cc = CameraCalibration.from_pickle(os.path.join('..', 'calibration.pkl'))

_section = ImageSection(
    top_left=Point(x=580, y=461.75),
    top_right=Point(x=702, y=461.75),
    bottom_right=Point(x=1013, y=660),
    bottom_left=Point(x=290, y=660),
)

_bev = BirdsEyeView(_section,
                    section_width=3.6576,  # one lane width in meters
                    section_height=2 * 13.8826)  # two dash distances in meters


def undistort_and_warp(img: np.ndarray) -> np.ndarray:
    img, _ = _cc.undistort(img, False)
    return _bev.warp(img)


def build_roi_mask() -> np.ndarray:
    h, w = 760, 300  # warped.shape[:2]
    roi = [
        [0, 0], [w, 0],
        [w, 630], [230, h],
        [70, h], [0, 630]
    ]
    roi_mask = np.zeros(shape=(h, w), dtype=np.uint8)
    return cv2.fillPoly(roi_mask, [np.array(roi)], 255, cv2.LINE_4)


def luminance_constancy_lab(lab: np.ndarray, kernel_size: int=127) -> np.ndarray:
    lightness = lab[..., 0]
    blurred = cv2.GaussianBlur(lightness, (kernel_size, kernel_size), 0)
    adjusted = lightness / (blurred + 0.001)
    vmin, vmax = adjusted.min(), adjusted.max()
    adjusted = (adjusted - vmin) / (vmax - vmin)
    adjusted = np.clip(adjusted * 255, 0, 255).astype(np.uint8)
    return np.stack([adjusted, lab[..., 1], lab[..., 2]], axis=2)
