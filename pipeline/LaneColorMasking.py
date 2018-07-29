import cv2
import numpy as np
from typing import Tuple


class LaneColorMasking:
    """
    Obtains masks for yellow and white lane markings from a color image.
    """
    def __init__(self, light_cutoff: float=.92, blue_threshold: int=30, luminance_kernel_width: int=127):
        self._light_cutoff = light_cutoff
        self._blue_threshold = blue_threshold
        self._lc_kernel_size = luminance_kernel_width

    def process(self, img: np.ndarray, is_lab: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes the specified image.
        :param img: The image to obtain masks from.
        :param is_lab: If False, the image is assumed to be BGR and will be converted to L*a*b*;
                       if True, the image is assumed to be L*a*b* already.
        :return: A tuple consisting of the white and yellow lane mask.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) if not is_lab else img
        return self._lab_mask2(lab)

    def _lab_mask2(self, lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Obtains the masks.
        :param lab: The L*a*b* image.
        :return: A tuple consisting of the white and yellow lane mask.
        """
        li = self._luminance_constancy_lab(lab[..., 0])
        li = cv2.equalizeHist(li)
        b = cv2.equalizeHist(lab[..., 2])

        l_mask = np.zeros(shape=lab.shape[:2], dtype=np.uint8)
        l_mask[li >= self._light_cutoff * li.max()] = 255
        b_mask = np.zeros_like(l_mask)
        b_mask[b <= self._blue_threshold] = 255
        return l_mask, b_mask

    def _luminance_constancy_lab(self, channel: np.ndarray) -> np.ndarray:
        """
        Performs local luminance equalization.
        :param channel: The channel to operate on.
        :return: The adjusted channel.
        """
        blurred = cv2.GaussianBlur(channel, (self._lc_kernel_size, self._lc_kernel_size), 0)
        adjusted = channel / (blurred + 0.001)
        vmin, vmax = adjusted.min(), adjusted.max()
        adjusted = (adjusted - vmin) / (vmax - vmin)
        return np.clip(adjusted * 255, 0, 255).astype(np.uint8)
