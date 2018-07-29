import cv2
import numpy as np


def luminance_constancy_lab(lab: np.ndarray, kernel_size: int=127) -> np.ndarray:
    lightness = lab[..., 0]
    blurred = cv2.GaussianBlur(lightness, (kernel_size, kernel_size), 0)
    adjusted = lightness / (blurred + 0.001)
    vmin, vmax = adjusted.min(), adjusted.max()
    adjusted = (adjusted - vmin) / (vmax - vmin)
    adjusted = np.clip(adjusted * 255, 0, 255).astype(np.uint8)
    return np.stack([adjusted, lab[..., 1], lab[..., 2]], axis=2)
