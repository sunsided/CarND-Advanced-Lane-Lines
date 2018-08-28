from typing import Optional, List, Tuple

import numpy as np
from .types import Fit, Track


def get_points(fit: Fit, y_lo: float, y_hi: float) -> np.ndarray:
    """
    Evaluates a fit in the specified Y value range.
    :param fit: The fit to evaluate.
    :param y_lo: The lower Y coordinate.
    :param y_hi: The higher Y coordinate.
    :return: The array of (X,Y) coordinates.
    """
    ys = np.linspace(y_lo, y_hi, int(abs(y_hi - y_lo) + 1))
    xs = np.polyval(fit, ys)
    return np.array([[x, y] for (x, y) in zip(xs, ys)], np.float32)


def blend_tracks(tracks: List[Track]) -> Optional[Fit]:
    assert len(tracks) > 0

    valid_tracks = [t for t in tracks if t.valid]
    if len(valid_tracks) == 0:
        return None

    curvature = np.mean([t.curvature_radius for t in valid_tracks])
    confidences = np.array([t.confidence for t in valid_tracks])
    sqe = np.array([(1 - t.curvature_radius / curvature) ** 2 for t in valid_tracks])
    error_coeffs = np.exp(-sqe)
    age_coeffs = np.linspace(.4, 1., len(error_coeffs)) ** 2
    mask = np.ones_like(error_coeffs)
    mask[error_coeffs < 0.6] = 0
    norm = np.sum(error_coeffs * mask * confidences * age_coeffs)
    if norm == 0:
        return None

    fits = np.array([t.fit for t in valid_tracks])
    for i in range(len(fits)):
        fits[i] *= error_coeffs[i] * mask[i] * confidences[i] * age_coeffs[i]

    return tuple(np.sum(fits, axis=0) / norm)


def recenter_rects(track: Track, rects: List[Tuple[int, int, int, int]]) -> Track:
    return Track(side=track.side, valid=track.valid, fit=track.fit,
                 confidence=track.confidence, rects=rects,
                 curvature_radius=track.curvature_radius)
