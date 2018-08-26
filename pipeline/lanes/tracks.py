from typing import Optional, List, Tuple, NamedTuple, Any

import numpy as np

from pipeline import curvature_radius
from pipeline.lanes import build_histogram, find_maxima

Fit = Tuple[np.ndarray, Any, np.ndarray]

Track = NamedTuple('Track', [('side', int), ('valid', bool),
                             ('curvature_radius', float),
                             ('confidence', float),
                             ('fit', Fit),
                             ('rects', list)])

InvalidLeftTrack = Track(side=-1, valid=False, curvature_radius=float('inf'),
                         confidence=0,
                         fit=tuple(np.zeros((3,), np.float32)),
                         rects=[])

InvalidRightTrack = Track(side=1, valid=False, curvature_radius=float('inf'),
                          confidence=0,
                          fit=tuple(np.zeros((3, ), np.float32)),
                          rects=[])


def get_window(mask: np.ndarray, seed_x: int, seed_y: int, box_width: int = 50, box_height: int = 50):
    hbw = box_width // 2
    xl = seed_x - hbw
    xr = xl + box_width
    yb = seed_y
    yt = yb - box_height

    h, w = mask.shape[:2]
    xl_ = max(0, xl)
    yt_ = max(0, yt)
    xr_ = min(w - 1, xr)
    yb_ = min(h - 1, yb)
    return mask[yt_:yb_, xl_:xr_, ...], (xl, yt, xr, yb)


def search_track(mask: np.ndarray, seed_x: int, seed_y: int,
                 box_width: int = 50, box_height: int = 50, threshold: float = 20,
                 fit_weight: float = 1., centroid_weight: float = 1.,
                 n_smooth: int = 10, min_n_smooth: int = 5, max_strikes=4, max_height: float = 0):
    rects, xs, ys = [], [], []
    h, w = mask.shape[:2]
    max_height = h - h * max_height
    strikes = 0
    while True:
        window, rect = get_window(mask, seed_x, seed_y, box_width, box_height)

        # Obtain the centroid for the next window
        col_sums = np.squeeze(window.sum(axis=0))
        total = col_sums.sum()
        valid = total > threshold
        if valid:
            strikes = 0
            rects.append(rect)
            # Propose a new search location by fitting a curve over the last N search
            # windows. The idea is that the streets always follow a clothoidal track,
            # so sharp deviations from that are unlikely. Thus, a local curve fit
            # is likely to point in the right direction. This corresponds to a very naive
            # local model of the curvature of the street.
            indexes = np.arange(0, col_sums.shape[0])
            seed_x = rect[0] + np.int32(np.average(indexes, weights=col_sums))

            # For the regression, we are actually using the corrected location rather than
            # the previous estimate.
            xs.append(seed_x)
            ys.append(seed_y)

            # For a curve of degree two, we need at least three samples.
            if len(rects) > max(3, min_n_smooth):
                n = min(len(rects), n_smooth)
                sxs = xs[-n:]
                sys = ys[-n:]
                sfit = np.polyfit(sys, sxs, deg=2)

                sy = rect[1]
                sseed_x = sfit[0] * sy ** 2 + sfit[1] * sy + sfit[2]
                seed_x = int((fit_weight * sseed_x + centroid_weight * seed_x) / (fit_weight + centroid_weight))
        else:
            strikes += 1

        # Apply the search limit.
        if max_height > 0 and seed_y < max_height:
            break

        # Don't attempt to find a line forever.
        if strikes == max_strikes:
            break

        # Stop if we leave the window
        if rect[0] < 0 or rect[0] >= w or rect[1] < -box_height:
            break

        # Stop if we are close the the edges of the window
        edge = (rect[2] - rect[0]) // 4
        center_x = (rect[0] + rect[2]) // 2
        if center_x <= edge:
            break
        elif center_x >= (w - edge):
            break

        # Edge case: The prediction is invalid because there are not enough
        # hits; however, the search window is already close to the edge.
        # We could terminate the search if we are on the window edge already
        # and are below threshold.

        # Update the seeds by moving up one window
        seed_y = rect[1]
    return rects, xs, ys


def regress_lanes(mask: np.ndarray, k: int = 2,
                  box_width: int = 75, box_height: int = 40, threshold: Optional[int] = None,
                  degree: int = 2, search_height: int = 4,
                  fit_weight: float = 1., centroid_weight: float = 1.,
                  n_smooth: int = 10, max_strikes=4, max_height: float = 0,
                  simple_check: bool = True,
                  detect_left: bool = True,
                  detect_right: bool = True,
                  mx: float = 1., my: float = 1.) -> List[Track]:
    h, w = mask.shape[:2]
    window = mask[-h // search_height:, ...]
    hist, bins = build_histogram(window, 2, 1)
    maxima, values = find_maxima(hist, bins, k=len(bins))
    if len(maxima) == 0:
        return []

    def is_left(m) -> bool:
        return (60 < m) & (m < 140)

    def is_right(m) -> bool:
        return (160 < m) & (m < 240)

    if simple_check:
        maxima = np.array(maxima)
        left = np.argmax(is_left(maxima)) if detect_left else None
        right = np.argmax(is_right(maxima)) if detect_right else None

        if left is not None and right is not None:
            good_maxima = [maxima[int(left)], maxima[int(right)]]
            good_values = [values[int(left)], values[int(right)]]
        elif left is not None:
            good_maxima = [maxima[int(left)]]
            good_values = [values[int(left)]]
        elif right is not None:
            good_maxima = [maxima[int(right)]]
            good_values = [values[int(right)]]
        else:
            return []
    else:
        # We now ensure to actually select distinct starting points by
        # suppressing all maxima that are within a window of the previous best one.
        good_maxima = [maxima[0]]
        good_values = [values[0]]

    # Early exit if we have found all requested maxima.
    if len(good_maxima) < k:
        threshold = threshold or (box_width * 0.5)
        for i in range(1, len(maxima)):
            mi = maxima[i]
            selected = True
            for j in range(0, len(good_maxima)):
                mj = maxima[j]
                if abs(mj - mi) < threshold:
                    selected = False
                    break
            if not selected:
                continue
            good_maxima.append(maxima[i])
            good_values.append(values[i])
            # Early exit if we have found all requested maxima.
            if len(good_maxima) >= k:
                break

    tracks = []
    for m in good_maxima:
        rects, xs, ys = search_track(mask, m, h - 1,
                                     box_width=box_width,
                                     box_height=box_height,
                                     threshold=threshold,
                                     fit_weight=fit_weight, centroid_weight=centroid_weight,
                                     n_smooth=n_smooth,
                                     max_strikes=max_strikes,
                                     max_height=max_height)
        if len(xs) < 3:
            continue

        side = -1 if is_left(m) else (1 if is_right(m) else 0)
        if side == 0:
            continue
        tracks.append(create_track(side, rects, xs, ys, mx, degree))
    return tracks


def create_track(side: int, rects: List[Tuple[int, int, int, int]], xs: Optional[List[int]], ys: Optional[List[int]], mx: float, degree: int=2) -> Track:
    if xs is None or len(xs) == 0:
        xs = [(r[2] + r[0]) // 2 for r in rects]
    if ys is None or len(xs) == 0:
        ys = [r[1] for r in rects]

    fit = np.polyfit(ys, xs, deg=degree)
    xs_ = np.polyval(fit, ys)
    rmse = np.mean(np.array((xs - xs_) ** 2))
    confidence = max(0, 1 - np.exp(rmse - 5))

    # Measure the curvature_radius close to the vehicle (at the bottom of the image)
    y_eval = np.max(ys)
    cr = curvature_radius(fit, y_eval, mx)

    return Track(side=side, fit=fit, rects=rects, curvature_radius=cr, valid=True, confidence=confidence)


def blend_tracks(tracks: List[Track]) -> Optional[Fit]:
    assert len(tracks) > 0

    valid_tracks = [t for t in tracks if t.valid]
    if len(valid_tracks) == 0:
        return None

    curvature = np.mean([t.curvature_radius for t in valid_tracks])
    confidences = np.array([t.confidence for t in valid_tracks])
    sqe = np.array([(1 - t.curvature_radius / curvature) ** 2 for t in valid_tracks])
    error_coeffs = np.exp(-sqe)
    age_coeffs = np.linspace(.4, 1., len(error_coeffs))
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


def validate_fit(img: np.ndarray, fit: Optional[Fit], lo: float = .05, hi: float = .8,
                 box_width: int = 75, box_height: int = 40,
                 min_support: float = .25) -> Optional[List[Tuple[int, int, int, int]]]:
    if fit is None:
        return None
    h, w = img.shape[:2]

    # We now evaluate the interpolated track in order to check for support in the image.
    box_hwidth = box_width // 2
    top_y = h // 2
    ys = np.linspace(h - 1, top_y, (h - top_y) / box_height)
    xs = np.polyval(fit, ys)

    supported, rects = [], []
    for yb, x in zip(ys, xs):
        xl = int(max(0, x - box_hwidth))
        xr = int(min(w - 1, x + box_hwidth))
        yb = int(yb)
        yt = int(max(0, yb - box_height))
        window = img[yt:yb, xl:xr]
        area = np.prod(window.shape)
        if area == 0:
            break
        support = np.sqrt(np.sum(window) / np.prod(window.shape))

        # We now find the centroid of the window again and refine the X coordinate.
        if support > 0:
            col_sums = np.squeeze(window.sum(axis=0))
            if len(col_sums.shape) != 1:
                continue
            indexes = np.arange(0, col_sums.shape[0])
            x_centroid = np.int32(np.average(indexes, weights=col_sums))
            xl = xl + x_centroid - box_hwidth
            xr = int(min(w - 1, xl + box_width))

        rects.append((xl, yt, xr, yb))
        supported.append(1. if lo < support < hi else 0)

    support = float(np.mean(supported))
    if support < min_support:
        return None
    return rects
