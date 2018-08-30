import logging
import numpy as np
from typing import Optional, List, Tuple, Callable, Iterable

from pipeline import curvature_radius, CURVATURE_INVALID
from .Lanes import Lanes
from .types import TrackType, Track, Fit, SeedFilter, Rects, SeedHistogram, Rect
from .rendering import render_rects, render_lane, LaneColor
from .seeds import build_histogram, find_maxima
from .LaneDetectionParams import LaneDetectionParams


log = logging.getLogger(__name__)


class LaneDetection:
    def __init__(self, params: LaneDetectionParams):
        self._params = params  # type: LaneDetectionParams
        self._lanes = Lanes(max_age=params.lane_max_age, max_history=params.lane_max_history)

    @property
    def lanes(self):
        return self._lanes

    def detect_and_render_lanes(self, img: np.ndarray, edges: np.ndarray):
        lanes = self._lanes

        log.info('Track ages: {}, {}'.format(lanes.left.age, lanes.right.age))

        # Obtain the histogram for seed position detection.
        histogram = self._build_histogram(edges)

        # Search the left line
        left_track = self._find_track(TrackType.Left, self._is_left, edges, histogram)
        print(left_track)

        # Search the right line
        right_track = self._find_track(TrackType.Right, self._is_right, edges, histogram)
        print(right_track)

        if self._params.render_boxes:
            render_rects(img, [r for (r, v) in zip(left_track.rects, left_track.rects_valid) if not v], .25)
            render_rects(img, [r for (r, v) in zip(right_track.rects, right_track.rects_valid) if not v], .25)
            render_rects(img, [r for (r, v) in zip(left_track.rects, left_track.rects_valid) if v], 1.0)
            render_rects(img, [r for (r, v) in zip(right_track.rects, right_track.rects_valid) if v], 1.0)

        if self._params.render_lanes:
            render_lane(img, left_track, LaneColor.Valid)
            render_lane(img, right_track, LaneColor.Valid)

        if left_track.valid:
            lanes.left.append(left_track)
        if right_track.valid:
            lanes.right.append(right_track)

    def _use_historical_track(self, img: np.ndarray, edges: np.ndarray, track: Optional[Track]):
        if track is None:
            return None

        rects = self._validate_fit(edges, track.fit)
        if rects is None:
            return None

        # TODO: Evaluate track from history
        return None

    def _find_track(self, side: TrackType, sieve: SeedFilter, edges: np.ndarray, histogram: SeedHistogram):
        histogram = self._filter_maxima(histogram, sieve)
        if len(histogram.pos) == 0:
            log.warning('No seed generated for {}.'.format(side))
            return self._invalid_track(side)
        log.debug('Seed for {} at x={}.'.format(side, histogram.pos[0]))
        rects, xs, ys, rects_valid = self._search_line(edges, histogram.pos[0])
        return self._create_track(side, rects, xs, ys, self._params.mx, rects_valid=rects_valid)

    def _validate_fit(self, edges: np.ndarray, fit: Fit) -> Optional[Rects]:
        assert fit is not None
        params = self._params
        lo, hi = params.validation_px_lo, params.validation_px_hi
        h, w = edges.shape[:2]

        # We now evaluate the interpolated track in order to check for support in the image.
        box_hwidth = params.search_box_width // 2
        top_y = h // 2
        ys = np.linspace(h - 1, top_y, (h - top_y) / params.search_box_height)
        xs = np.polyval(fit, ys)

        # We regenerate rectangles along the line fit as re-using old rectangles
        # could result in previously filled rects not filled and previously missing
        # rects now being valid.
        supported, rects = [], []
        for yb, x in zip(ys, xs):
            xl = int(max(0, x - box_hwidth))
            xr = int(min(w - 1, x + box_hwidth))
            yb = int(yb)
            yt = int(max(0, yb - params.search_box_height))
            window = edges[yt:yb, xl:xr]
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
                xr = int(min(w - 1, xl + params.search_box_width))

            local_support = 1. if lo <= support <= hi else 0
            supported.append(local_support)
            if local_support:
                rects.append((xl, yt, xr, yb))

        support = float(np.mean(supported))
        if support < params.validation_min_box_support:
            return None

        assert len(rects) > 0
        return rects

    @staticmethod
    def _is_left(m):
        return (60 < m) & (m < 140)

    @staticmethod
    def _is_right(m):
        return (160 < m) & (m < 240)

    def _build_histogram(self, edges: np.ndarray) -> SeedHistogram:
        params = self._params
        h, w = edges.shape[:2]
        top = int(h * (1. - params.seed_height))
        window = edges[top:, ...]
        hist, bins = build_histogram(window, 2, 1)
        maxima, values = find_maxima(hist, bins, k=len(bins))
        return SeedHistogram(pos=maxima, val=values)

    @staticmethod
    def _filter_maxima(histogram: SeedHistogram,
                       predicate: Callable[[Iterable[int]], List[int]], k: int = 1) -> SeedHistogram:
        maxima, values = histogram.pos, histogram.val
        if len(maxima) == 0:
            return SeedHistogram(pos=[], val=[])

        maxima = np.array(maxima)
        values = np.array(values)

        candidates = predicate(maxima)
        if not any(candidates):
            return SeedHistogram(pos=[], val=[])
        maxima = maxima[candidates][:k].tolist()
        values = values[candidates][:k].tolist()
        return SeedHistogram(pos=maxima, val=values)

    def _search_line(self, edges: np.ndarray, seed_x: int) -> Tuple[Rects, List[int], List[int], List[bool]]:
        params = self._params
        threshold = params.search_px_lo

        strikes = 0
        rects, xs, ys = [], [], []
        rect_valid = []
        h, w = edges.shape[:2]
        max_height = h - h * params.search_height
        next_x, next_y = seed_x, h

        while True:
            seed_x, seed_y = next_x, next_y
            window, rect = self._get_window(edges, next_x, next_y)
            hits = np.sum(window > 0)

            # The top coordinate of the current box will be the next seed Y coordinate.
            next_y = rect[1]

            # Register the current information
            rects.append(rect)
            xs.append(seed_x)
            ys.append(seed_y)

            # Obtain the centroid for the next window
            col_sums = np.squeeze(window.sum(axis=0))

            area = np.prod(window.shape[:2])
            if area == 0:
                log.warning('Search window has zero area at {}, {}.'.format(seed_x, seed_y))
                break
            total = hits / area
            if total < threshold:
                strikes += 1
                rect_valid.append(False)
                log.debug('Strike {} at {}, {}: {} < {}.'.format(strikes, seed_x, seed_y, total, threshold))
            else:
                strikes = 0
                rect_valid.append(True)

                # Obtain the next search position by finding the horizontal location of
                # the peak in local column-wise intensity (i.e. where most pixels are).
                indexes = np.arange(0, col_sums.shape[0])
                next_x = rect[0] + np.int32(np.average(indexes, weights=col_sums))
                next_x = self._refine_local_fit(next_x, rect[1], xs, ys)

            # Apply the search limit.
            if max_height > 0 and seed_y < max_height:
                break

            # Don't attempt to find a line forever.
            if strikes >= params.search_max_strikes:
                log.debug('Terminating search; max strikes reached.')
                break

            # Stop if we leave the window
            if rect[0] < 0 or rect[0] >= w or rect[1] < -params.search_box_height:
                log.debug('Terminating search; window boundaries violated.')
                break

            # Stop if we are close the the edges of the window
            edge = (rect[2] - rect[0]) // 4
            center_x = (rect[0] + rect[2]) // 2
            if center_x <= edge:
                log.debug('Terminating search; too close to left edge.')
                break
            elif center_x >= (w - edge):
                log.debug('Terminating search; too close to right edge.')
                break

            # Edge case: The prediction is invalid because there are not enough
            # hits; however, the search window is already close to the edge.
            # We could terminate the search if we are on the window edge already
            # and are below threshold.

        if strikes > 0:
            rects = rects[:-strikes]
            xs = xs[:-strikes]
            ys = ys[:-strikes]
            rect_valid = rect_valid[:-strikes]

        return rects, xs, ys, rect_valid

    def _refine_local_fit(self, next_x: int, next_y: int, xs: List[int], ys: List[int]):
        """
        Propose a new search location by fitting a curve over the last N search
        windows. The idea is that the streets always follow a clothoidal track,
        so sharp deviations from that are unlikely. Thus, a local curve fit
        is likely to point in the right direction. This corresponds to a very naive
        local model of the curvature of the street.
        :param next_x: The next X seed.
        :param next_y: The next Y seed.
        :param xs: The previous X coordinates.
        :param ys: The previous Y coordinates.
        :return: The updated fit.
        """
        alpha = max(0, min(1, self._params.search_local_fit))

        if alpha == 0:
            return next_x

        # For a polynomial of order 2 we need at least three points.
        n_smooth = self._params.search_local_fit_hist
        if len(ys) < max(3, n_smooth):
            return next_x

        n = min(len(ys), n_smooth)
        sxs = xs[-n:]
        sys = ys[-n:]
        fit = np.polyfit(sys, sxs, deg=2)

        estimated_x = fit[0] * (next_y ** 2) + fit[1] * next_y + fit[2]
        return int(alpha * estimated_x + (1 - alpha) * next_x)

    def _get_window(self, img: np.ndarray, seed_x: int, seed_y: int) -> Tuple[np.ndarray, Rect]:
        box_width = self._params.search_box_width
        box_height = self._params.search_box_height

        hbw = box_width // 2
        xl = seed_x - hbw
        xr = xl + box_width
        yb = seed_y
        yt = yb - box_height

        h, w = img.shape[:2]
        xl_ = max(0, xl)
        yt_ = max(0, yt)
        xr_ = min(w - 1, xr)
        yb_ = min(h - 1, yb)
        return img[yt_:yb_, xl_:xr_, ...], (xl, yt, xr, yb)

    @staticmethod
    def _invalid_track(side: TrackType):
        return Track(side=side, fit=(0, 0, 0), curvature_radius=CURVATURE_INVALID, confidence=0,
                     rects=[], valid=False, rects_valid=[])

    def _create_track(self, side: TrackType, rects: Rects,
                      xs: Optional[List[int]], ys: Optional[List[int]], mx: float,
                      rects_valid: List[bool]) -> Track:
        alpha = self._params.fit_quality_decay
        beta = self._params.fit_quality_allowed_deviation

        min_boxes = self._params.boxes_thresh
        if len(rects) < min_boxes and (len(xs) < min_boxes or len(ys) < min_boxes):
            return self._invalid_track(side)

        if xs is None or len(xs) == 0:
            xs = [(r[2] + r[0]) // 2 for r in rects]
        if ys is None or len(xs) == 0:
            ys = [r[1] for r in rects]

        # Fit a polynomial and determine quality by determining how much the rectangles deviate from the fit.
        fit = np.polyfit(ys, xs, deg=2)
        xs_ = np.polyval(fit, ys)
        rmse = np.mean(np.array((xs - xs_) ** 2))
        confidence = min(1, np.exp(-alpha * (rmse - beta)))

        # Measure the curvature_radius close to the vehicle (at the bottom of the image)
        y_eval = np.max(ys)
        cr = curvature_radius(fit, y_eval, mx)

        return Track(side=side, fit=fit, rects=rects, curvature_radius=cr, valid=True, confidence=confidence,
                     rects_valid=rects_valid)
