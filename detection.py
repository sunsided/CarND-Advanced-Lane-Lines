"""
Runs the actual lane line detection on the specified video.
"""

import argparse
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, List, Union, Optional, NamedTuple, Any

from pipeline import CameraCalibration, BirdsEyeView, ImageSection, Point
from pipeline import EdgeDetectionNaive, EdgeDetectionTemporal, EdgeDetectionConf, EdgeDetectionSWT
from pipeline import LaneColorMasking


Track = NamedTuple('Track', [('side', int), ('valid', bool),
                             ('curvature_radius', float),
                             ('fit', Tuple[np.ndarray, Any, np.ndarray]),
                             ('rects', list)])

InvalidLeftTrack = Track(side=-1, valid=False, curvature_radius=float('inf'),
                         fit=tuple(np.zeros((3,), np.float32)),
                         rects=[])

InvalidRightTrack = Track(side=1, valid=False, curvature_radius=float('inf'),
                          fit=tuple(np.zeros((3, ), np.float32)),
                          rects=[])


def get_mask(frame: np.ndarray, edg: Optional[Union[EdgeDetectionConf, EdgeDetectionNaive]],
             tmp: Optional[Union[EdgeDetectionTemporal, EdgeDetectionSWT]],
             lcm: Optional[LaneColorMasking]) -> np.ndarray:
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
    return scaled ** 1


def build_histogram(window: np.ndarray, binwidth: int = 20, binstep: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    half_binwidth = binwidth // 2
    bins = np.uint32(np.arange(half_binwidth, window.shape[1] - half_binwidth, binstep))
    histogram = np.zeros(shape=len(bins), dtype=np.float32)
    for i, x in enumerate(bins):
        left = max(x - half_binwidth, 0)
        right = min(x + half_binwidth, window.shape[1])
        bin_area = window.shape[0] * (right - left)
        histogram[i] = np.sum(window[:, left:right]) / bin_area
    return histogram, bins


def find_maxima(histogram: np.ndarray, bins: np.ndarray, epsilon: float = .0, k: int = 2) -> Tuple[
    List[int], List[int]]:
    maxima, heights = [], []
    if histogram[0] > (histogram[1] + epsilon):
        maxima.append(bins[0])
        heights.append(histogram[0])
    for i in range(1, histogram.shape[0] - 1):
        left_smaller = histogram[i] > (histogram[i - 1] + epsilon)
        right_smaller = histogram[i] > (histogram[i + 1] + epsilon)
        if left_smaller and right_smaller:
            maxima.append(bins[i])
            heights.append(histogram[i])
    if histogram[-1] > (histogram[-2] + epsilon):
        maxima.append(bins[-1])
        heights.append(histogram[-1])

    if k == 0:
        k = len(heights)
    idxs = np.array(heights).argsort()[-k:][::-1]
    maxima = np.array(maxima)[idxs].tolist()
    heights = np.array(heights)[idxs].tolist()
    return maxima, heights


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

            # Now for the regression, we are actually using the corrected location rather than
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
                  mx: float=1., my: float=1.):
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
        track, xs, ys = search_track(mask, m, h - 1,
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
        fit = np.polyfit(ys, xs, deg=degree)

        # Measure the curvature_radius close to the vehicle (at the bottom of the image)
        y_eval = np.max(ys)
        coeff_a, coeff_b, coeff_c = fit
        curvature_radius = ((1. + (2. * coeff_a * y_eval + coeff_b) ** 2.) ** 1.5) / np.abs(2. * coeff_a)
        curvature_radius *= mx

        t = Track(side=side, fit=fit, rects=track, curvature_radius=curvature_radius, valid=True)
        if side > 0:
            print(curvature_radius)
        tracks.append(t)
    return tracks


def blend_fits(tracks: List[Track]) -> Optional[Tuple[np.ndarray, Any, np.ndarray]]:
    assert len(tracks) > 0

    valid_tracks = [t for t in tracks if t.valid]
    if len(valid_tracks) == 0:
        return None

    curvature = np.mean([t.curvature_radius for t in valid_tracks])
    sqe = np.array([(1 - t.curvature_radius / curvature) ** 2 for t in valid_tracks])
    error_coeffs = np.exp(-sqe)
    mask = np.ones_like(error_coeffs)
    mask[error_coeffs < 0.7] = 0
    if np.sum(mask) == 0:
        return None
    norm = np.sum(error_coeffs * mask)

    fits = np.array([t.fit for t in valid_tracks])
    for i in range(len(fits)):
        fits[i] *= error_coeffs[i] * mask[i]

    return tuple(np.sum(fits, axis=0) / norm)


def render_lane(img: np.ndarray, tracks: List[Track],
                thresh: int = 50, filter: bool=True) -> np.ndarray:
    assert tracks is not None
    h, w = img.shape[:2]

    # We pick the latest track for rendering the rectangles.
    track = tracks[-1]
    assert track is not None

    # The rendering of the lane line estimate will extend up to either
    # the "highest" position of a matched box or half the image size.
    highest_rect = h // 2
    if track.valid:
        highest_rect = track.rects[-1][1]
        ys = np.linspace(h - 1, highest_rect, h - highest_rect)
        xs = np.polyval(track.fit, ys)
        if xs[0] < thresh or xs[0] > (w - thresh):
            for rect in track.rects:
                cv2.rectangle(img, rect[:2], rect[2:], color=(0, 0, .25), thickness=1)
            return img
        for rect in track.rects:
            cv2.rectangle(img, rect[:2], rect[2:], color=(0, 0, 1), thickness=1)

    # Select the topmost position for lane line extrapolation.
    top_y = min(h // 2, highest_rect)
    ys = np.linspace(h - 1, top_y, h - top_y)

    # For the polylines, we add filtering across previous detections.
    if filter:
        fit = blend_fits(tracks)
        if fit is None:
            # TODO: Start search from scratch!
            return img
        xs = np.polyval(fit, ys)
    elif not track.valid:
        return img

    # noinspection PyUnboundLocalVariable
    pts = np.int32([(x, y) for (x, y) in zip(xs, ys)])
    cv2.polylines(img, [pts], False, color=(0, 1, 1), thickness=2, lineType=cv2.LINE_AA)
    return img


def render_lanes(img: np.ndarray, left_tracks: List[Track], right_tracks: List[Track],
                 left_thresh: int = 50, right_thresh: int = 50, filter: bool=True) -> np.ndarray:
    assert left_tracks is not None
    assert right_tracks is not None
    render_lane(img, left_tracks, left_thresh, filter=filter)
    render_lane(img, right_tracks, right_thresh, filter=filter)
    return img


def main(args):
    cap = cv2.VideoCapture(args.file)
    if not cap:
        print('Failed reading video file.')
        return
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_scale = 1024 * 768 / (width * height)

    cc = CameraCalibration.from_pickle('calibration.pkl')

    section = ImageSection(
        top_left=Point(x=580, y=461.75),
        top_right=Point(x=702, y=461.75),
        bottom_right=Point(x=1013, y=660),
        bottom_left=Point(x=290, y=660),
    )

    def build_roi_mask(pad: int = 0) -> np.ndarray:
        h, w = 760, 300  # warped.shape[:2]
        roi = [
            [0, 0],
            [w, 0],
            [w, 610 - pad],
            [230 - pad, h],
            [60 + pad, h],
            [0, 610 - pad]
        ]
        roi_mask = np.zeros(shape=(h, w), dtype=np.uint8)
        roi_mask = cv2.fillPoly(roi_mask, [np.array(roi)], 255, lineType=cv2.LINE_4)
        return np.float32(roi_mask) / 255

    bev = BirdsEyeView(section,
                       section_width=3.6576,  # one lane width in meters
                       section_height=2 * 13.8826)  # two dash distances in meters

    mx = bev.units_per_pixel_x
    my = bev.units_per_pixel_y

    roi_mask = build_roi_mask()
    roi_mask_hard = build_roi_mask(10)

    edg = EdgeDetectionNaive(detect_lines=False, mask=roi_mask)
    swt = EdgeDetectionSWT(mask=roi_mask, max_length=8)
    edt = EdgeDetectionTemporal(mask=roi_mask, detect_lines=False)

    lcm = LaneColorMasking(luminance_kernel_width=33)
    lcm.detect_lines = False
    lcm.blue_threshold = 250
    lcm.light_cutoff = .95

    tracks_left = []
    tracks_right = []

    while True:
        t_start = datetime.now()
        ret, img = cap.read()
        if not ret:
            break
        print('Processing frame {} ...'.format(cap.get(cv2.CAP_PROP_POS_FRAMES)))

        img, _ = cc.undistort(img, False)
        warped = bev.warp(img)

        warped_f = np.float32(warped) / 255
        lab = cv2.cvtColor(warped_f, cv2.COLOR_BGR2LAB)
        yellows = lab[..., 2] / 127
        yellows[yellows < 0.5] = 0
        cv2.normalize(yellows, yellows, 1, norm_type=cv2.NORM_MINMAX)
        gray = cv2.max(lab[..., 0] / 100, yellows)
        lab[..., 0] = gray * 100

        warped_f = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        warped = np.uint8(warped_f * 255)

        cv2.imshow('warped_f', warped_f)

        edges = get_mask(warped, edg, swt, lcm) * roi_mask_hard
        canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        tracks = regress_lanes(edges,
                               k=2, degree=2,
                               search_height=10,
                               max_height=0.55,
                               max_strikes=15,
                               box_width=30, box_height=10, threshold=5,
                               fit_weight=2, centroid_weight=1, n_smooth=10,
                               mx=mx, my=my)

        detection_left = [t for t in tracks if t.side < 0]
        detection_right = [t for t in tracks if t.side > 0]

        tracks_left.append(detection_left[0] if len(detection_left) > 0 else InvalidLeftTrack)
        tracks_right.append(detection_right[0] if len(detection_right) > 0 else InvalidRightTrack)

        max_history = 10
        if len(tracks_left) > max_history:
            tracks_left.pop(0)
        if len(tracks_right) > max_history:
            tracks_right.pop(0)
        render_lanes(canvas, tracks_left, tracks_right, filter=True)

        # img = cv2.resize(img, (0, 0), fx=display_scale, fy=display_scale)
        cv2.imshow('video', canvas)

        # Attempt to stay close to the original FPS.
        t_end = datetime.now()
        t_delta = (t_end - t_start).total_seconds() * 1000
        t_wait = int(max(1, fps - t_delta))

        if cv2.waitKey(t_wait) == 27:
            break


def parse_args():
    parser = argparse.ArgumentParser()
    v = parser.add_argument_group('Video')
    v.add_argument(metavar='VIDEO', dest='file', default='project_video.mp4',
                   help='The video file to process.')
    args = parser.parse_args()
    if not os.path.exists(args.file):
        parser.error('The specified video {} could not be found.'.format(args.file))

    return args


if __name__ == '__main__':
    main(parse_args())
