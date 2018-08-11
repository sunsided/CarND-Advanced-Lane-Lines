"""
Runs the actual lane line detection on the specified video.
"""

import argparse
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, List

from pipeline import CameraCalibration, BirdsEyeView, ImageSection, Point
from pipeline import EdgeDetectionTemporal, EdgeDetection, LaneColorMasking


def get_mask(frame: np.ndarray, edg: EdgeDetection, tmp: EdgeDetectionTemporal, lcm: LaneColorMasking) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mask_l, mask_b = lcm.process(lab, is_lab=True)
    edges_static = edg.filter(lab, is_lab=True)
    edges_temporal = tmp.filter(lab, is_lab=True)
    mask_sum = edges_static
    mask_sum + edges_temporal
    mask_sum += np.float32(mask_l) / 255.
    mask_sum += np.float32(mask_b) / 255.
    scaled = (mask_sum / 4) ** 2
    return scaled / scaled.max()


def build_histogram(window: np.ndarray, binwidth: int=20, binstep: int=20) -> Tuple[np.ndarray, np.ndarray]:
    half_binwidth = binwidth // 2
    bins = np.uint32(np.arange(half_binwidth, window.shape[1] - half_binwidth, binstep))
    histogram = np.zeros(shape=len(bins), dtype=np.float32)
    for i, x in enumerate(bins):
        left = max(x - half_binwidth, 0)
        right = min(x + half_binwidth, window.shape[1])
        bin_area = window.shape[0] * (right - left)
        histogram[i] = np.sum(window[:, left:right]) / bin_area
    return histogram, bins


def find_maxima(histogram: np.ndarray, bins: np.ndarray, epsilon: float = .0, k: int = 2) -> Tuple[List[int], List[int]]:
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

    if k > 0:
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
                 n_smooth: int=10):
    rects, xs, ys = [], [], []
    h, w = mask.shape[:2]
    while True:
        window, rect = get_window(mask, seed_x, seed_y, box_width, box_height)

        # Obtain the centroid for the next window
        col_sums = np.squeeze(window.sum(axis=0))
        total = col_sums.sum()
        valid = total > threshold
        if valid:
            rects.append(rect)
            xs.append(seed_x)
            ys.append(seed_y)
            # Propose a new search location by fitting a curve over the last N search
            # windows. The idea is that the streets always follow a clothoidal track,
            # so sharp deviations from that are unlikely. Thus, a local curve fit
            # is likely to point in the right direction. This corresponds to a very naive
            # local model of the curvature of the street.
            indexes = np.arange(0, col_sums.shape[0])
            seed_x = rect[0] + np.int32(np.average(indexes, weights=col_sums))

            # For a curve of degree two, we need at least three samples.
            if len(rects) > 3:
                n = min(len(rects), n_smooth)
                sxs = xs[-n:]
                sys = ys[-n:]
                sfit = np.polyfit(sys, sxs, deg=2)

                sy = rect[1]
                sseed_x = sfit[0] * sy ** 2 + sfit[1] * sy + sfit[2]
                seed_x = int((fit_weight * sseed_x + centroid_weight * seed_x) / (fit_weight + centroid_weight))

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
                  box_width: int = 75, box_height: int = 40, threshold: int = 20,
                  degree: int = 2, search_height: int = 4,
                  fit_weight: float = 1., centroid_weight: float = 1.,
                  n_smooth: int=10):
    h, w = mask.shape[:2]
    window = mask[-h // search_height:, ...]
    hist, bins = build_histogram(window, 2, 1)
    maxima, values = find_maxima(hist, bins, k=k)

    fits, rects = [], []
    for m in maxima:
        track, xs, ys = search_track(mask, m, h - 1,
                                     box_width=box_width,
                                     box_height=box_height,
                                     threshold=threshold,
                                     fit_weight=fit_weight, centroid_weight=centroid_weight,
                                     n_smooth=n_smooth)
        if len(xs) == 0:
            continue
        fits.append(np.polyfit(ys, xs, deg=degree))
        rects.append(track)
    return fits, rects


def render_lanes(img: np.ndarray, fits, rects, left_thresh: int=50, right_thresh: int=50) -> np.ndarray:
    assert rects is not None
    h, w = img.shape[:2]

    for track in rects:
        for rect in track:
            cv2.rectangle(img, rect[:2], rect[2:], color=(0, 0, .25), thickness=1)

    for fit, track in zip(fits, rects):
        top_y = track[-1][1]
        ys = np.linspace(h - 1, top_y, h - top_y)
        xs = np.polyval(fit, ys)
        if xs[0] < left_thresh or xs[0] > (w - right_thresh):
            continue

        for rect in track:
            cv2.rectangle(img, rect[:2], rect[2:], color=(0, 0, 1), thickness=1)

        pts = np.int32([(x, y) for (x, y) in zip(xs, ys)])
        cv2.polylines(img, [pts], False, color=(0, 1, 1), thickness=2, lineType=cv2.LINE_AA)

    return img


def main(args):
    cap = cv2.VideoCapture(args.file)
    if not cap:
        print('Failed reading video file.')
        return
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_scale = 1024*768 / (width*height)

    cc = CameraCalibration.from_pickle('calibration.pkl')

    section = ImageSection(
        top_left=Point(x=580, y=461.75),
        top_right=Point(x=702, y=461.75),
        bottom_right=Point(x=1013, y=660),
        bottom_left=Point(x=290, y=660),
    )

    def build_roi_mask() -> np.ndarray:
        h, w = 760, 300  # warped.shape[:2]
        roi = [
            [0, 0],
            [w, 0],
            [w, 620],
            [240, h],
            [60, h],
            [0, 620]
        ]
        roi_mask = np.zeros(shape=(h, w), dtype=np.uint8)
        return cv2.fillPoly(roi_mask, [np.array(roi)], 255, lineType=cv2.LINE_4)

    bev = BirdsEyeView(section,
                       section_width=3.6576,  # one lane width in meters
                       section_height=2 * 13.8826)  # two dash distances in meters

    roi_mask = build_roi_mask()
    roi_mask_f = np.float32(roi_mask) / 255.

    edg = EdgeDetection(detect_lines=True, mask=roi_mask)
    edt = EdgeDetectionTemporal(mask=roi_mask)

    lcm = LaneColorMasking()
    lcm.detect_lines = False
    lcm.blue_threshold = 25
    lcm.light_cutoff = .9

    while True:
        t_start = datetime.now()
        ret, img = cap.read()
        if not ret:
            break

        img, _ = cc.undistort(img, False)
        warped = bev.warp(img)

        #edg.detect_lines = True
        #edges_static = edg.detect(warped, is_lab=False)
        #edges_temporal = edt.filter(warped, is_lab=False)
        #edges = edges_static * edges_temporal
        #img = np.hstack([np.float32(warped) / 255., cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)])
        #cv2.imshow('edges', img)
        #cv2.waitKey(33)
        #continue

        edges = get_mask(warped, edg, edt, lcm) * roi_mask_f
        canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        fits, rects = regress_lanes(edges, k=2, search_height=5, box_width=50, box_height=30, threshold=5,
                                    fit_weight=1, centroid_weight=1, n_smooth=20)
        render_lanes(canvas, fits, rects)

        #img = cv2.resize(img, (0, 0), fx=display_scale, fy=display_scale)
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
