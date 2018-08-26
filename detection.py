"""
Runs the actual lane line detection on the specified video.
"""

import argparse
import os
import cv2
import numpy as np
from datetime import datetime

from pipeline import ImageSection, detect_and_render_lanes, VALID_COLOR, CACHED_COLOR, WARNING_COLOR
from pipeline.preprocessing import detect_lane_pixels
from pipeline.transform import *
from pipeline.edges import *
from pipeline.lanes import *


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
        mask = np.zeros(shape=(h, w), dtype=np.uint8)
        mask = cv2.fillPoly(mask, [np.array(roi)], 255, lineType=cv2.LINE_4)
        return np.float32(mask) / 255

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

    state = LaneDetectionState()

    while True:
        t_start = datetime.now()
        ret, img = cap.read()
        if not ret:
            break
        print('Processing frame {} ...'.format(int(cap.get(cv2.CAP_PROP_POS_FRAMES))))

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
        warped = np.uint8(warped_f * 255)  # type: np.ndarray

        edges = detect_lane_pixels(warped, edg, swt, lcm) * roi_mask_hard
        canvas = warped_f.copy()

        results = detect_and_render_lanes(canvas, edges, state, mx, my, render_lanes=True, render_boxes=True)
        (left_valid, left), (right_valid, right) = results

        img = np.float32(img) / 255.
        img_alpha = img.copy()

        y_bottom, y_top = warped.shape[0], warped.shape[0] // 2
        if left is not None:
            left = get_points(left, y_bottom, y_top)
            left = np.floor(bev.unproject(left)).astype(np.int32)
        if right is not None:
            right = get_points(right, y_bottom, y_top)
            right = np.floor(bev.unproject(right)).astype(np.int32)

        # Only fill if we have valid tracks
        if (left is not None) or (right is not None):
            if (left is not None) and (right is not None):
                all = np.vstack([left, np.flipud(right)])
            elif left is not None:
                all = left
            else:
                all = right
            color = VALID_COLOR if (left_valid and right_valid) else \
                (CACHED_COLOR if left_valid or right_valid else WARNING_COLOR)
            alpha = 0.4 if left_valid and right_valid else 0.1
            cv2.fillPoly(img_alpha, [all], color, lineType=cv2.LINE_AA)
            img = cv2.addWeighted(img, (1 - alpha), img_alpha, alpha, 0)

        # Draw the lane lines
        if left is not None:
            color = VALID_COLOR if left_valid else CACHED_COLOR
            cv2.polylines(img, [left], False, color, 3, lineType=cv2.LINE_AA)
        if right is not None:
            color = VALID_COLOR if right_valid else CACHED_COLOR
            cv2.polylines(img, [right], False, color, 3, lineType=cv2.LINE_AA)

        img = cv2.resize(img, (0, 0), fx=display_scale, fy=display_scale)
        cv2.imshow('edges', edges)
        cv2.imshow('canvas', canvas)
        cv2.imshow('video', img)

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
