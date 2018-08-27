"""
Runs the actual lane line detection on the specified video.
"""

import argparse
import os
import cv2
import numpy as np
from datetime import datetime

from pipeline import ImageSection, detect_and_render_lanes, VALID_COLOR, CACHED_COLOR, WARNING_COLOR, curvature_radius, \
    CURVATURE_INVALID, curvature_valid
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
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_scale = 1024 * 768 / (width * height)

    wrt = None
    if args.write:
        # out_file = os.path.splitext(os.path.basename(args.file))
        # out_file = os.path.join('out', '{}-processed.mp4'.format(out_file[0]))
        out_file = args.write
        wrt = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1848, 720))

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
    edc = EdgeDetectionConv(detect_lines=False, mask=roi_mask)
    swt = EdgeDetectionSWT(mask=roi_mask, max_length=8)
    edt = EdgeDetectionTemporal(mask=roi_mask, detect_lines=False)

    edg_primary = edc
    edg_secondary = swt
    edg_threshold = 0.3

    lcm = LaneColorMasking(luminance_kernel_width=33)
    lcm.detect_lines = False
    lcm.blue_threshold = 250
    lcm.light_cutoff = .95

    state = LaneDetectionState()
    curvature_hist = CURVATURE_INVALID
    curvature_age = 0
    curvature_max_age = 16

    seek_to = max(0, min(num_frames - 1, args.seek))
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)

    while True:
        t_start = datetime.now()
        ret, img = cap.read()
        if not ret:
            break
        print('Processing frame {} ...'.format(int(cap.get(cv2.CAP_PROP_POS_FRAMES))))

        # Undistort and transform to bird's eye view
        img, _ = cc.undistort(img, False)
        warped = bev.warp(img)
        warped_f = np.float32(warped) / 255

        # Convert to grayscale and normalize OpenCV L*a*b* value ranges.
        lab = cv2.cvtColor(warped_f, cv2.COLOR_BGR2LAB)
        yellows = lab[..., 2] / 127
        yellows[yellows < 0.5] = 0
        cv2.normalize(yellows, yellows, 1, norm_type=cv2.NORM_MINMAX)
        gray = cv2.max(lab[..., 0] / 100, yellows)
        lab[..., 0] = gray * 100

        # Preprocessing: Detect lane line pixel candidates
        warped_f = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        warped = np.uint8(warped_f * 255)  # type: np.ndarray
        edges = detect_lane_pixels(warped, edg_primary, edg_secondary, lcm, edg_threshold) * roi_mask_hard

        # Detect the lane lines
        canvas = warped_f.copy()
        results = detect_and_render_lanes(canvas, edges, state, mx, my, render_lanes=True, render_boxes=True)
        (left_valid, left_fit), (right_valid, right_fit) = results

        # Prepare a image for alpha blending.
        img = np.float32(img) / 255.
        img_alpha = img.copy()

        # We are tracking the bottom lane line points. By definition of the BEV transformation,
        # the left line is supposed to be at x=100, while the right lane should be at x=200.
        bottom_left, bottom_right = None, None

        # We now obtain the lane lines and transform them to camera space.
        y_bottom, y_top = warped.shape[0], warped.shape[0] // 2
        left, right = None, None
        if left_fit is not None:
            left = get_points(left_fit, y_bottom, y_top)
            bottom_left = left[0][0]
            bottom_right = bottom_left + 100
            left = np.floor(bev.unproject(left)).astype(np.int32)
        if right_fit is not None:
            right = get_points(right_fit, y_bottom, y_top)
            bottom_right = right[0][0]
            bottom_left = bottom_right - 100 if bottom_left is None else bottom_left
            right = np.floor(bev.unproject(right)).astype(np.int32)

        # By checking the lane line center point (which should be at x=150) we can determine
        # the deviation of the car's center point from the lane's center.
        # Since each half-lane is 50 pixels, we normalize by this.
        if bottom_left is not None:
            lane_center = (bottom_left + bottom_right) / 2
            delta = 150 - lane_center
            deviation_from_center = delta / 50
            deviation_from_center_m = delta * mx
        else:
            deviation_from_center = None
            deviation_from_center_m = None

        # Prepare the HUD
        hud_height = 64
        cv2.fillPoly(img, [np.array([[0, 0], [img.shape[1], 0],
                                     [img.shape[1], hud_height], [0, hud_height]])], color=(.25, .25, .25))
        img = cv2.addWeighted(img, .5, img_alpha, .5, 0)
        img_alpha = img.copy()

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
            measurement_alpha = 0.4 if left_valid and right_valid else 0.1
            cv2.fillPoly(img_alpha, [all], color, lineType=cv2.LINE_AA)
            img = cv2.addWeighted(img, (1 - measurement_alpha), img_alpha, measurement_alpha, 0)

        # Draw the lane lines
        if left is not None:
            color = VALID_COLOR if left_valid else CACHED_COLOR
            cv2.polylines(img, [left], False, color, 3, lineType=cv2.LINE_AA)
        if right is not None:
            color = VALID_COLOR if right_valid else CACHED_COLOR
            cv2.polylines(img, [right], False, color, 3, lineType=cv2.LINE_AA)

        # Render the HUD text
        curvature = 0
        if (left is not None) and (right is None):
            cl_b = curvature_radius(left_fit, warped.shape[0], mx)
            cl_t = curvature_radius(left_fit, 0, mx)
            curvature = (cl_b + cl_t) / 2
        elif (left is None) and (right is not None):
            cr_b = curvature_radius(right_fit, warped.shape[0], mx)
            cr_t = curvature_radius(right_fit, 0, mx)
            curvature = (cr_b + cr_t) / 2
        elif (left is not None) and (right is not None):
            cl_b = curvature_radius(left_fit, warped.shape[0], mx)
            cl_t = curvature_radius(left_fit, 0, mx)
            cl = (cl_b + cl_t) / 2
            cr_b = curvature_radius(right_fit, warped.shape[0], mx)
            cr_t = curvature_radius(right_fit, 0, mx)
            cr = (cr_b + cr_t) / 2
            agreement = cl * cr > 0
            measurement_alpha = 0.5
            if agreement and cl < 0:
                curvature = measurement_alpha * cl + (1 - measurement_alpha) * cr
            elif agreement and cl > 0:
                curvature = (1 - measurement_alpha) * cl + measurement_alpha * cr
            else:
                curvature = 0

        # If the curvature suddenly flips signs from the previous value, drop it
        if curvature * curvature_hist > 0 and curvature_valid(curvature_hist):
            mix_alpha = 0.1
            curvature_hist = mix_alpha * curvature + (1 - mix_alpha) * curvature_hist
            curvature_age = 0
        elif not curvature_valid(curvature_hist):
            curvature_hist = curvature
            curvature_age = 0
        else:
            curvature_age += 1
            if curvature_age >= curvature_max_age:
                curvature_hist = CURVATURE_INVALID
                curvature_age = 0

        # Display lane center deviation
        text = 'Deviation from lane center: {0:.0f}% ({1:0.2}m)'.format(deviation_from_center * 100,
                                                                        deviation_from_center_m) \
            if deviation_from_center is not None else 'Position: unknown'
        cv2.putText(img, text, (4, 24), cv2.FONT_HERSHEY_DUPLEX, 0.75, (1, 1, 1), 1, cv2.LINE_AA)

        # Display curvature
        text = 'Curvature radius: {0:0.2f}m'.format(curvature_hist)
        if not curvature_valid(curvature_hist):
            text = 'Curvature radius: disagreement'
        elif curvature_hist == 0:
            text = 'Curvature radius: none'
        cv2.putText(img, text, (4, 48), cv2.FONT_HERSHEY_DUPLEX, 0.75, (1, 1, 1), 1, cv2.LINE_AA)

        scale = img.shape[0] / edges.shape[0]
        edges = cv2.resize(edges, (int(scale*edges.shape[1]), img.shape[0]))
        canvas = cv2.resize(canvas, (int(scale * canvas.shape[1]), img.shape[0]))
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        img = np.hstack([img, edges, canvas])
        if wrt is not None:
            assert img.shape[:2] == (720, 1848)
            img = np.uint8(np.clip(img * 255, 0, 255))
            wrt.write(img)

        resized = cv2.resize(img, (0, 0), fx=display_scale, fy=display_scale)
        cv2.imshow('video', resized)

        # Attempt to stay close to the original FPS.
        t_end = datetime.now()
        t_delta = (t_end - t_start).total_seconds() * 1000
        t_wait = int(max(1, fps - t_delta))

        if cv2.waitKey(t_wait) == 27:
            break

    if wrt is not None:
        wrt.release()
    cap.release()


def parse_args():
    parser = argparse.ArgumentParser()
    v = parser.add_argument_group('Video')
    v.add_argument(metavar='VIDEO', dest='file', default='project_video.mp4',
                   help='The video file to process.')
    v.add_argument('-w', '--write', dest='write', default=None,
                   help='Writes an output video file')
    v.add_argument('-s', '--seek', dest='seek', default=0, type=int,
                   help='The video frame to seek to')
    args = parser.parse_args()
    if not os.path.exists(args.file):
        parser.error('The specified video {} could not be found.'.format(args.file))

    return args


if __name__ == '__main__':
    main(parse_args())
