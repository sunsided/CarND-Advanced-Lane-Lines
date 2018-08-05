"""
Temporally smoothed lane line proposals.
See video at https://www.youtube.com/watch?v=n1aZLTyl9BI
"""

import cv2
import numpy as np
from typing import Tuple
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip

from pipeline import CameraCalibration, BirdsEyeView, ImageSection, Point

# Set to True to render a full video with the results
#EXPORT_VIDEO_TO = 'challenge_video_temporal_diff.mp4'
#EXPORT_VIDEO_TO = 'harder_challenge_video_temporal_diff.mp4'

PATH = 'harder_challenge_video.mp4'
#PATH = 'challenge_video.mp4'


def get_mask(img: np.ndarray, range: np.ndarray, nstd: float) -> np.ndarray:
    span = nstd * range[1, ...]
    lo = np.clip(np.round(range[0, ...] - span), 0, 255).astype(np.uint8)
    hi = np.clip(np.round(range[0, ...] + span), 0, 255).astype(np.uint8)
    return cv2.inRange(img, lo, hi)


def get_soft_mask(img: np.ndarray, value_range: np.ndarray, steps: int = 4, max_scale: float = 2.5) -> np.ndarray:
    soft_mask = None
    scale = max_scale / steps
    coeffs = np.logspace(0, -2, steps).astype(np.float32)
    coeffs = coeffs / coeffs.sum()
    for i in range(0, steps):
        mask = get_mask(img, value_range, i * scale).astype(np.float32)
        mask *= coeffs[i]
        soft_mask = soft_mask + mask if soft_mask is not None else mask
    soft_mask = (soft_mask / soft_mask.max()) * 255
    return np.clip(soft_mask, 0, 255).astype(np.uint8)


def main():
    params = joblib.load('color_ranges_model.pkl')
    mlp = params['mlp']  # type: MLPRegressor
    hist_mean = params['hist_mean']  # type: np.ndarray
    hist_std = params['hist_std']  # type: np.ndarray
    yuv_mean = params['yuv_mean']  # type: np.ndarray
    yuv_std = params['yuv_std']  # type: np.ndarray
    nbins = params['nbins']  # type: int

    cc = CameraCalibration.from_pickle('calibration.pkl')

    section = ImageSection(
        top_left=Point(x=580, y=461.75),
        top_right=Point(x=702, y=461.75),
        bottom_right=Point(x=1013, y=660),
        bottom_left=Point(x=290, y=660),
    )

    bev = BirdsEyeView(section,
                       section_width=3.6576,  # one lane width in meters
                       section_height=2 * 13.8826)  # two dash distances in meters

    previous_edges = None
    previous_grays_slow = None
    previous_grays_fast = None

    def float2uint8(img: np.ndarray, scale: float=255) -> np.ndarray:
        return np.clip(img * scale, 0, 255).astype(np.uint8)

    def uint82float(img: np.ndarray, scale: float=255) -> np.ndarray:
        return np.float32(img) / scale

    def rescale(img: np.ndarray) -> np.ndarray:
        min_, max_ = img.min(), img.max()
        return (img - min_) / (max_ - min_)

    def process_frame(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nonlocal previous_edges, previous_grays_slow, previous_grays_fast
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        gray = lab[..., 0]
        gray = cv2.GaussianBlur(gray, (3, 3), 5)

        # Equalize for edge detection
        equalized = np.ma.masked_equal(gray, 0)
        slice_height = 5
        for y in range(0, gray.shape[0], slice_height):
            top, bottom = y, y + slice_height
            equalized[top:bottom, ...] = cv2.equalizeHist(equalized[top:bottom, ...])
        gray = np.ma.filled(equalized, 0)

        if previous_grays_slow is None:
            temporally_smoothed_slow = gray
            temporally_smoothed_fast = gray
        else:
            alpha_slow = 0.1
            temporally_smoothed_slow = alpha_slow * gray + (1-alpha_slow) * previous_grays_slow

            alpha_fast = 0.8
            temporally_smoothed_fast = alpha_fast * gray + (1 - alpha_fast) * previous_grays_fast

        # For edge detection we're going to need an integral image.
        temporally_smoothed_slow_8 = float2uint8(temporally_smoothed_slow, 1)
        temporally_smoothed_fast_8 = float2uint8(temporally_smoothed_fast, 1)

        # The reflections of the dashboard can be found mostly in vertical edges.
        ts_edges_y = np.sqrt((cv2.Scharr(temporally_smoothed_slow_8, cv2.CV_32F, 0, 1) / 255.)**2)
        dashboard_mask = 1 - ts_edges_y
        dashboard_mask = cv2.medianBlur(dashboard_mask, 5)
        dashboard_mask = np.clip(dashboard_mask, 0, 1)

        # Apply difference of gaussian edge detection.
        inp_8 = cv2.morphologyEx(temporally_smoothed_fast_8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        inp = np.float32(inp_8) / 255.
        dog = cv2.GaussianBlur(inp, (9, 9), 5) - cv2.GaussianBlur(inp, (17, 17), 9)
        dog = rescale(np.clip(dog, 0, 1)) * dashboard_mask

        # Obtain new edges.
        if previous_edges is None:
            previous_edges = np.zeros_like(dog)

        edge_alpha = .4
        edges_filtered = edge_alpha * dog + (1-edge_alpha) * previous_edges
        edges_filtered = rescale(edges_filtered)
        edges_filtered = cv2.GaussianBlur(edges_filtered, (5, 5), 5)

        # Run canny on the pre-filtered edges
        edges_filtered_8 = float2uint8(edges_filtered)
        edges_canny_8 = cv2.Canny(edges_filtered_8, 64, 100)

        # We perform blob detection; for this, we close nearby contours.
        edges_contours_8 = cv2.morphologyEx(edges_canny_8, cv2.MORPH_BLACKHAT, np.ones((13, 13), np.uint8))
        edges_contours_8 = cv2.morphologyEx(edges_contours_8, cv2.MORPH_ERODE, np.ones((2, 2), np.uint8))
        m2, contours, hierarchy = cv2.findContours(edges_contours_8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        good_contours = []
        ok_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            if area > 600:
                good_contours.append(cnt)
            else:
                ok_contours.append(cnt)

        filled = edges_canny_8 // 2
        cv2.drawContours(filled, ok_contours, -1, 64, cv2.FILLED, cv2.LINE_4)
        cv2.drawContours(filled, good_contours, -1, 255, cv2.FILLED, cv2.LINE_4)

        # Carry the current state on to the next time stamp
        previous_grays_fast = temporally_smoothed_fast
        previous_grays_slow = temporally_smoothed_slow
        previous_edges = edges_filtered

        # Convert for returning
        return edges_filtered_8, filled

    def video_process_frame(img: np.ndarray) -> np.ndarray:
        nonlocal previous_edges
        img, _ = cc.undistort(img, False)
        warped = bev.warp(img)

        edges, filtered = process_frame(warped)

        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        edges[..., 2] = 0

        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
        filtered[..., 0] = 0

        #edges = bev.unwarp(edges, (img.shape[1], img.shape[0]))
        filtered_unwarped = bev.unwarp(filtered, (img.shape[1], img.shape[0]))

        # Resize the original image to fit the on top of the unwarped filtered image.
        frame_height = filtered.shape[0]
        img_width = int(img.shape[1] * (frame_height / 2) / img.shape[0])
        img = cv2.resize(img, (img_width, frame_height // 2))

        # Resize the filtered image to fit underneath the original image
        filtered_unwarped = cv2.resize(filtered_unwarped, (img.shape[1], img.shape[0]))

        left = np.vstack([img, filtered_unwarped])
        frame = np.hstack([left, edges, filtered])

        return frame

    # Process the video
    if isinstance(EXPORT_VIDEO_TO, str):
        clip = VideoFileClip(PATH)  # .subclip(0, 5)
        clip = clip.fl_image(video_process_frame)
        clip.write_videofile(EXPORT_VIDEO_TO, audio=False)

    cap = cv2.VideoCapture(PATH)
    window = 'Video'
    cv2.namedWindow(window, cv2.WINDOW_KEEPRATIO)

    previous_edges = None
    previous_grays_slow = None
    previous_grays_fast = None
    while True:
        ret, img = cap.read()
        if not ret:
            print('End of video.')
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = video_process_frame(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow(window, img)
        key = cv2.waitKey()
        if key == 27:
            break


if __name__ == '__main__':
    main()
