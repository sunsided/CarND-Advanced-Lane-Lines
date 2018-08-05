"""
This script is used to sample yellow and white lane marker colors from images
given their overall appearance.
"""

import cv2
import numpy as np
from typing import Tuple
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip

from notebooks.scripts.histogram import histogram_vec
from pipeline import CameraCalibration, BirdsEyeView, ImageSection, Point

# Set to True to render a full video with the results
EXPORT_VIDEO = True

PATH = 'harder_challenge_video.mp4'
# PATH = 'challenge_video.mp4'


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

    def process_frame(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nonlocal previous_edges, previous_grays_slow, previous_grays_fast
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 5)

        if previous_grays_slow is None:
            temporally_smoothed_slow = gray
            temporally_smoothed_fast = gray
        else:
            alpha_slow = 0.1
            temporally_smoothed_slow = alpha_slow*gray + (1-alpha_slow) * previous_grays_slow

            alpha_fast = 0.5
            temporally_smoothed_fast = alpha_fast * gray + (1 - alpha_fast) * previous_grays_fast

        #edges = np.float32(cv2.Canny(temporally_smoothed_slow, 64, 240)) / 255.

        # For edge detection we're going to need an integral image.
        temporally_smoothed_slow_8 = float2uint8(temporally_smoothed_slow, 1)
        temporally_smoothed_fast_8 = float2uint8(temporally_smoothed_fast, 1)

        # The reflections of the dashboard can be found mostly in vertical edges.
        ts_edges_y = np.sqrt((cv2.Scharr(temporally_smoothed_slow_8, cv2.CV_32F, 0, 1) / 255.)**2)
        dasboard_mask = 1 - ts_edges_y

        # Obtain new edges.
        edges_x = cv2.Sobel(temporally_smoothed_fast_8, cv2.CV_32F, 1, 0, ksize=3) / 255.
        edges = np.sqrt(edges_x ** 2)
        #edges = np.float32(cv2.Canny(temporally_smoothed_fast_8, 16, 160)) / 255.
        #edges_x = cv2.Scharr(temporally_smoothed_fast_8, cv2.CV_32F, 1, 0) / 255.
        #edges_y = np.zeros_like(edges_x) #cv2.Scharr(temporally_smoothed_fast_8, cv2.CV_32F, 0, 1) / 255.
        #edges = np.sqrt(edges_x**2 + edges_y**2) * dasboard_mask

        # Suppress all the edges detected due to the dashboard
        edges *= dasboard_mask

        #if previous_edges is not None:
        #    diff = 1 - np.sqrt((edges - previous_edges)**2)
        #    diff = (diff - diff.min()) / (diff.max() - diff.min())
        #else:
        #    diff = np.ones_like(gray)

        #diff = float2uint8(diff)
        #_, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #diff = uint82float(diff)

        #edges_filtered = edges * diff

        edges = float2uint8(edges)
        #edges_filtered = float2uint8(edges_filtered)

        #diff = float2uint8(np.sqrt((uint82float(gray) - uint82float(previous_grays_slow)**2)))

        # Carry the current state on to the next time stamp
        previous_grays_fast = temporally_smoothed_fast
        previous_grays_slow = temporally_smoothed_slow
        previous_edges = edges

        return temporally_smoothed_slow_8, edges #edges, edges_filtered

    def video_process_frame(img: np.ndarray) -> np.ndarray:
        nonlocal previous_edges
        img, _ = cc.undistort(img, False)
        warped = bev.warp(img)

        edges, filtered = process_frame(warped)

        edges = bev.unwarp(edges, (img.shape[1], img.shape[0]))
        filtered = bev.unwarp(filtered, (img.shape[1], img.shape[0]))

        frame = np.hstack([edges, filtered])
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        left = frame.shape[1] // 2 - img.shape[1] // 2
        frame[0:img.shape[0], left:left+img.shape[1], ...] = img

        return frame

    # Process the video
    if EXPORT_VIDEO:
        clip = VideoFileClip(PATH)  # .subclip(0, 5)
        clip = clip.fl_image(video_process_frame)
        clip.write_videofile('test-edges.mp4', audio=False)

    cap = cv2.VideoCapture(PATH)
    window = 'Video'
    cv2.namedWindow(window, cv2.WINDOW_KEEPRATIO)

    previous_edges = None
    previous_grays_slow = None
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
