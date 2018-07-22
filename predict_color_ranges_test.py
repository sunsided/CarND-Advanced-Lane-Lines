"""
This script is used to sample yellow and white lane marker colors from images
given their overall appearance.
"""

import cv2
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from moviepy.editor import VideoFileClip

from notebooks.scripts.histogram import histogram_vec
from pipeline import CameraCalibration, BirdsEyeView, ImageSection, Point

# Set to True to render a full video with the results
EXPORT_VIDEO = False

# PATH = 'harder_challenge_video.mp4'
PATH = 'challenge_video.mp4'


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

    def process_masks(img: np.ndarray):
        yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        histogram = histogram_vec(yuv, nbins) / np.prod(yuv.shape[:2])

        # Normalize the histogram
        histogram = (histogram - hist_mean) / hist_std

        # Predict the color range
        X = np.expand_dims(np.hstack([1, histogram]), axis=0).astype(np.float32)
        white_range = (mlp.predict(X) * yuv_std + yuv_mean).reshape((2, 3))

        X = np.expand_dims(np.hstack([0, histogram]), axis=0).astype(np.float32)
        yellow_range = (mlp.predict(X) * yuv_std + yuv_mean).reshape((2, 3))

        white_mask = get_soft_mask(yuv, white_range, max_scale=3, steps=3)
        yellow_mask = get_soft_mask(yuv, yellow_range, max_scale=3, steps=3)

        return white_mask, yellow_mask

    def process_frame(img: np.ndarray) -> np.ndarray:
        white_mask, yellow_mask = process_masks(img)
        return np.stack([yellow_mask, np.zeros_like(white_mask), white_mask], axis=2)

    def video_process_frame(img: np.ndarray) -> np.ndarray:
        img, _ = cc.undistort(img, False)
        warped = bev.warp(img)

        white_mask, yellow_mask = process_masks(warped)

        # Global non-maximum suppression
        max_white = white_mask.max()
        max_yellow = yellow_mask.max()
        white_mask[white_mask < max_white] = 0
        yellow_mask[yellow_mask < max_yellow] = 0

        color_mask = np.stack([yellow_mask, np.zeros_like(white_mask), white_mask], axis=2)
        white_mask_rgb = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2RGB)
        yellow_mask_rgb = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2RGB)

        split = np.ones(shape=(warped.shape[0], 10, 3), dtype=np.uint8) * 255
        img = np.hstack([warped, split, color_mask, split, white_mask_rgb, split, yellow_mask_rgb])
        return cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # Process the video
    if EXPORT_VIDEO:
        clip = VideoFileClip(PATH)  # .subclip(0, 5)
        clip = clip.fl_image(video_process_frame)
        clip.write_videofile('test.mp4', audio=False)

    cap = cv2.VideoCapture(PATH)
    window = 'Video'
    cv2.namedWindow(window, cv2.WINDOW_KEEPRATIO)

    while True:
        ret, img = cap.read()
        if not ret:
            print('End of video.')
            break

        img, _ = cc.undistort(img, False)
        warped = bev.warp(img)

        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        mask = process_frame(warped)

        # Obtain the pixel histogram in the lower region
        slice_height = 100
        yellow_sum = np.sum(mask[-slice_height:, ..., 0], axis=0)
        yellow_hist = yellow_sum / (slice_height*255)

        white_sum = np.sum(mask[-slice_height:, ..., 2], axis=0)
        white_hist = white_sum / (slice_height * 255)

        for x in range(0, mask.shape[1]):
            y_offset = (yellow_hist[x] * 200).astype(np.uint8)
            w_offset = (white_hist[x] * 200).astype(np.uint8)
            offset = max(y_offset, w_offset)
            pt0 = (x, mask.shape[0] - 1)
            pt1 = (x, mask.shape[0] - 1 - offset)
            cv2.line(mask, pt0, pt1, (255, 255, 255))

        unwarped = bev.unwarp(mask, (img.shape[1], img.shape[0]))
        unwarped = cv2.cvtColor(unwarped, cv2.COLOR_RGB2BGR)

        img = np.hstack([img, unwarped])
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        cv2.imshow(window, img)
        key = cv2.waitKey()
        if key == 27:
            break


if __name__ == '__main__':
    main()
