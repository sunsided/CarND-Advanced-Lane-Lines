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


def get_mask(img: np.ndarray, range: np.ndarray, nstd: float) -> np.ndarray:
    span = nstd * range[1, ...]
    lo = np.clip(np.round(range[0, ...] - span), 0, 255).astype(np.uint8)
    hi = np.clip(np.round(range[0, ...] + span), 0, 255).astype(np.uint8)
    return cv2.inRange(img, lo, hi)


def get_soft_mask(img: np.ndarray, value_range: np.ndarray, steps: int = 4, max_scale: float = 2.5) -> np.ndarray:
    soft_mask = None
    scale = max_scale / steps
    for i in range(0, steps):
        mask = get_mask(img, value_range, i * scale) // steps
        soft_mask = soft_mask + mask if soft_mask is not None else mask
    return soft_mask


def main():
    params = joblib.load('color_ranges_model.pkl')
    mlp = params['mlp']  # type: MLPRegressor
    hist_mean = params['hist_mean']  # type: np.ndarray
    hist_std = params['hist_std']  # type: np.ndarray
    yuv_mean = params['yuv_mean']  # type: np.ndarray
    yuv_std = params['yuv_std']  # type: np.ndarray
    nbins = params['nbins']  # type: int

    path = 'harder_challenge_video.mp4'

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

        white_mask = get_soft_mask(yuv, white_range, max_scale=4.5)
        yellow_mask = get_soft_mask(yuv, yellow_range, max_scale=2.5)
        return white_mask, yellow_mask

    def process_frame(img: np.ndarray) -> np.ndarray:
        white_mask, yellow_mask = process_masks(img)
        return np.stack([yellow_mask, np.zeros_like(white_mask), white_mask], axis=2)

    def video_process_frame(img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        white_mask, yellow_mask = process_masks(img)
        color_mask = np.stack([yellow_mask, np.zeros_like(white_mask), white_mask], axis=2)
        white_mask_rgb = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2RGB)
        yellow_mask_rgb = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2RGB)
        top_row = np.hstack([img, color_mask])
        bottom_row = np.hstack([white_mask_rgb, yellow_mask_rgb])
        return np.vstack([top_row, bottom_row])

    # Process the video
    clip = VideoFileClip(path)
    clip = clip.fl_image(video_process_frame)
    clip.write_videofile('test.mp4', audio=False)

    cap = cv2.VideoCapture(path)
    window = 'Video'
    cv2.namedWindow(window, cv2.WINDOW_KEEPRATIO)

    while True:
        ret, img = cap.read()
        if not ret:
            print('End of video.')
            break

        # Take YUV histogram
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        mask = process_frame(img)
        img = np.hstack([img, mask])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow(window, img)
        key = cv2.waitKey()
        if key == 27:
            break


if __name__ == '__main__':
    main()
