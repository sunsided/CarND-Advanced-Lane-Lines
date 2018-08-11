"""
Temporally smoothed lane line proposals.
See video at https://www.youtube.com/watch?v=n1aZLTyl9BI
"""

import cv2
import numpy as np
from typing import Tuple
from moviepy.editor import VideoFileClip

from pipeline import CameraCalibration, BirdsEyeView, ImageSection, Point, EdgeDetectionTemporal

# Set to True to render a full video with the results
#EXPORT_VIDEO_TO = 'challenge_video_temporal_diff.mp4'
#EXPORT_VIDEO_TO = 'harder_challenge_video_temporal_diff.mp4'
EXPORT_VIDEO_TO = None

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

    edt = EdgeDetectionTemporal()

    def process_frame(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        filled = edt.filter(img, is_lab=False)
        return edt.edges_filtered, filled

    def video_process_frame(img: np.ndarray) -> np.ndarray:
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

    edt.reset()
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
