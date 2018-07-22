"""
This script is used to sample yellow and white lane marker colors from images
given their overall appearance.
"""

import csv
import os
from typing import List

import cv2
import numpy as np
from notebooks.scripts.histogram import histogram_vec

from pipeline import CameraCalibration, BirdsEyeView, ImageSection, Point

RETURN = 13
SPACE = 32
BACKSPACE = 8

NBINS = 16

# PATH = 'harder_challenge_video.mp4'
PATH = 'challenge_video.mp4'


class Frame:
    def __init__(self, path: str, frame: int):
        self._path = path
        self._frame = frame
        self._white = []
        self._yellow = []
        self._dirty = False
        self._hist = None

    def set_histogram(self, histogram: np.ndarray):
        self._hist = histogram

    def add_white(self, value: np.ndarray):
        self._white.append(value)
        self._dirty = True

    def add_yellow(self, value: np.ndarray):
        self._yellow.append(value)
        self._dirty = True

    @property
    def hist(self):
        return self._hist

    @property
    def white(self):
        return self._white

    @property
    def yellow(self):
        return self._yellow

    @property
    def changed(self):
        return self._dirty


def onMouse(event, x, y, flags, params):
    img = params['yuv']
    frame = params['frame']
    h, w = img.shape[:2]
    value = img[y, x]

    is_shift = (flags & cv2.EVENT_FLAG_SHIFTKEY) > 0
    sample_white = not is_shift

    if event == 0 or event == 4:
        return
    if event == 1:
        if sample_white:
            print('Adding white sample:  {}'.format(value))
            frame.add_white(value)
        else:
            print('Adding yellow sample: {}'.format(value))
            frame.add_yellow(value)
    else:
        print('Unhandled mouse event: {}, {}'.format(event, flags))


def main():
    basename = os.path.splitext(os.path.basename(PATH))[0]

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

    cap = cv2.VideoCapture(PATH)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    current_pos, last_pos = 0, None

    frame = None
    frames = []  # type: List[Frame]

    window = 'Video'
    cv2.namedWindow(window, cv2.WINDOW_KEEPRATIO)
    while True:
        current_pos = min(length, max(0, current_pos))
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        ret, img = cap.retrieve()
        if not ret:
            print('End of video.')
            break

        img, _ = cc.undistort(img, False)
        img = bev.warp(img)

        # Stretch for better sampling
        img = cv2.resize(img, (0, 0), fx=3, fy=1)

        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if last_pos != current_pos:
            if frame is not None and frame.changed:
                frames.append(frame)

            print('Video progress: {}/{} ({:.2f}%)'.format(current_pos, length, 100*current_pos/length))
            # Prepare the frame
            frame = Frame(PATH, current_pos)
            last_pos = current_pos

        # Take YUV histogram
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        histogram = histogram_vec(yuv, NBINS) / np.prod(yuv.shape[:2])
        assert frame is not None
        frame.set_histogram(histogram)

        # Hook up the window callback
        params = dict(bgr=img, yuv=yuv, frame=frame)
        cv2.setMouseCallback(window, onMouse, params)

        cv2.imshow(window, img)
        key = cv2.waitKey()
        if key == 27:
            break
        if key == SPACE:
            current_pos += 1
        elif key == BACKSPACE:
            current_pos -= 1
        elif key == RETURN:
            current_pos += 60

    cap.release()

    # Dangling frames are considered as well
    if frame is not None and frame.changed and ((len(frames) > 0 and frames[-1] != frame) or len(frames) == 0):
        frames.append(frame)

    if len(frames) == 0:
        return

    # Store the frames
    with open('lane-samples-{}.csv'.format(basename), 'w') as f:
        hist_fields = ['h{}'.format(i) for i in range(0, 3 * NBINS)]
        fieldnames = ['id', 'white', 'y', 'u', 'v', 'y_mean', 'u_mean', 'v_mean', 'y_std', 'u_std', 'v_std', *hist_fields]
        writer = csv.writer(f, fieldnames)
        writer.writerow(fieldnames)
        i = 0
        for frame in frames:
            if len(frame.white) > 0:
                white_mean = np.mean(frame.white, axis=0)
                white_std = np.std(frame.white, axis=0)
                for sample in frame.white:
                    values = [i, 1, sample[0], sample[1], sample[2], *white_mean, *white_std, *frame.hist]
                    writer.writerow(values)
                    i += 1

            if len(frame.yellow) > 0:
                yellow_mean = np.mean(frame.yellow, axis=0)
                yellow_std = np.std(frame.yellow, axis=0)
                for sample in frame.white:
                    values = [i, 0, sample[0], sample[1], sample[2], *yellow_mean, *yellow_std, *frame.hist]
                    writer.writerow(values)
                    i += 1


if __name__ == '__main__':
    main()
