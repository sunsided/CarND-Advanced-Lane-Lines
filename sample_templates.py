"""
This script is used to sample templates for template matching.
"""

import os
import cv2
import numpy as np
from pipeline import CameraCalibration, BirdsEyeView, ImageSection, Point
from pipeline import lab_enhance_yellow

RETURN = 13
SPACE = 32
BACKSPACE = 8

WINDOW_WIDTH = 16
WINDOW_HEIGHT = 16

NEGATIVES = 0
POSITIVES = 0

# PATH = 'harder_challenge_video.mp4'
# PATH = 'challenge_video.mp4'
PATH = 'project_video.mp4'


def onMouse(event, x, y, flags, params):
    global NEGATIVES, POSITIVES

    params['position'] = x, y
    img = params['gray']

    is_shift = (flags & cv2.EVENT_FLAG_SHIFTKEY) > 0
    sample_negative = is_shift

    if event == 0 or event == 4:
        return
    if event == 1:
        hw = WINDOW_WIDTH // 2
        hh = WINDOW_HEIGHT // 2
        window = img[y - hh: y - hh + WINDOW_HEIGHT, x - hw: x - hw + WINDOW_WIDTH]
        if sample_negative:
            print('Sample negative')
            path = os.path.join('templates', 'negative-{}.png'.format(NEGATIVES))
            NEGATIVES += 1
        else:
            print('Sample positive')
            path = os.path.join('templates', 'positive-{}.png'.format(POSITIVES))
            POSITIVES += 1
        print(window.min(), window.max())
        cv2.imwrite(path, np.uint8(window * 255))
        return
    else:
        print('Unhandled mouse event: {}, {}'.format(event, flags))


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

    cap = cv2.VideoCapture(PATH)
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    current_pos, last_pos = 0, None
    params = dict(gray=None, position=None, action=0)

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
        img, _ = lab_enhance_yellow(img, normalize=True)

        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

        params['gray'] = img
        cv2.setMouseCallback(window, onMouse, params)

        position = params['position']
        if position is not None:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            pt1 = (position[0] - WINDOW_WIDTH // 2, position[1] - WINDOW_HEIGHT // 2)
            pt2 = pt1[0] + WINDOW_WIDTH, pt1[1] + WINDOW_HEIGHT
            cv2.rectangle(img, pt1, pt2, (0, 0, 1))

        cv2.imshow(window, img)
        key = cv2.waitKey(33)
        if key == 27:
            break
        if key == SPACE:
            current_pos += 1
        elif key == BACKSPACE:
            current_pos -= 1
        elif key == RETURN:
            current_pos += 60

    cap.release()


if __name__ == '__main__':
    main()
