import os
import glob
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from pipeline import CameraCalibration, BirdsEyeView, ImageSection, Point

from pipeline import lab_enhance_yellow

RETURN = 13
SPACE = 32
BACKSPACE = 8

WINDOW_WIDTH = 16
WINDOW_HEIGHT = 16

# PATH = 'harder_challenge_video.mp4'
# PATH = 'challenge_video.mp4'
PATH = 'project_video.mp4'


def main():
    negatives = [np.float32(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) / 255
                 for path in glob.glob(os.path.join('templates', '**', 'negative-*.png'))]
    positives = [np.float32(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) / 255
                 for path in glob.glob(os.path.join('templates', '**', 'positive-*.png'))]

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

    window = 'Video'
    cv2.namedWindow(window, cv2.WINDOW_KEEPRATIO)

    pe = ThreadPoolExecutor(max_workers=8)

    while True:
        current_pos = min(length, max(0, current_pos))
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        print('Frame: {}'.format(current_pos))
        ret, img = cap.retrieve()
        if not ret:
            print('End of video.')
            break

        img, _ = cc.undistort(img, False)
        warped = bev.warp(img)
        gray, _ = lab_enhance_yellow(warped, normalize=True)
        gray_orig = gray.copy()

        cv2.imwrite('sample.jpg', img)
        cv2.imwrite('sample-warped.jpg', warped)

        mode = cv2.TM_CCOEFF
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        def filter(template):
            m = cv2.matchTemplate(gray, template, mode)
            m[m < 0] = 0
            return m

        pos_matched = pe.map(filter, positives)
        neg_matched = pe.map(filter, negatives)

        pos_sum = np.zeros_like(gray)
        for result in pos_matched:
            pos_sum[8:745+8, 8:285+8] += result
        pos_sum /= len(positives)

        neg_sum = np.zeros_like(gray)
        for result in neg_matched:
            neg_sum[8:745 + 8, 8:285 + 8] += result
        neg_sum /= len(negatives)

        mask = (1 - neg_sum) * pos_sum
        mask[mask < 0] = 0
        mask = cv2.normalize(mask, 1, cv2.NORM_MINMAX)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask[mask < 0.05] = 0
        mask = cv2.normalize(mask, 1, cv2.NORM_MINMAX)

        img = np.hstack([warped.astype(np.float32) / 255.,
                         cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR),
                         cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                         cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(window, img)
        key = cv2.waitKey(0)
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
