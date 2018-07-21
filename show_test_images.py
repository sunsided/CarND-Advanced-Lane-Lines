import os
import glob
import cv2


RIGHT_KEY = 100
LEFT_KEY = 97


def onMouse(event, x, y, mouse, params):
    hls = params['hls'][y, x]
    bgr = params['bgr'][y, x]
    h, w = params['hls'].shape[:2]
    if event == 1:
        print("({0}, {1}, {2}),".format(
           hls[0], hls[1], hls[2]))


def main():
    paths = glob.glob(os.path.join('test_images', '*.jpg'))

    i, prev_i = 0, None
    hls_mode = False

    window = 'image'
    cv2.namedWindow(window, cv2.WINDOW_KEEPRATIO)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))

    def load_image(i, clahe=False, blur=False):
        path = paths[i % len(paths)]
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        bgr = cv2.resize(bgr, (0, 0), fx=0.5, fy=0.5)
        hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

        if clahe:
            hls[..., 1] = clahe.apply(hls[..., 1])

        if blur:
            for c in range(3):
                hls[..., c] = cv2.medianBlur(hls[..., c], 3)

        bgr = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
        return bgr, hls

    while True:
        if prev_i != i:
            bgr, hls = load_image(i)
        img = hls if hls_mode else bgr

        cv2.setMouseCallback(window, onMouse, {'hls': hls, 'bgr': bgr})

        cv2.imshow(window, img)
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == RIGHT_KEY:
            i += 1
        elif key == LEFT_KEY:
            i -= 1
        elif key == 104:
            hls_mode = True
        elif key == 114:
            hls_mode = False

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()