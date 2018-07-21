import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def show_single_image(img: np.ndarray):
    plt.imshow(img)
    plt.tight_layout()
    sns.despine()


def show_channels(img, title0: str, title1: str, title2: str):
    f, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    cmap = 'gray'
    axs[0].imshow(img[..., 0], cmap=cmap)
    axs[1].imshow(img[..., 1], cmap=cmap)
    axs[2].imshow(img[..., 2], cmap=cmap)
    axs[0].set_title(title0)
    axs[1].set_title(title1)
    axs[2].set_title(title2)
    plt.tight_layout()
    sns.despine()


def channel_gallery(bgr: np.ndarray) -> None:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    luv = cv2.cvtColor(bgr, cv2.COLOR_BGR2Luv)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

    f, axs = plt.subplots(nrows=6, ncols=3, figsize=(13, 5 * 3), sharex=True, sharey=True)
    cmap = 'gray'

    axs[0, 0].imshow(bgr[..., 2], cmap=cmap)
    axs[0, 1].imshow(bgr[..., 1], cmap=cmap)
    axs[0, 2].imshow(bgr[..., 0], cmap=cmap)
    axs[0, 0].set_title('RGB red')
    axs[0, 1].set_title('RGB green')
    axs[0, 2].set_title('RGB blue')

    axs[1, 0].imshow(hsv[..., 0], cmap=cmap)
    axs[1, 1].imshow(hsv[..., 1], cmap=cmap)
    axs[1, 2].imshow(hsv[..., 2], cmap=cmap)
    axs[1, 0].set_title('HSV hue')
    axs[1, 1].set_title('HSV saturation')
    axs[1, 2].set_title('HSV value')

    axs[2, 0].imshow(hls[..., 0], cmap=cmap)
    axs[2, 1].imshow(hls[..., 1], cmap=cmap)
    axs[2, 2].imshow(hls[..., 2], cmap=cmap)
    axs[2, 0].set_title('HLS hue')
    axs[2, 1].set_title('HLS lightness')
    axs[2, 2].set_title('HLS saturation')

    axs[3, 0].imshow(lab[..., 0], cmap=cmap)
    axs[3, 1].imshow(lab[..., 1], cmap=cmap)
    axs[3, 2].imshow(lab[..., 2], cmap=cmap)
    axs[3, 0].set_title('CIE L*a*b* luminance (L*)')
    axs[3, 1].set_title('CIE L*a*b* green-red (a*)')
    axs[3, 2].set_title('CIE L*a*b* blue-yellow (b*)')

    axs[4, 0].imshow(luv[..., 0], cmap=cmap)
    axs[4, 1].imshow(luv[..., 1], cmap=cmap)
    axs[4, 2].imshow(luv[..., 2], cmap=cmap)
    axs[4, 0].set_title('CIE L*u*v* luminance (L*)')
    axs[4, 1].set_title('CIE L*u*v* green-red (u*)')
    axs[4, 2].set_title('CIE L*u*v* blue-yellow (v*)')

    axs[5, 0].imshow(yuv[..., 0], cmap=cmap)
    axs[5, 1].imshow(yuv[..., 1], cmap=cmap)
    axs[5, 2].imshow(yuv[..., 2], cmap=cmap)
    axs[5, 0].set_title('YUV luma')
    axs[5, 1].set_title('YUV chroma U')
    axs[5, 2].set_title('YUV chroma V')

    plt.tight_layout()
    sns.despine()
