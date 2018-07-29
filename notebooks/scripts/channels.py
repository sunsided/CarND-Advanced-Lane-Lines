import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Tuple


cmap_lab_l = LinearSegmentedColormap.from_list('L*', [(0, 0, 0), (1, 1, 1)], N=255)
cmap_lab_a = LinearSegmentedColormap.from_list('a*', [(0, 1, 0), (1, 0, 0)], N=255)
cmap_lab_b = LinearSegmentedColormap.from_list('b*', [(1, 1, 0), (0, 0, 1)], N=255)


def bgr2lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)


def show_single_image(img: np.ndarray, preview: bool=False, size: Tuple[int, int] = (8, 8), cmap='gray',
                      colorbar: bool=False, vmin=None, vmax=None):
    if preview:
        pixels, max_pixels = np.prod(img.shape[:2]), 1024*768
        if pixels > max_pixels:
            scale = max_pixels / pixels
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=size)
    p = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    sns.despine()
    if colorbar:
        plt.colorbar(p)


def show_two_images(img: np.ndarray, img2: np.ndarray, preview: bool=False, size: Tuple[int, int] = (9, 8),
                    vmin=None, vmax=None, cmap: str='gray', cmap0=None, cmap1=None, colorbar: bool=False):
    if preview:
        pixels, max_pixels = np.prod(img.shape[:2]), 1024*768
        if pixels > max_pixels:
            scale = max_pixels / pixels
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        pixels, max_pixels = np.prod(img2.shape[:2]), 1024 * 768
        if pixels > max_pixels:
            scale = max_pixels / pixels
            img2 = cv2.resize(img2, (0, 0), fx=scale, fy=scale)
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=size)
    a = axs[0].imshow(img, cmap=cmap0 or cmap, vmin=vmin, vmax=vmax)
    b = axs[1].imshow(img2, cmap=cmap1 or cmap, vmin=vmin, vmax=vmax)
    plt.tight_layout()
    sns.despine()
    if colorbar:
        plt.colorbar(a, ax=axs[0])
        plt.colorbar(b, ax=axs[1])


def show_channels_lab(lab, size: Tuple[int, int]=(16, 8), vmin=None, vmax=None):
    show_channels(lab, 'L*', 'a* (green-red)', 'b* (yellow-blue)',
                  size, cmap0=cmap_lab_l, cmap1=cmap_lab_a, cmap2=cmap_lab_b, vmin=vmin, vmax=vmax)


def show_channels(img, title0: str, title1: str, title2: str, size: Tuple[int, int] = (16, 4), cmap: str='gray',
                  vmin=None, vmax=None,
                  cmap0=None, cmap1=None, cmap2=None):
    f, axs = plt.subplots(nrows=1, ncols=3, figsize=size)
    a = axs[0].imshow(img[..., 0], cmap=cmap0 or cmap, vmin=vmin, vmax=vmax)
    b = axs[1].imshow(img[..., 1], cmap=cmap1 or cmap, vmin=vmin, vmax=vmax)
    c = axs[2].imshow(img[..., 2], cmap=cmap2 or cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(title0)
    axs[1].set_title(title1)
    axs[2].set_title(title2)
    plt.tight_layout()
    sns.despine()
    plt.colorbar(a, ax=axs[0])
    plt.colorbar(b, ax=axs[1])
    plt.colorbar(c, ax=axs[2])


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
