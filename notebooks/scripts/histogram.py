import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def histogram_vec(img: np.ndarray, nbins: int) -> np.ndarray:
    hists = []
    for c in range(img.shape[2]):
        hist, bins = np.histogram(img[..., c].reshape(-1).flatten(), bins=nbins)
        hists.append(hist)
    return np.stack(hists, axis=0).flatten()


def plot_hls_histogram(img, nbins, nbins_lo):
    test_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Note that in uint8 type, the Hue channel is 0..180, representing 0..360 degrees!
    # This does not hold in float32 mode where the range is 0..360.

    f, axs = plt.subplots(nrows=3, ncols=3, figsize=(16, 12))
    axs[0, 0].imshow(test_hls[..., 0] / 360, cmap='gray')
    axs[0, 1].imshow(test_hls[..., 1] / 360, cmap='gray')
    axs[0, 2].imshow(test_hls[..., 2] / 360, cmap='gray')
    axs[1, 0].hist((test_hls[..., 0]).flatten(), bins=nbins, range=(0, 360), color='m')
    axs[1, 1].hist(test_hls[..., 1].flatten(), bins=nbins, range=(0, 1), color='y')
    axs[1, 2].hist(test_hls[..., 2].flatten(), bins=nbins, range=(0, 1), color='c')
    axs[2, 0].hist((test_hls[..., 0]).flatten(), bins=nbins_lo, range=(0, 360), color='m')
    axs[2, 1].hist(test_hls[..., 1].flatten(), bins=nbins_lo, range=(0, 1), color='y')
    axs[2, 2].hist(test_hls[..., 2].flatten(), bins=nbins_lo, range=(0, 1), color='c')

    axs[0, 0].set_title('hue')
    axs[0, 1].set_title('lightness')
    axs[0, 2].set_title('saturation')

    axs[1, 0].set_title('hue distribution')
    axs[1, 1].set_title('lightness distribution')
    axs[1, 2].set_title('saturation distribution')

    axs[2, 0].set_title('hue distribution (low resolution)')
    axs[2, 1].set_title('lightness distribution (low resolution)')
    axs[2, 2].set_title('saturation distribution (low resolution)')

    plt.tight_layout()
    sns.despine()