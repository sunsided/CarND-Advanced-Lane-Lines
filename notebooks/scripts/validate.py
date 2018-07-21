import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable


def color_mask(img, mask):
    return img * (np.stack([mask]*3, axis=2) > 0)


def display_with_mask(img, mask):
    f, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    axs[0].imshow(img)
    axs[1].imshow(color_mask(img, mask))
    plt.tight_layout()
    sns.despine()


def test_processing(train_images, func: Callable[[np.ndarray], np.ndarray]):
    f, axs = plt.subplots(nrows=len(train_images), ncols=2, figsize=(13, 3*len(train_images)))

    for i in range(0, len(train_images)):
        img = cv2.cvtColor(train_images[i], cv2.COLOR_BGR2RGB)
        other = func(img)
        axs[i, 0].imshow(img, cmap='gray')
        axs[i, 1].imshow(other, cmap='gray')

    plt.tight_layout()
    sns.despine()


def test_mask(train_images, func: Callable[[np.ndarray], np.ndarray]):
    f, axs = plt.subplots(nrows=len(train_images), ncols=2, figsize=(13, 3*len(train_images)))

    for i in range(0, len(train_images)):
        img = cv2.cvtColor(train_images[i], cv2.COLOR_BGR2RGB)
        other = func(img)
        other = color_mask(img, other)
        axs[i, 0].imshow(img, cmap='gray')
        axs[i, 1].imshow(other, cmap='gray')

    plt.tight_layout()
    sns.despine()
