from typing import Tuple, List

import numpy as np


def build_histogram(window: np.ndarray, binwidth: int = 20, binstep: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    half_binwidth = binwidth // 2
    bins = np.uint32(np.arange(half_binwidth, window.shape[1] - half_binwidth, binstep))
    histogram = np.zeros(shape=len(bins), dtype=np.float32)
    for i, x in enumerate(bins):
        left = max(x - half_binwidth, 0)
        right = min(x + half_binwidth, window.shape[1])
        bin_area = window.shape[0] * (right - left)
        histogram[i] = np.sum(window[:, left:right]) / bin_area
    return histogram, bins


def find_maxima(histogram: np.ndarray, bins: np.ndarray, epsilon: float = .0, k: int = 2) -> Tuple[List[int], List[int]]:
    maxima, heights = [], []
    if histogram[0] > (histogram[1] + epsilon):
        maxima.append(bins[0])
        heights.append(histogram[0])
    for i in range(1, histogram.shape[0] - 1):
        left_smaller = histogram[i] > (histogram[i - 1] + epsilon)
        right_smaller = histogram[i] > (histogram[i + 1] + epsilon)
        if left_smaller and right_smaller:
            maxima.append(bins[i])
            heights.append(histogram[i])
    if histogram[-1] > (histogram[-2] + epsilon):
        maxima.append(bins[-1])
        heights.append(histogram[-1])

    if k == 0:
        k = len(heights)
    idxs = np.array(heights).argsort()[-k:][::-1]
    maxima = np.array(maxima)[idxs].tolist()
    heights = np.array(heights)[idxs].tolist()
    return maxima, heights
