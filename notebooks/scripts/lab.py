import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_lab():
    a = np.repeat(np.linspace(0, 255, 256, dtype=np.uint8), 256, axis=0).reshape(256, 256).T
    b = a.T
    l64 = np.ones((256, 256), dtype=np.uint8) * 64
    l127 = np.ones((256, 256), dtype=np.uint8) * 127
    l224 = np.ones((256, 256), dtype=np.uint8) * 224
    lab64 = np.stack([l64, a, b], axis=2)
    lab127 = np.stack([l127, a, b], axis=2)
    lab224 = np.stack([l224, a, b], axis=2)
    rgb64 = cv2.cvtColor(lab64, cv2.COLOR_LAB2RGB)
    rgb127 = cv2.cvtColor(lab127, cv2.COLOR_LAB2RGB)
    rgb224 = cv2.cvtColor(lab224, cv2.COLOR_LAB2RGB)

    f, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True, sharey=True)
    axs[0].imshow(rgb64)
    axs[0].set_xlabel('a')
    axs[0].set_ylabel('b')
    axs[0].set_title('LAB at L=64')
    axs[0].invert_yaxis()

    axs[1].imshow(rgb127)
    axs[1].set_xlabel('a')
    axs[1].set_ylabel('b')
    axs[1].set_title('LAB at L=127')
    axs[1].invert_yaxis()

    axs[2].imshow(rgb224)
    axs[2].set_xlabel('a')
    axs[2].set_ylabel('b')
    axs[2].set_title('LAB at L=224')
    axs[2].invert_yaxis()

    plt.tight_layout()
    sns.despine()

    return lab127

