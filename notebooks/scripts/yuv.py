import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_yuv():
    u = np.repeat(np.linspace(0, 255, 256, dtype=np.uint8), 256, axis=0).reshape(256, 256).T
    v = u.T
    y64 = np.ones((256, 256), dtype=np.uint8) * 64
    y127 = np.ones((256, 256), dtype=np.uint8) * 127
    y224 = np.ones((256, 256), dtype=np.uint8) * 224
    yuv64 = np.stack([y64, u, v], axis=2)
    yuv127 = np.stack([y127, u, v], axis=2)
    yuv224 = np.stack([y224, u, v], axis=2)
    rgb64 = cv2.cvtColor(yuv64, cv2.COLOR_YUV2RGB)
    rgb127 = cv2.cvtColor(yuv127, cv2.COLOR_YUV2RGB)
    rgb224 = cv2.cvtColor(yuv224, cv2.COLOR_YUV2RGB)

    f, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True, sharey=True)
    axs[0].imshow(rgb64)
    axs[0].set_xlabel('u')
    axs[0].set_ylabel('v')
    axs[0].set_title('YUV at Y=64')
    axs[0].invert_yaxis()

    axs[1].imshow(rgb127)
    axs[1].set_xlabel('u')
    axs[1].set_ylabel('v')
    axs[1].set_title('YUV at Y=127')
    axs[1].invert_yaxis()

    axs[2].imshow(rgb224)
    axs[2].set_xlabel('u')
    axs[2].set_ylabel('v')
    axs[2].set_title('YUV at Y=224')
    axs[2].invert_yaxis()

    plt.tight_layout()
    sns.despine()

    return yuv127

