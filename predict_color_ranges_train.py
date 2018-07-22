"""
This script is used to sample yellow and white lane marker colors from images
given their overall appearance.
"""

import csv
import glob

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib


NBINS = 16
BATCH_SIZE = 0
EPOCHS = 500
INCLUDE_ACTUAL_SAMPLES = False


def main():
    Xs = []
    Ys = []

    for path in glob.glob('lane-samples-*.csv'):
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            histogram_fields = [f for f in reader.fieldnames if f.startswith('h')]

            for row in reader:
                is_white = row['white']
                hist = np.array([row[k] for k in histogram_fields])
                yuv_mean = np.array([row['y_mean'], row['u_mean'], row['v_mean']])
                yuv_std = np.array([row['y_std'], row['u_std'], row['v_std']])

                Xs.append(np.hstack([is_white, hist]))
                Ys.append(np.hstack([yuv_mean, yuv_std]))

                if INCLUDE_ACTUAL_SAMPLES:
                    yuv = np.array([row['y'], row['u'], row['v']])
                    Xs.append(np.hstack([is_white, hist]))
                    Ys.append(np.hstack([yuv, yuv_std]))

    Xs = np.array(Xs).astype(np.float32)
    Ys = np.array(Ys).astype(np.float32)

    # Normalize the values
    hist_mean = np.mean(Xs[:, 1:], axis=0)
    hist_std = np.std(Xs[:, 1:], axis=0)
    yuv_mean = np.mean(Ys, axis=0)
    yuv_std = np.std(Ys, axis=0)

    mlp = MLPRegressor(hidden_layer_sizes=(32, 16),
                       activation='logistic',
                       solver='lbfgs',
                       alpha=0.1,
                       warm_start=True,
                       max_iter=10,
                       verbose=False)
    for e in range(EPOCHS):
        count = Xs.shape[0]
        batch_size = BATCH_SIZE if BATCH_SIZE > 0 else count
        for i in range(0, count // batch_size):
            indexes = np.random.choice(range(count), batch_size, replace=False)
            Xs_, Ys_ = Xs[indexes, ...].copy(), Ys[indexes, ...].copy()
            Xs_[:, 1:] += np.random.rand(Xs_.shape[0], 3 * NBINS) * 0.01
            Xs_[:, 1:] = (Xs_[:, 1:] - hist_mean) / hist_std

            Ys_ += np.random.rand(Ys_.shape[0], Ys_.shape[1]) * 0.0001
            Ys_ = (Ys_ - yuv_mean) / yuv_std

            mlp.fit(Xs_, Ys_)
            print('Epoch {}, loss: {}'.format(e, mlp.loss_))

    params = dict(hist_mean=hist_mean, hist_std=hist_std,
                  yuv_mean=yuv_mean, yuv_std=yuv_std,
                  nbins=NBINS,
                  mlp=mlp)
    joblib.dump(params, 'color_ranges_model.pkl')


if __name__ == '__main__':
    main()
