import numpy as np
from find_best_threshold import find_best_threshold
import math


def sign(v):
    return 2 * (v >= 0) - 1


def stump_booster(x_set, y_set, T):
    mm, nn = x_set.shape
    # the first distribution D1 - the uniform distribution
    p_dist = np.ones((mm, 1))
    p_dist = p_dist / np.sum(p_dist)

    theta, feature_inds, thresholds = [], [], []

    for i in range(1, T + 1):
        ind, thresh = find_best_threshold(x_set, y_set, p_dist)

        w_plus = np.sum(p_dist.T * np.array([sign(x_set.loc[x, ind] - thresh) == y_set[x] for x in range(0, mm)]))
        w_minus = np.sum(p_dist.T * np.array([sign(x_set.loc[x, ind] - thresh) != y_set[x] for x in range(0, mm)]))

        theta += [0.5 * math.log(w_plus / w_minus)]
        feature_inds += [ind]
        thresholds += [thresh]

        p_dist = np.dot(sign(x_set.loc[:, feature_inds] - np.tile(np.array(thresholds).T, (mm, 1))), np.array(theta).T.reshape(len(theta), 1))
        p_dist = y_set.to_numpy().reshape(mm, 1) * (-1) * p_dist
        p_dist = np.exp(p_dist)
        print('Iter %d, empirical risk = %1.4f, empirical error = %1.4f' % (i, np.sum(p_dist), sum([e >= 1 for e in p_dist.reshape(1, mm)[0]])))
        p_dist = p_dist / np.sum(p_dist)

    return theta, feature_inds, thresholds
