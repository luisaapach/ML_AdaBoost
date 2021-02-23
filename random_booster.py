import numpy as np
import math


def sign(v):
    return 2 * (v >= 0) - 1


def random_booster(x_set, y_set, T):
    mm, nn = x_set.shape
    # the first distribution D1 - the uniform distribution
    p_dist = np.ones((mm, 1))
    p_dist = p_dist / np.sum(p_dist)

    theta = []
    feature_inds = []
    thresholds = []
    z_product = []

    for i in range(1, T + 1):
        # random.uniform() returns a single uniformly distributed random number in the interval (0,1).
        ind = math.ceil(nn * np.random.uniform(0, 1))
        # random.normal() returns a random scalar drawn from the standard normal distribution.
        item = x_set.loc[math.ceil(mm * np.random.uniform(0, 1)), ind]
        thresh = item + 10 ** (-8) * np.random.normal()

        # sum_{i = 1}^m p(i) * 1{sign(X(i, ind) - thresh) == y(i)}
        Wplus = np.sum(p_dist.T * np.array([sign(x_set.loc[x, ind] - thresh) == y_set[x] for x in range(0, mm)]))
        # sum_{i = 1}^m p(i) * 1{sign(X(i, ind) - thresh) ~= y(i)}
        Wminus = np.sum(p_dist.T * np.array([sign(x_set.loc[x, ind] - thresh) != y_set[x] for x in range(0, mm)]))

        theta += [0.5 * math.log(Wplus / Wminus)]  # the optimal weight for the newest decision stump.
        feature_inds += [ind]
        thresholds += [thresh]
        a0 = np.array(thresholds).T
        # e^(-y_i*f_t(x_i))
        p_dist = np.exp(y_set.to_numpy().reshape(mm, 1) * (-1) *
                        np.dot(sign(x_set.loc[:, feature_inds] - np.tile(a0, (mm, 1))),
                               np.array(theta).T.reshape(len(theta), 1)))
        # z_product += [np.sum(p_dist.reshape(1,mm)[0])*(1/mm) if len(z_product)==0 else np.sum(p_dist.reshape(1,mm)[0])*(1/mm)*z_product[-1]]

        # empirical risk/loss = sum_{i=1}^m e^(-y_i*F(x_i)) - AB tries to minimize it
        print('Iter %d, empirical risk = %1.4f, empirical error = %1.4f' % (
        i, np.sum(p_dist), sum([e >= 1 for e in p_dist.reshape(1, mm)[0]])))
        # unrecursive form for computing the distribution - e^(-y_i*f_t(x_i)) / sum(e^(-y_i*f_t(x_i)))
        p_dist = p_dist / np.sum(p_dist)
    return theta, feature_inds, thresholds


