import pandas as pd
import numpy as np
import math

def sign(v):
    return 2*(v>=0)-1

def random_booster(x_set, y_set, T):
    mm, nn = x_set.shape
    # the first distribution D1 - the uniform distribution
    p_dist = np.ones((mm, 1))
    p_dist = p_dist / np.sum(p_dist)

    theta = []
    feature_inds = []
    thresholds = []

    for i in range(1, T+1):
        # rand returns a single uniformly distributed random number in the interval (0,1).
        ind = math.ceil(nn*np.random.uniform(0,1))
        # X = randn returns a random scalar drawn from the standard normal distribution.
        thresh = x_set.loc[math.ceil(mm*np.random.uniform(0,1)), ind] + 10**(-8)*np.random.normal()

        Wplus = np.sum(p_dist.T * np.array([sign(x_set.loc[x,ind] - thresh) == y_set[x] for x in range(0,mm)]))
        Wminus = np.sum(p_dist.T * np.array([sign(x_set.loc[x,ind] - thresh) != y_set[x] for x in range(0,mm)]))

        theta += [0.5*math.log(Wplus / Wminus)]
        feature_inds += [ind]
        thresholds += [thresh]
        a0 = np.array(thresholds).T
        # print(sign(x_set.loc[:, feature_inds] - np.tile(a0, (mm, 1))).to_numpy().shape)
        # print(np.array(theta).T.reshape(len(theta),1).shape)
        p_dist = np.dot(sign(x_set.loc[:,feature_inds] - np.tile(a0,(mm,1))), np.array(theta).T.reshape(len(theta),1))
        #print(y_set.to_numpy().reshape(mm,1)*(-1))
        p_dist = y_set.to_numpy().reshape(mm,1)*(-1) * p_dist
        p_dist = np.exp(p_dist)
        print('Iter %d, empirical risk = %1.4f, empirical error = %1.4f' % (i, np.sum(p_dist), sum([e>=1 for e in p_dist.reshape(1,mm)[0]])))
        p_dist = p_dist / np.sum(p_dist)
    return theta, feature_inds, thresholds
