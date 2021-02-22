import pandas
import math
import numpy as np


def find_best_threshold(x_set, y_set, p_dist):
    mm, nn = x_set.shape
    best_error = math.inf
    ind = 1
    thresh = 0
    for j in range(1, nn + 1):
        # print(x_set.loc[:, j])
        x_set_j = np.array(x_set.loc[:, j])
        x_sort = np.asarray(sorted(x_set_j)[::-1])
        inds = x_set_j.argsort(axis=0)[::-1]
        y_sort = np.asarray(y_set)[np.ix_(inds)]
        p_sort = np.asarray(p_dist)[np.ix_(inds)]
        s = x_sort[0] + 1
        possible_thresholds = x_sort
        possible_thresholds = (x_sort + np.roll(x_sort, 1)) / 2
        # print(possible_thresholds)
        # print(type(x_sort))
        # print(type(inds))
        # print(y_sort)
        possible_thresholds[0] = x_sort[0] + 1
        increments = np.roll(p_sort * y_sort, 1)
        increments[0] = 0
        emp_errs = np.ones(mm, 1) * (p_sort.T * int(y_sort == 1))
        # print(emp_errs)
        emp_errs = emp_errs - np.cumsum(increments)

