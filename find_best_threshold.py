import math
import numpy as np


def find_best_threshold(x_set, y_set, p_dist):
    mm, nn = x_set.shape
    best_error = math.inf
    ind = 1
    thresh = 0
    for j in range(1, nn + 1):
        x_set_j = np.array(x_set.loc[:, j])  # each column in df
        x_sort = np.asarray(sorted(x_set_j)[::-1])  # sort x values desc
        sorted_indexes = x_set_j.argsort(axis=0)[::-1]
        y_sort = np.asarray(y_set)[np.ix_(sorted_indexes)]  # sort y values according to x
        p_sort = np.asarray(p_dist)[np.ix_(sorted_indexes)]  # sort p values according to x

        possible_thresholds = (x_sort + np.roll(x_sort, 1)) / 2 # jumatatile dintre x-uri
        possible_thresholds[0] = x_sort[0] + 1 # prag exterior = max+1

        increments = np.multiply(p_sort, y_sort.reshape((y_sort.shape[0], 1)))
        increments[0] = 0
        emp_errs = np.ones((mm, 1)) * np.dot(p_sort.T, np.where(y_sort == -1, 0, y_sort))  # replace -1 with 0 -> prima eroare e suma(p_dist_x_poz) - toate gresit clasificate - pe pragul exterior
        emp_errs = emp_errs - np.cumsum(increments).reshape(increments.shape) # -1 | +1

        best_low, thresh_ind = np.min(emp_errs), np.argmin(emp_errs)
        best_high, thresh_high = np.max(emp_errs), np.argmax(emp_errs)
        best_high = 1 - best_high
        best_err_j = min(best_high, best_low)

        if best_high < best_low:
            thresh_ind = thresh_high

        if best_err_j < best_error:
            ind = j
            thresh = possible_thresholds[thresh_ind]
            best_error = best_err_j

    return ind, thresh
