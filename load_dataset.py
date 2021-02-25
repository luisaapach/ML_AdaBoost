import pandas as pd
import matplotlib.pyplot as plt


def retrieve_x_y_sets(dataset_type):
    """
    This function retreives training or test set for AdaBoost
    :param dataset_type: 1 if it is for training set, 2 if it is for testing set
    :return:
    """
    if dataset_type == 1:
        dataset = pd.read_csv("boost_data/boosting-train.csv", header=None)
    else:
        dataset = pd.read_csv("boost_data/boosting-test.csv", header=None)
    # dataset.ndim
    return dataset.loc[:, 1:], dataset.loc[:, 0]


x_train, y_train = retrieve_x_y_sets(1)
x_test, y_test = retrieve_x_y_sets(2)

# train boosters
from random_booster import random_booster
from stump_booster import stump_booster

theta_rnd, feature_inds_rnd, thresholds_rnd, gamma_rnd = random_booster(x_train, y_train, 200)
theta_stp, feature_inds_stp, thresholds_stp, gamma_stp = stump_booster(x_train, y_train, 200)

print(theta_rnd, feature_inds_rnd, thresholds_rnd)
print(theta_stp, feature_inds_stp, thresholds_stp)

import plot_error_in_time
fig, (ax1, ax2) = plt.subplots(1,2)
fig.suptitle("Error in time")
print("RANDOM")
ax1, min_train_err_rnd, train_err_rnd, min_test_err_rnd, test_err_rnd = plot_error_in_time.plot_error_in_time(x_train,y_train,x_test,y_test,theta_rnd,feature_inds_rnd,thresholds_rnd, ax1)
print("DECISION STUMPS")
ax2, min_train_err_stp, train_err_stp, min_test_err_stp, test_err_stp = plot_error_in_time.plot_error_in_time(x_train,y_train,x_test,y_test,theta_stp,feature_inds_stp,thresholds_stp, ax2)
ax1.title.set_text('Random boosting error')
ax2.title.set_text('Boosted decision stumps error')
plt.ylim([0.19, 0.5])

test = lambda c: c == min_train_err_rnd
it_train_rnd = list(map(test, train_err_rnd)).index(True)
print("Iteration_RND for minimum {} on TRAIN SET is {}".format(min_train_err_rnd,it_train_rnd))
nearest_neighbour = [abs(x-min_train_err_rnd) for x in train_err_stp]
minimum = min(nearest_neighbour)
it_train_stp = nearest_neighbour.index(minimum)
print("Iteration_DS for aprox minimum {} on TRAIN SET is {}".format(min_train_err_rnd,it_train_stp))
test = lambda c: c == min_test_err_rnd
it_test_rnd = list(map(test, test_err_rnd)).index(True)
print("Iteration_RND for minimum {} on TEST SET is {}".format(min_test_err_rnd,it_test_rnd))
nearest_neighbour = [abs(x-min_test_err_rnd) for x in test_err_stp]
minimum = min(nearest_neighbour)
it_test_stp = nearest_neighbour.index(minimum)
print("Iteration_DS for aprox minimum {} on TEST SET is {}".format(min_test_err_rnd,it_test_stp))

import json
report = {"MIN_TRAIN_ERROR_RND": min_train_err_rnd[0],
          "MIN_TRAIN_ERROR_STP": min_train_err_stp[0],
          "MIN_TEST_ERROR_RND": min_test_err_rnd[0],
          "MIN_TEST_ERROR_STP": min_test_err_stp[0],
          "ITERATION_RND_TRAIN":it_train_rnd,
          "ITERATION_STP_TRAIN":it_train_stp,
          "ITERATION_RND_TEST":it_test_rnd,
          "ITERATION_STP_TEST":it_test_stp}
f=open("REPORT_FULL_PLOT_3.json", "w")
json.dump(report,f,indent=4)
f.close()
plt.show()


plt.plot(range(len(gamma_rnd)), gamma_rnd,label="Gamma RDS")
plt.plot(range(len(gamma_stp)), gamma_stp, label="Gamma BDS")
plt.plot(range(len(theta_stp)),[1/(x_train.shape[0]*2) for _ in range(len(theta_stp))], label = "Gamma = 1/(2*m)")
plt.title("Gamma-weak learnability")
plt.legend()
plt.show()