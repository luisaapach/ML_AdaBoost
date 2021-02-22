import pandas as pd


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

# train boosters
import random_booster
theta_rnd, feature_inds_rnd, thresholds_rnd = random_booster.random_booster(x_train, y_train,50)
print(theta_rnd, feature_inds_rnd, thresholds_rnd)