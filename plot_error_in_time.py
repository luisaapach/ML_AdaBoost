import numpy as np
import matplotlib.pyplot as plt

def sign(v):
    return 2*(v>=0)-1

def plot_error_in_time(Xtrain, ytrain, Xtest, ytest, theta, feature_inds, thresholds):
    number_of_thresholds = len(thresholds)

    train_errors = np.zeros((number_of_thresholds,1))
    test_errors = np.zeros((number_of_thresholds, 1))

    mtrain = Xtrain.shape[0]
    mtest = Xtest.shape[0]

    #Predicted margins for train and test
    train_predictions = np.zeros((mtrain,1))
    test_predictions = np.zeros((mtest,1))

    # Iteratively compute the margin predicted by the
    # thresholded classifier, updating both test and training predictions.

    for i in range(0,number_of_thresholds):
        train_predictions += (sign(Xtrain.loc[:,feature_inds[i]].to_numpy() - thresholds[i]) * theta[i]).reshape((mtrain,1))
        test_predictions += (sign(Xtest.loc[:,feature_inds[i]].to_numpy() - thresholds[i]) * theta[i]).reshape((mtest,1))


        train_errors[i] = sum([el[0]<=0 for el in ytrain.to_numpy().reshape((mtrain,1)) * train_predictions]) / mtrain
        test_errors[i] = sum([el[0] <= 0 for el in ytest.to_numpy().reshape((mtest,1)) * test_predictions]) / mtest

    plt.plot(range(len(train_errors)),train_errors, label = "Train error rate")
    plt.plot(range(len(test_errors)),test_errors, label = "Test error rate")
    plt.xlabel('Iterations')
    plt.ylabel('Error rate')
    plt.legend()
    plt.show()