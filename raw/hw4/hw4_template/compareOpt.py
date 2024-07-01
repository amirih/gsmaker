import argparse
import numpy as np
import pandas as pd
import timeit
from compareFeat import eval_model, evaluate_unreg

# part of solution
from sklearn.linear_model import SGDRegressor


def evaluate_sgd(trainDF, testDF, mepoch):
    """
    Evaluate the performance of a SGD-based linear regression
    without regularization. Train the regression using the 
    training dataset and evaluate the performance on the test set.

    Parameters
    ----------
    trainDF : pandas.DataFrame
        the training dataframe
    testDF : pandas.DataFrame
        the test dataframe   
    mepoch : int
        Maximum number of epochs

    Returns
    -------
    res : dictionary
        return the dictionary with the following keys --
        r2, mse, time, nfeat
    """
    start = timeit.default_timer()
    time_lr = timeit.default_timer() - start
    return {}


def main():
    """
    Main file to run from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("fullTrain",
                        help="filename of the full-featured training data")
    parser.add_argument("fullTest",
                        help="filename of the full-featured  test data")

    args = parser.parse_args()
    # load the data
    print("Loading data ----")
    full_train = pd.read_csv(args.fullTrain)
    full_test = pd.read_csv(args.fullTest)
    print("Training models now ----")
    perf = {}
    print("Closed Form")
    perf["closed"] = evaluate_unreg(full_train, full_test)
    print("Max Epoch = 1")
    perf["sgd (e=1) r1"] = evaluate_sgd(full_train, full_test, 1)
    perf["sgd (e=1) r2"] = evaluate_sgd(full_train, full_test, 1)
    perf["sgd (e=1) r3"] = evaluate_sgd(full_train, full_test, 1)
    print("Max Epoch = 5")
    perf["sgd (e=5) r1"] = evaluate_sgd(full_train, full_test, 5)
    perf["sgd (e=5) r2"] = evaluate_sgd(full_train, full_test, 5)
    perf["sgd (e=5) r3"] = evaluate_sgd(full_train, full_test, 5)
    print("Max Epoch = 25")
    perf["sgd (e=25) r1"] = evaluate_sgd(full_train, full_test, 25)
    perf["sgd (e=25) r2"] = evaluate_sgd(full_train, full_test, 25)
    perf["sgd (e=25) r3"] = evaluate_sgd(full_train, full_test, 25)

    print(pd.DataFrame.from_dict(perf, orient='index'))


if __name__ == "__main__":
    main()



