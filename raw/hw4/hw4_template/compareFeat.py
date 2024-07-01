import argparse
import numpy as np
import pandas as pd
import timeit
from sklearn.metrics import mean_squared_error, r2_score



def eval_model(ytrue, yhat, elapsed, nfeat):
    """
    Calculate the R2 and MSE given the true y and predicted y. It 
    also structures the output to meet the requirements for evaluate_unreg,
    evaluate_lasso, and evaluate_sgd.

    Parameters
    ----------
    ytrue : np.ndarray or pandas.Series
        The true regression label
    yhat : np.ndarray or pandas.Series
        The predicted regression value  
    elapsed : float
        Elapsed time for the model fit function
    nfeat : int / np.integer
        The number of non-zero coefficients        

    Returns
    -------
    res : dictionary
        return the dictionary with the following keys --
        r2, mse, time, nfeat
    """
    return {
        'r2': r2_score(ytrue, yhat),
        'mse': mean_squared_error(ytrue, yhat),
        'time': elapsed,
        'nfeat': nfeat
    }


def evaluate_unreg(trainDF, testDF):
    """
    Evaluate the performance of a closed-form linear regression
    without regularization. Train the regression using the training dataset
    and evaluate the performance on the test set.

    Parameters
    ----------
    trainDF : pandas.DataFrame
        the training dataframe
    testDF : pandas.DataFrame
        the test dataframe   

    Returns
    -------
    res : dictionary
        return the dictionary with the following keys --
        r2, mse, time, nfeat
    """
    start = timeit.default_timer()
    time_lr = timeit.default_timer() - start
    return {}


def numcoef_lasso(lasso):
    """
    Calculate the number of non-zero coefficients in the lasso model

    Parameters
    ----------
    lasso : sklearn.linear_model
        An Sklearn regression model  

    Returns
    -------
    ncoef : int
        return the number of non-zero coefficients in the model.
    """
    return int(np.sum(lasso.coef_ != 0))


def evaluate_lasso(trainDF, testDF, alphaList):
    """
    Evaluate the performance of a LASSO-regularized linear regression
    model. The method should perform GridSearchCV on the passed in
    alphaList to choose the optimal alpha. It should then train the lasso
    model using the training dataset and this optimal alpha,
    and evaluate the performance on the test set.

    Parameters
    ----------
    trainDF : pandas.DataFrame
        the training dataframe
    testDF : pandas.DataFrame
        the test dataframe  
    alphaList : list
        List of alpha parameters to run GridSearchCV against    

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
    parser.add_argument("fsTrain",
                        help="filename of the feature selected training data")
    parser.add_argument("fsTest",
                        help="filename of the feature selected test data")

    args = parser.parse_args()
    # load the 2 sets of train and test data
    full_train = pd.read_csv(args.fullTrain)
    full_test = pd.read_csv(args.fullTest)
    fs_train = pd.read_csv(args.fsTrain)
    fs_test = pd.read_csv(args.fsTest)
    perf = {}
    perf["full"] = evaluate_unreg(full_train, full_test)
    perf["feat-sel"] = evaluate_unreg(fs_train, fs_test)
    # a sample of alpha list to try
    perf["lasso"] = evaluate_lasso(full_train, full_test,
                                   [0.00001, 0.0001, 0.001, 0.01])

    print(pd.DataFrame.from_dict(perf, orient='index'))


if __name__ == "__main__":
    main()



