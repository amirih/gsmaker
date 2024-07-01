import argparse
import numpy as np
import pandas as pd
# Non-default
from collections import defaultdict
import sklearn.metrics as skmet
import sklearn.model_selection as skms
from sklearn.tree import DecisionTreeClassifier
import time


def holdout(model, xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : numpy nd-array with shape (n, d)
        Features of the dataset 
    y : 1-array with shape (n, )
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    resultDict: dict
        A Python dictionary with the following 7 keys:
        "trainAUC", "testAUC", "trainAUPRC", "testAUPRC",
        "trainF1", "testF1", and "timeElapsed".
        The values are the floats associated with them.
    """
    timeElapsed = 0
    start = time.time()
    xTrain, xTest, yTrain, yTest = skms.train_test_split(xFeat,
                                                         y,
                                                         test_size=testSize)
    resultDict = eval_dt_perf(model,
                              xTrain,
                              yTrain,
                              xTest,
                              yTest)
    timeElapsed = time.time() - start
    resultDict["timeElapsed"] = timeElapsed
    return resultDict


def kfold_cv(model, xFeat, y, k):
    """
    Split xFeat into k different groups, and then use each of the
    k-folds as a validation set, with the model fitting on the remaining
    k-1 folds. 

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : numpy nd-array with shape (n, d)
        Features of the dataset 
    y : 1-array with shape (n, )
        Labels of the dataset
    k : int
        Number of folds or groups (approximately equal size)

    Returns
    -------
    resultDict: dict
        A Python dictionary with the following 7 keys:
        "trainAUC", "testAUC", "trainAUPRC", "testAUPRC",
        "trainF1", "testF1", and "timeElapsed".
        The values are the floats associated with them.
    """
    # TODO FILL IN
    # initialize a default dict with 0
    resultDict = defaultdict(float) 
    start = time.time()
    kf = skms.KFold(n_splits=k)
    for trainIdx, testIdx in kf.split(xFeat):
        xTrain, xTest = xFeat[trainIdx, :], xFeat[testIdx, :]
        yTrain, yTest = y[trainIdx], y[testIdx]
        foldDict = eval_dt_perf(model,
                                xTrain,
                                yTrain,
                                xTest,
                                yTest)
        for perf_key, perf_val in foldDict.items():
            resultDict[perf_key] += perf_val/k
    timeElapsed = time.time() - start
    resultDict["timeElapsed"] = timeElapsed
    return resultDict


def mc_cv(model, xFeat, y, testSize, s):
    """
    Perform s-samples of the Monte Carlo cross validation 
    approach where for each sample you split xFeat into
    random train and test based on the testSize

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    resultDict: dict
        A Python dictionary with the following 7 keys:
        "trainAUC", "testAUC", "trainAUPRC", "testAUPRC",
        "trainF1", "testF1", and "timeElapsed".
        The values are the floats associated with them.
    """
    # TODO FILL IN
    resultDict = defaultdict(float) 
    timeElapsed = 0
    start = time.time()
    for sample in range(s):
        sampleDict = holdout(model, xFeat, y, testSize)
        for perf_key, perf_val in sampleDict.items():
            resultDict[perf_key] += perf_val/s
    timeElapsed = time.time() - start
    resultDict["timeElapsed"] = timeElapsed
    return resultDict


def eval_dt_perf(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn tree model, train the model using
    the training dataset, and evaluate the model on the
    test dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier object
        An instance of the decision tree classifier 
    xTrain : nd-array with shape (n, d)
        Training data
    yTrain : 1d array with shape (n, )
        Array of labels associated with training data
    xTest : nd-array with shape (m, d)
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    resultDict: dict
        A Python dictionary with the following 6 keys:
        "trainAUC", "testAUC", "trainAUPRC", "testAUPRC",
        "trainF1", "testF1". The values are the floats
        associated with them.
    """
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)

    # calculate AUC
    trainAUC = skmet.roc_auc_score(yTrain, yHatTrain[:, 1])
    testAUC = skmet.roc_auc_score(yTest, yHatTest[:, 1])
    # calculate auprc
    trainAUPRC = skmet.average_precision_score(yTrain,
                                               yHatTrain[:, 1])
    testAUPRC = skmet.average_precision_score(yTest,
                                              yHatTest[:, 1])
    # f1
    trainF1 = skmet.f1_score(yTrain,  model.predict(xTrain))
    testF1 = skmet.f1_score(yTest,  model.predict(xTest))
    return {"trainAUC": trainAUC,
            "testAUC": testAUC, 
            "trainAUPRC": trainAUPRC,
            "testAUPRC": testAUPRC,
            "trainF1": trainF1,
            "testF1": testF1}


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="space_trainx.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="space_trainy.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="space_testx.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="space_testy.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create the decision tree classifier
    dtClass = DecisionTreeClassifier(max_depth=25,
                                     min_samples_leaf=5)
    # simulate 2-fold, 3-fold, 5-fold and 10-fold percentages using holdout
    result_dict = {}
    result_dict["Holdout (0.5)"] = holdout(dtClass, xTrain, yTrain, 0.5)
    result_dict["Holdout (0.33)"] = holdout(dtClass, xTrain, yTrain, 0.33)
    result_dict["Holdout (0.2)"] = holdout(dtClass, xTrain, yTrain, 0.2)
    result_dict["Holdout (0.1)"] = holdout(dtClass, xTrain, yTrain, 0.1)
    # do 2-fold, 3-fold, 5-fold and 10-fold
    result_dict["2-fold"] = kfold_cv(dtClass, xTrain, yTrain, 2)
    result_dict["3-fold"] = kfold_cv(dtClass, xTrain, yTrain, 3)
    result_dict["5-fold"] = kfold_cv(dtClass, xTrain, yTrain, 5)
    result_dict["10-fold"] = kfold_cv(dtClass, xTrain, yTrain, 10)

    # simulate 2-fold, 3-fold, 5-fold and 10-fold using MCCV
    result_dict["MCCV w/ 2"] = mc_cv(dtClass, xTrain, yTrain, 0.5, 2)
    result_dict["MCCV w/ 3"] = mc_cv(dtClass, xTrain, yTrain, 0.33, 3)
    result_dict["MCCV w/ 5"] = mc_cv(dtClass, xTrain, yTrain, 0.2, 5)
    result_dict["MCCV w/ 10"] = mc_cv(dtClass, xTrain, yTrain, 0.1, 10)

    result_dict["True Test"] = eval_dt_perf(dtClass, xTrain, yTrain, xTest, yTest)

    perfDF = pd.DataFrame.from_dict(result_dict, orient='index')
    print(perfDF)


if __name__ == "__main__":
    main()
