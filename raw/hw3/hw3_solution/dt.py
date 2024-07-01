import argparse
import numpy as np
import pandas as pd
# for solutions only
from scipy import stats
from sklearn.tree import DecisionTreeClassifier


def calculate_score(y, criterion):
    """
    Given a numpy array of labels associated with a node, y, 
    calculate the score based on the crieterion specified.

    Parameters
    ----------
    y : numpy.1d array with shape (n, )
        Array of labels associated with a node
    criterion : String
        The function to measure the quality of a split.
        Supported criteria are "gini" for the Gini impurity
        and "entropy" for the information gain.
    Returns
    -------
    score : float
        The gini or entropy associated with a node
    """
    nTot = len(y)
    # get all the unique values
    classVals = np.unique(y)
    classPor = []
    # tabulate the different portions
    for i in range(len(classVals)):
        classPor.append(float(np.sum(y == classVals[i]))/nTot)
    classPor = np.array(classPor)
    if criterion == "gini":
        return np.sum(np.multiply(classPor, 1-classPor))
    elif criterion == "entropy":
        return -np.sum(np.multiply(classPor, np.log(classPor, where=classPor>0)))
    else:
        raise Exception("The criterion specified not supported")        


### hidden function for solutions only
def _partition_data(xFeat, y, splitVar, splitVal):
    # left side will be <= splitVal
    leftIdx = xFeat[:, splitVar] <= splitVal
    rightIdx = xFeat[:, splitVar] > splitVal
    return xFeat[leftIdx, :], xFeat[rightIdx, :], y[leftIdx], y[rightIdx]


def find_best_splitval(xcol, y, criterion, minLeafSample):
    """
    Given a feature column (i.e., measurements for feature d),
    and the corresponding labels, calculate the best split
    value, v, such that the data will be split into two subgroups:
    xcol <= v and xcol > v. If there is a tie (i.e., multiple values
    that yield the same split score), you can return any of the
    possible values.

    Parameters
    ----------
    xcol : numpy.1d array with shape (n, )
    y : numpy.1d array with shape (n, )
        Array of labels associated with a node
    criterion : string
        The function to measure the quality of a split.
        Supported criteria are "gini" for the Gini impurity
        and "entropy" for the information gain.
    minLeafSample : int
            The min
    Returns
    -------
    v:  float / int (depending on the column)
        The best split value to use to split into 2 subgroups.
    score : float
        The gini or entropy associated with the best split
    """
    # sort the feature to identify splits
    idx = np.argsort(xcol)
    xSorted = xcol[idx]
    ySorted = y[idx]
    bestScore = 10  # set the score to be high above valid ranges
    # set the best split to be none
    bestSplitVal = None
    # get the possible splitvalues 
    uniq_spv, spv_idx = np.unique(xSorted, return_index=True)
    if len(uniq_spv) == 1:
        return bestSplitVal, bestScore
    # check all possible split values
    for splitVal in uniq_spv:
        leftIdx = xSorted <= splitVal
        rightIdx = xSorted > splitVal
        yL = ySorted[leftIdx]
        yR = ySorted[rightIdx]
        # sanity check yL and yR are >= minLeafSample
        if len(yL) < minLeafSample or len(yR) < minLeafSample:
            continue
        sL = calculate_score(yL, criterion)
        sR = calculate_score(yR, criterion)
        score = float(len(yL) / len(y)) * sL + float(len(yR) / len(y)) * sR
        if score < bestScore:
            # update the split
            bestSplitVal = splitVal
            bestScore = score
    return bestSplitVal, bestScore


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion
    model = None       # the tree model

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    # for solutions only
    def is_leaf(self, xFeat, y, depth):
        # if the number of nodes is too small
        if len(y) < self.minLeafSample*2:
            return True
        # check if depth is reached
        elif depth >= self.maxDepth:
            return True
        # if it's pure labels, you should exit
        elif len(np.unique(y)) == 1:
            return True
        else:
            return False

    # for solutions only
    def decision_tree(self, xFeat, y, depth):
        # Stopping criteria
        if self.is_leaf(xFeat, y, depth):  
            return stats.mode(y)[0]
        # find the split
        # go through each feature while keeping track of the best scenario
        # initialize them to the first column + value
        best_var = None
        best_score = 10                                 # set default score
        best_split_val = xFeat[self.minLeafSample, 0]   # default to something
        for split_var in range(xFeat.shape[1]):
            sv, ss = find_best_splitval(xFeat[:, split_var],
                                        y,
                                        self.criterion,
                                        self.minLeafSample)
            if sv is None:
                continue
            if ss < best_score:
                # if it's good then update
                best_split_val = sv
                best_var = split_var
                best_score = ss
        # if None of them are good for splits then exit
        if best_var is None:
            # set it to be a leaf
            return stats.mode(y)[0]
        # Partition data into two sets
        xL, xR, yL, yR = _partition_data(xFeat, y, 
                                         best_var,
                                         best_split_val)
        # Recursive call of decision_tree()
        return {"split_variable": best_var,
                "split_value": best_split_val,
                "right": self.decision_tree(xR,
                                            yR,
                                            depth+1), 
                "left": self.decision_tree(xL,
                                           yL,
                                           depth+1)}

    # for solutions only
    def predict_sample(self, node, x):
        newNode = node['left']
        if x[node['split_variable']] > node['split_value']:
            newNode = node['right']
        # check if it's a dictionary which is another node
        if isinstance(newNode, dict):
            # recursive call of predict_sample
            return self.predict_sample(newNode, x)
        else:
            return newNode

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (n, d)
            Training data 
        y : numpy 1d array with shape (n, )
            Array of labels associated with training data.

        Returns
        -------
        self : DecisionTree
            The decision tree model instance
        """
        self.model = self.decision_tree(xFeat, y, 0)
        return self

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (m, d)
            The data to predict.  

        Returns
        -------
        yHat : numpy 1d array with shape (m, )
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        # for each node do the prediction
        for i in range(xFeat.shape[0]):
            yHat.append(self.predict_sample(self.model,
                                            xFeat[i, :]))
        return np.array(yHat)


def _accuracy(yTrue, yHat):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yTrue : 1d-array with shape (n, )
        True labels associated with the n samples
    yHat : 1d-array with shape (n,)
        Predicted class label for n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    acc = np.sum(yHat == yTrue) / len(yTrue)
    return acc

def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : numpy.nd-array with shape n x d
        Training data 
    yTrain : numpy.1d array with shape n
        Array of labels associated with training data.
    xTest : numpy.nd-array with shape m x d
        Test data 
    yTest : numpy.1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = _accuracy(yTrain, yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = _accuracy(yTest, yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain",
                        default="simxTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="simyTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="simxTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="simyTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)
    # Solution specific
    skdt = DecisionTreeClassifier(max_depth=args.md,
                                  min_samples_leaf=args.mls)
    skdt.fit(xTrain, yTrain)
    trainAcc = _accuracy(yTrain, skdt.predict(xTrain))
    testAcc = _accuracy(yTest, skdt.predict(xTest))
    print("Sklearn Impl ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

if __name__ == "__main__":
    main()
