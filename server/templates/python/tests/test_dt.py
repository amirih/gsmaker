import numpy as np
import numpy.testing as npt
import pandas as pd
import unittest
from gradescope_utils.autograder_utils.decorators import (
    weight,
    number,
    visibility,
)
import sklearn.datasets as skd
import sklearn.model_selection as skms
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from dt import DecisionTree, dt_train_test, calculate_score, find_best_splitval


def _calc_gini(y):
    n = len(y)
    y_1 = float(sum(y))
    y_0 = float(n - y_1)
    yprob = np.array([y_0 / n, y_1 / n])
    return np.sum(np.multiply(yprob, 1 - yprob))


def _calc_entropy(y, logbase=2):
    n = len(y)
    y_1 = float(sum(y))
    y_0 = float(n - y_1)
    yprob = np.array([y_0 / n, y_1 / n])
    if logbase == 2:
        return -np.sum(np.multiply(yprob, np.log2(yprob, where=yprob > 0)))
    elif logbase == 10:
        return -np.sum(np.multiply(yprob, np.log10(yprob, where=yprob > 0)))
    else:
        return -np.sum(np.multiply(yprob, np.log(yprob, where=yprob > 0)))


class TestDT(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        iris_data = skd.load_iris()
        iris_x = iris_data.data
        iris_y = iris_data.target
        # turn into binary classification
        mask = iris_y != 2
        iris_x = iris_x[mask, :]
        iris_y = iris_y[mask]
        # split data into train test
        cls.train_x, cls.test_x, cls.train_y, cls.test_y = (
            skms.train_test_split(
                iris_x, iris_y, test_size=0.3, random_state=20
            )
        )
        cls.space_trainx = pd.read_csv("data/space_trainx.csv").to_numpy()
        cls.space_trainy = (
            pd.read_csv("data/space_trainy.csv").to_numpy().flatten()
        )
        cls.space_testx = pd.read_csv("data/space_testx.csv").to_numpy()
        cls.space_testy = (
            pd.read_csv("data/space_testy.csv").to_numpy().flatten()
        )

        rng = np.random.default_rng(334)
        xFeat1 = rng.integers(low=0, high=10, size=(50, 1))
        xFeat2 = rng.integers(low=15, high=25, size=(50, 1))
        y1 = np.zeros(50, dtype=int)
        y2 = np.ones(50, dtype=int)
        cls.simple_x = np.vstack((xFeat1, xFeat2))
        cls.simple_y = np.hstack((y1, y2))

    @weight(1)
    @number("1.A1")
    def test_giniA(self):
        score = calculate_score(self.train_y, "gini")
        self.assertTrue(score <= 1 and score >= 0, "Incorrect Gini range")

    @weight(1)
    @number("1.A2")
    def test_entropyA(self):
        score = calculate_score(self.train_y, "entropy")
        self.assertTrue(score <= 1 and score >= 0, "Incorrect Entropy range")

    @weight(1)
    @visibility("after_published")
    @number("1.A3")
    def test_giniB(self):
        y_tmp = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
        true_score = _calc_gini(y_tmp)
        score = calculate_score(y_tmp, "gini")
        npt.assert_almost_equal(
            score, true_score, decimal=5, err_msg="Incorrect Gini value"
        )

    @weight(1)
    @visibility("after_published")
    @number("1.A4")
    def test_giniC(self):
        y_tmp = np.zeros(20)
        true_score = _calc_gini(y_tmp)
        score = calculate_score(y_tmp, "gini")
        npt.assert_almost_equal(
            score, true_score, decimal=5, err_msg="Incorrect Gini value"
        )

    @weight(1)
    @visibility("after_published")
    @number("1.A5")
    def test_entropyB(self):
        y_tmp = np.zeros(20)
        true_score = _calc_entropy(y_tmp, logbase=-1)
        score = calculate_score(y_tmp, "entropy")
        npt.assert_almost_equal(
            score, true_score, decimal=5, err_msg="Incorrect Entropy value"
        )

    @weight(1)
    @visibility("after_published")
    @number("1.A6")
    def test_entropyC(self):
        y_tmp = np.random.choice([0, 1], size=20, p=[0.7, 0.3])
        true_score = _calc_entropy(y_tmp, logbase=-1)
        score = calculate_score(y_tmp, "entropy")
        try:
            npt.assert_almost_equal(
                score, true_score, decimal=5, err_msg="Incorrect Entropy value"
            )
        except AssertionError:
            try:
                true_score = _calc_entropy(y_tmp, logbase=2)
                npt.assert_almost_equal(
                    score,
                    true_score,
                    decimal=5,
                    err_msg="Incorrect Entropy value",
                )
            except AssertionError:
                true_score = _calc_entropy(y_tmp, logbase=10)
                npt.assert_almost_equal(
                    score,
                    true_score,
                    decimal=5,
                    err_msg="Incorrect Entropy value",
                )

    @weight(1)
    @number("1.B1")
    def test_findbest1(self):
        v, score = find_best_splitval(
            np.random.rand(100),
            np.random.choice([0, 1], size=100, p=[0.7, 0.3]),
            "gini",
            1,
        )
        self.assertIsInstance(
            v,
            float,
            "v is not a float for passed in float column with gini as criteria",
        )
        self.assertIsInstance(score, float, "returned score is not a float")

    @weight(1)
    @number("1.B2")
    def test_findbest2(self):
        v, score = find_best_splitval(
            np.random.rand(100),
            np.random.choice([0, 1], size=100, p=[0.7, 0.3]),
            "entropy",
            4,
        )
        self.assertIsInstance(
            v, float, "v is not a float for passed in float column"
        )
        self.assertIsInstance(score, float, "returned score is not a float")

    @weight(1)
    @number("1.B3")
    def test_findbest3(self):
        xFeat = np.random.rand(100)
        v, score = find_best_splitval(
            xFeat,
            np.random.choice([0, 1], size=100, p=[0.7, 0.3]),
            "entropy",
            4,
        )
        self.assertTrue(v in xFeat, "Returns a split value not in the feature")

    @weight(1)
    @number("1.B4")
    def test_findbest4(self):
        xFeat = np.random.rand(100)
        v, score = find_best_splitval(
            xFeat,
            np.random.choice([0, 1], size=100, p=[0.7, 0.3]),
            "entropy",
            4,
        )
        self.assertTrue(score <= 1 and score >= 0, "Incorrect Entropy range")

    @weight(1)
    @number("1.B5")
    def test_findbest5(self):
        xFeat = np.random.rand(100)
        v, score = find_best_splitval(
            xFeat, np.random.choice([0, 1], size=100, p=[0.7, 0.3]), "gini", 4
        )
        self.assertTrue(score <= 1 and score >= 0, "Incorrect gini range")

    @weight(1)
    @visibility("after_published")
    @number("1.B6")
    def test_findbest6(self):
        rng = np.random.default_rng(334)
        xFeat = rng.random(100)
        y = rng.choice([0, 1], size=100, p=[0.7, 0.3])
        v, score = find_best_splitval(xFeat, y, "gini", 5)
        # given v, verify it does t
        leftIdx = xFeat <= v
        rightIdx = xFeat > v
        # use the gini score from their implementation
        sL = calculate_score(y[leftIdx], "gini")
        sR = calculate_score(y[rightIdx], "gini")
        true_score = (
            float(len(y[leftIdx]) / len(y)) * sL
            + float(len(y[rightIdx]) / len(y)) * sR
        )
        npt.assert_almost_equal(
            score,
            true_score,
            decimal=4,
            err_msg="Returned Gini score is inconsistent with the calculate split",
        )

    @weight(1)
    @visibility("after_published")
    @number("1.B7")
    def test_findbest7(self):
        xFeat = np.random.rand(100)
        y = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
        v, score = find_best_splitval(xFeat, y, "entropy", 5)
        # given v, verify it does t
        leftIdx = xFeat <= v
        rightIdx = xFeat > v
        # use the gini score from their implementation
        sL = calculate_score(y[leftIdx], "entropy")
        sR = calculate_score(y[rightIdx], "entropy")
        true_score = (
            float(len(y[leftIdx]) / len(y)) * sL
            + float(len(y[rightIdx]) / len(y)) * sR
        )
        npt.assert_almost_equal(
            score,
            true_score,
            decimal=4,
            err_msg="Returned entropy score is inconsistent with the calculate split",
        )

    @weight(1)
    @visibility("after_published")
    @number("1.B8")
    def test_findbest8(self):
        xFeat = np.random.rand(100)
        y = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
        mls = 10
        v, score = find_best_splitval(xFeat, y, "gini", mls)
        # given v, verify it does t
        leftIdx = xFeat <= v
        rightIdx = xFeat > v
        self.assertGreaterEqual(
            np.sum(leftIdx), mls, "Left split is less than minimum leaf sample"
        )
        self.assertGreaterEqual(
            np.sum(rightIdx),
            mls,
            "Right split is less than minimum leaf sample",
        )

    @weight(1)
    @visibility("after_published")
    @number("1.B9")
    def test_findbest9(self):
        xFeat = np.random.rand(100)
        y = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
        mls = 10
        v, score = find_best_splitval(xFeat, y, "entropy", mls)
        # given v, verify it does t
        leftIdx = xFeat <= v
        rightIdx = xFeat > v
        self.assertGreaterEqual(
            np.sum(leftIdx), mls, "Left split is less than minimum leaf sample"
        )
        self.assertGreaterEqual(
            np.sum(rightIdx),
            mls,
            "Right split is less than minimum leaf sample",
        )

    @weight(1)
    @visibility("after_published")
    @number("1.B10")
    def test_findbest10(self):
        xFeat1 = np.random.randint(low=0, high=10, size=50)
        xFeat2 = np.random.randint(low=15, high=25, size=50)
        y1 = np.zeros(50)
        y2 = np.ones(50)
        xFeat = np.hstack((xFeat1, xFeat2))
        y = np.hstack((y1, y2))
        v, score = find_best_splitval(xFeat, y, "gini", 5)
        # verify v should be between 10 and 15
        self.assertLessEqual(v, 15, "v is incorrect value")

    @weight(1)
    @visibility("after_published")
    @number("1.B11")
    def test_findbest11(self):
        xFeat1 = np.random.randint(low=0, high=10, size=50)
        xFeat2 = np.random.randint(low=15, high=25, size=50)
        y1 = np.zeros(50)
        y2 = np.ones(50)
        xFeat = np.hstack((xFeat1, xFeat2))
        y = np.hstack((y1, y2))
        v, score = find_best_splitval(xFeat, y, "entropy", 5)
        # verify v should be between 10 and 15
        self.assertLessEqual(v, 15, "v is incorrect value")

    @weight(1)
    @visibility("after_published")
    @number("1.B12")
    def test_findbest12(self):
        xFeat1 = np.random.randint(low=0, high=10, size=95)
        xFeat2 = np.random.randint(low=15, high=25, size=5)
        y1 = np.zeros(95)
        y2 = np.ones(5)
        xFeat = np.hstack((xFeat1, xFeat2))
        y = np.hstack((y1, y2))
        v, score = find_best_splitval(xFeat, y, "gini", 10)
        # verify v should be between 10 and 15
        self.assertLessEqual(v, 10, "v is incorrect value")

    @weight(1)
    @number("1.C1")
    def test_train1(self):
        # test if train method runs with gini
        xFeat = np.random.rand(1000, 4)
        y = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
        maxDepth = 3
        minLeaf = 1
        dct = DecisionTree("gini", maxDepth, minLeaf)
        dct_obj = dct.train(xFeat, y)
        self.assertEqual(dct, dct_obj, "Does not return itself")
        self.assertEqual(dct.maxDepth, maxDepth, "Wrong max Depth")
        self.assertEqual(dct.minLeafSample, minLeaf, "Wrong min leaf sample")

    @weight(1)
    @number("1.C2")
    def test_train2(self):
        # test if train method runs with entropy
        xFeat = np.random.rand(1000, 4)
        y = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
        maxDepth = 3
        minLeaf = 1
        dct = DecisionTree("entropy", maxDepth, minLeaf)
        dct_obj = dct.train(xFeat, y)
        self.assertEqual(dct, dct_obj, "Does not return itself")
        self.assertEqual(dct.maxDepth, maxDepth, "Wrong max Depth")
        self.assertEqual(dct.minLeafSample, minLeaf, "Wrong min leaf sample")

    @weight(1)
    @visibility("after_published")
    @number("1.C3")
    def test_train3(self):
        # test if train method runs with iris and gini
        maxDepth = 5
        minLeaf = 8
        dct = DecisionTree("gini", maxDepth, minLeaf)
        dct.train(self.train_x, self.train_y)
        self.assertEqual(dct.maxDepth, maxDepth, "Wrong max Depth")
        self.assertEqual(dct.minLeafSample, minLeaf, "Wrong max Depth")

    @weight(1)
    @visibility("after_published")
    @number("1.C4")
    def test_train4(self):
        # test if train method runs with iris and entropy
        maxDepth = 4
        minLeaf = 10
        dct = DecisionTree("entropy", maxDepth, minLeaf)
        dct.train(self.train_x, self.train_y)
        self.assertEqual(dct.maxDepth, maxDepth, "Wrong max Depth")
        self.assertEqual(
            dct.minLeafSample, minLeaf, "Wrong mininum leaf sample"
        )

    @weight(1)
    @visibility("after_published")
    @number("1.C5")
    def test_train5(self):
        maxDepth = 15
        minLeaf = 5
        dct = DecisionTree("gini", maxDepth, minLeaf)
        dct.train(self.space_trainx, self.space_trainy)
        self.assertEqual(dct.maxDepth, maxDepth, "Wrong max Depth")
        self.assertEqual(
            dct.minLeafSample, minLeaf, "Wrong mininum leaf sample"
        )

    @weight(1)
    @visibility("after_published")
    @number("1.C6")
    def test_train6(self):
        maxDepth = 10
        minLeaf = 15
        dct = DecisionTree("entropy", maxDepth, minLeaf)
        dct.train(self.space_trainx, self.space_trainy)
        self.assertEqual(dct.maxDepth, maxDepth, "Wrong max Depth")
        self.assertEqual(
            dct.minLeafSample, minLeaf, "Wrong mininum leaf sample"
        )

    @weight(1)
    @number("1.D1")
    def test_predict1(self):
        # test if train method runs with gini
        xFeat = np.random.rand(1000, 4)
        y = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
        maxDepth = 3
        minLeaf = 1
        dct = DecisionTree("gini", maxDepth, minLeaf)
        dct_obj = dct.train(xFeat, y)
        ntest = 200
        y_pred = dct.predict(np.random.rand(ntest, 4))
        self.assertIsInstance(y_pred, np.ndarray, "Is not a numpy array")

    @weight(1)
    @number("1.D2")
    def test_predict2(self):
        # test if train method runs with gini
        rng = np.random.default_rng(334)
        xFeat = rng.random((1000, 4))
        y = rng.choice([0, 1], size=1000, p=[0.7, 0.3])
        maxDepth = 3
        minLeaf = 1
        dct = DecisionTree("entropy", maxDepth, minLeaf)
        dct_obj = dct.train(xFeat, y)
        ntest = 200
        xTest = rng.random((ntest, 4))
        y_pred = dct.predict(xTest)
        self.assertEqual(y_pred.ndim, 1, "Not a 1d array")
        self.assertEqual(
            y_pred.shape[0],
            ntest,
            "Predict method returns wrong number of predictions",
        )

    @weight(1)
    @number("1.D3")
    def test_predict3(self):
        maxDepth = 5
        minLeaf = 5
        dct = DecisionTree("gini", maxDepth, minLeaf)
        dct_obj = dct.train(self.simple_x, self.simple_y)
        # predict itself, should be simple tree
        y_pred = dct.predict(self.simple_x)
        # check if labels are binary
        self.assertEqual(
            len(np.unique(np.array(y_pred))),
            2,
            "Labels are not binary for gini criteria",
        )

    @weight(1)
    @number("1.D4")
    def test_predict4(self):
        maxDepth = 5
        minLeaf = 5
        dct = DecisionTree("entropy", maxDepth, minLeaf)
        dct_obj = dct.train(self.simple_x, self.simple_y)
        # predict itself, should be simple tree
        y_pred = dct.predict(self.simple_x)
        # check if labels are binary
        self.assertEqual(
            len(np.unique(np.array(y_pred))),
            2,
            "Labels are not binary for entropy criteria",
        )

    @weight(1)
    @visibility("after_published")
    @number("1.D5")
    def test_predict5(self):
        maxDepth = 5
        minLeaf = 5
        dct = DecisionTree("gini", maxDepth, minLeaf)
        dct_obj = dct.train(self.simple_x, self.simple_y)
        # predict itself, should be simple tree
        y_pred = dct.predict(self.simple_x)
        # prediction should be the same as input for this simple case!
        npt.assert_almost_equal(
            y_pred,
            self.simple_y,
            decimal=0,
            err_msg="Prediction for simple 1d case is incorrect",
        )

    @weight(1)
    @visibility("after_published")
    @number("1.D6")
    def test_predict6(self):
        maxDepth = 5
        minLeaf = 10
        dct = DecisionTree("entropy", maxDepth, minLeaf)
        dct_obj = dct.train(self.simple_x, self.simple_y)
        # predict itself, should be simple tree
        y_pred = dct.predict(self.simple_x)
        # prediction should be the same as input for this simple case!
        npt.assert_almost_equal(
            y_pred,
            self.simple_y,
            decimal=0,
            err_msg="Prediction for simple 1d case is incorrect",
        )

    @weight(1)
    @visibility("after_published")
    @number("1.D7")
    def test_predict7(self):
        maxDepth = 5
        minLeaf = 5
        dct = DecisionTree("gini", maxDepth, minLeaf)
        # add noise to dimension 2
        xfeat = np.hstack(
            (
                self.simple_x,
                np.random.rand(self.simple_x.shape[0], self.simple_x.shape[1]),
            )
        )
        dct_obj = dct.train(xfeat, self.simple_y)
        # predict itself, should be simple tree
        y_pred = dct.predict(xfeat)
        # prediction should be the same as input for this simple case!
        npt.assert_almost_equal(
            y_pred,
            self.simple_y,
            decimal=0,
            err_msg="Prediction for simple 1d case is incorrect",
        )

    @weight(1)
    @number("1.D8")
    def test_predict8(self):
        maxDepth = 5
        minLeaf = 5
        dct = DecisionTree("entropy", maxDepth, minLeaf)
        # add noise to dimension 2
        xfeat = np.hstack(
            (
                self.simple_x,
                np.random.rand(self.simple_x.shape[0], self.simple_x.shape[1]),
            )
        )
        dct_obj = dct.train(xfeat, self.simple_y)
        # predict itself, should be simple tree
        y_pred = dct.predict(xfeat)
        # prediction should be the same as input for this simple case!
        npt.assert_almost_equal(
            y_pred,
            self.simple_y,
            decimal=0,
            err_msg="Prediction for simple 1d case is incorrect",
        )

    @weight(1)
    @visibility("after_published")
    @number("1.D9")
    def test_predict9(self):
        maxDepth = 2
        minLeaf = 5
        dct = DecisionTree("gini", maxDepth, minLeaf)
        dct.train(self.train_x, self.train_y)
        yhat = dct.predict(self.train_x)
        sk_dt = DecisionTreeClassifier(
            criterion="gini",
            max_depth=maxDepth,
            min_samples_leaf=minLeaf,
            random_state=10,
        )
        sk_dt.fit(self.train_x, self.train_y)
        yhat_true = sk_dt.predict(self.train_x)
        npt.assert_equal(yhat, yhat_true, "unexpected predicted values")

    @weight(1)
    @visibility("after_published")
    @number("1.D10")
    def test_predict10(self):
        maxDepth = 2
        minLeaf = 5
        dct = DecisionTree("gini", maxDepth, minLeaf)
        dct.train(self.train_x, self.train_y)
        yhat = dct.predict(self.train_x)
        sk_dt = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=maxDepth,
            min_samples_leaf=minLeaf,
            random_state=10,
        )
        sk_dt.fit(self.train_x, self.train_y)
        yhat_true = sk_dt.predict(self.train_x)
        npt.assert_equal(yhat, yhat_true, "unexpected predicted values")

    @weight(1)
    @visibility("after_published")
    @number("1.D11")
    def test_predict11(self):
        # test dt accuracy against sklearn dt
        maxDepth = 10
        minLeaf = 20
        ctn = "gini"
        dct = DecisionTree(ctn, maxDepth, minLeaf)
        trainAcc_s, testAcc_s = dt_train_test(
            dct,
            self.space_trainx,
            self.space_trainy,
            self.space_testx,
            self.space_testy,
        )
        sk_dt = DecisionTreeClassifier(
            criterion=ctn,
            max_depth=maxDepth,
            min_samples_leaf=minLeaf,
            random_state=0,
        )
        sk_dt.fit(self.space_trainx, self.space_trainy)
        yHatTrain = sk_dt.predict(self.space_trainx)
        trainAcc = accuracy_score(self.space_trainy, yHatTrain)
        # check if accuracies are within reasonable range
        self.assertTrue(
            abs(trainAcc - trainAcc_s) < 0.15,
            "Unexpected low training accuracy",
        )

    @weight(1)
    @visibility("after_published")
    @number("1.D12")
    def test_predict12(self):
        # test dt accuracy against sklearn dt
        maxDepth = 10
        minLeaf = 20
        ctn = "entropy"
        dct = DecisionTree(ctn, maxDepth, minLeaf)
        trainAcc_s, testAcc_s = dt_train_test(
            dct,
            self.space_trainx,
            self.space_trainy,
            self.space_testx,
            self.space_testy,
        )
        sk_dt = DecisionTreeClassifier(
            criterion=ctn,
            max_depth=maxDepth,
            min_samples_leaf=minLeaf,
            random_state=0,
        )
        sk_dt.fit(self.space_trainx, self.space_trainy)
        yHatTrain = sk_dt.predict(self.space_trainx)
        trainAcc = accuracy_score(self.space_trainy, yHatTrain)
        # check if accuracies are within reasonable range
        self.assertTrue(abs(trainAcc - trainAcc_s) < 0.15, "Low train accuracy")

    @weight(1)
    @visibility("after_published")
    @number("1.D13")
    def test_predict13(self):
        # test dt accuracy against sklearn dt
        maxDepth = 10
        minLeaf = 20
        ctn = "gini"
        dct = DecisionTree(ctn, maxDepth, minLeaf)
        trainAcc_s, testAcc_s = dt_train_test(
            dct,
            self.space_trainx,
            self.space_trainy,
            self.space_testx,
            self.space_testy,
        )
        sk_dt = DecisionTreeClassifier(
            criterion=ctn,
            max_depth=maxDepth,
            min_samples_leaf=minLeaf,
            random_state=0,
        )
        sk_dt.fit(self.space_trainx, self.space_trainy)
        yHat = sk_dt.predict(self.space_testx)
        testAcc = accuracy_score(self.space_testy, yHat)
        # check if accuracies are within reasonable range
        self.assertTrue(
            abs(testAcc - testAcc_s) < 0.15, "Unexpected low training accuracy"
        )
