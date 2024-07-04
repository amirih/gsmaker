import numpy as np
import numpy.testing as npt
import pandas as pd
import unittest
from gradescope_utils.autograder_utils.decorators import weight, number, visibility
from sklearn.tree import DecisionTreeClassifier
import sklearn.datasets as skd
import sklearn.model_selection as skms
from modelAssess import holdout, kfold_cv, mc_cv, eval_dt_perf


class TestAssess(unittest.TestCase):

    @classmethod 
    def setUpClass(cls):
        iris_data = skd.load_iris()
        iris_x = iris_data.data
        iris_y = iris_data.target
        # turn into binary classification
        mask = (iris_y != 2)
        iris_x = iris_x[mask, :]
        iris_y = iris_y[mask]
        # split data into train test 
        cls.train_x, cls.test_x, cls.train_y, cls.test_y = skms.train_test_split(iris_x,
                                                                                 iris_y,
                                                                                 test_size=0.3,
                                                                                 random_state=20)
        cls.perf_keys = ["trainAUC", "testAUC", "trainAUPRC", "testAUPRC", "trainF1", "testF1"]
        cls.assess_keys = cls.perf_keys + ["timeElapsed"]

        cls.space_x = pd.read_csv("data/space_trainx.csv").to_numpy()
        cls.space_y = pd.read_csv("data/space_trainy.csv").to_numpy().flatten()


    @weight(1)
    @number("2.A1")
    def test_eval_dt(self):
        dc = DecisionTreeClassifier()
        res = eval_dt_perf(dc, self.train_x, self.train_y, self.test_x, self.test_y)
        # make sure it is a decision tree
        self.assertIsInstance(res, dict)
        # make sure it has the following keys
        for k in self.perf_keys:
            self.assertTrue(k in res,
                "Missing the following key in the dictionary: " + k)

    @weight(1)
    @number("2.A2")
    def test_eval_dt2(self):
        dc = DecisionTreeClassifier()
        res = eval_dt_perf(dc, self.train_x, self.train_y, self.test_x, self.test_y)
        for k in self.perf_keys:
            self.assertIsInstance(res[k], float,
                "Not a float value associated with key" + k)

    @weight(1)
    @number("2.A3")
    def test_eval_dt3(self):
        dc = DecisionTreeClassifier()
        res = eval_dt_perf(dc, self.train_x, self.train_y, self.test_x, self.test_y)
        for k in self.perf_keys:
            self.assertTrue(res[k] >= 0 and res[k] <= 1,
                "Range is not between 0 and 1 for key " + k)

    @weight(1)
    @visibility('after_published')
    @number("2.A4")
    def test_eval_dt4(self):
        dc = DecisionTreeClassifier()
        res = eval_dt_perf(dc, self.train_x, self.train_y, self.test_x, self.test_y)
        for k in self.perf_keys:
            self.assertTrue(res[k] >= 0 and res[k] <= 1,
                "Range is not between 0 and 1 for key " + k)

    @weight(1)
    @visibility('after_published')
    @number("2.A5")
    def test_eval_dt5(self):
        dc = DecisionTreeClassifier()
        res = eval_dt_perf(dc, self.train_x, self.train_y, self.test_x, self.test_y)
        for k in self.perf_keys:
            self.assertTrue(res[k] >= 0 and res[k] <= 1,
                "Range is not between 0 and 1 for key " + k)

    @weight(1)
    @visibility('after_published')
    @number("2.A6")
    def test_eval_dt6(self):
        dc = DecisionTreeClassifier()
        res = eval_dt_perf(dc, self.train_x, self.train_y, self.test_x, self.test_y)
        for k in self.perf_keys:
            self.assertTrue(res[k] >= 0 and res[k] <= 1,
                "Range is not between 0 and 1 for key " + k)

    @weight(1)
    @number("2.B1")
    def test_holdout1(self):
        dc = DecisionTreeClassifier()
        res = holdout(dc, self.train_x, self.train_y, 0.4)
        self.assertIsInstance(res, dict)
        # make sure it has the following keys
        for k in self.assess_keys:
            self.assertTrue(k in res,
                "Missing the following key in the dictionary: " + k)
            self.assertIsInstance(res[k], float,
                "Not a float value associated with key" + k)

    @weight(1)
    @number("2.B2")
    def test_holdout2(self):
        dc=DecisionTreeClassifier()
        res = holdout(dc, self.train_x, self.train_y, 0.2)
        for k in self.perf_keys:
            self.assertTrue(res[k] >= 0 and res[k] <= 1,
                "Range is not between 0 and 1 for key " + k)

    @weight(1)
    @visibility('after_published')
    @number("2.B3")
    def test_holdout3(self):
        dc = DecisionTreeClassifier()
        res = holdout(dc, self.train_x, self.train_y, 0.3)
        #sanity check - AUC>0.5
        self.assertTrue(res["trainAUC"]>0.5 and res["testAUC"]>0.5,
            "Train/test AUC is lower than expected")

    @weight(1)
    @visibility('after_published')
    @number("2.B4")
    def test_holdout4(self):
        dc = DecisionTreeClassifier()
        res = holdout(dc, self.train_x, self.train_y, 0.3)
        #sanity check - AUC>0.5
        self.assertTrue(res["trainF1"]>0.5 and res["trainF1"]>0.5,
            "Train/test F1 is lower than expected")

    @weight(1)
    @visibility('after_published')
    @number("2.B5")
    def test_holdout_diff(self):
        # value should be different both times
        dc = DecisionTreeClassifier()
        res1 = holdout(dc, self.space_x, self.space_y, 0.5)
        dc2=DecisionTreeClassifier()
        res2 = holdout(dc, self.space_x, self.space_y, 0.5)
        try:
            #sanity check the results should be different!
            self.assertTrue(res1["testAUC"] != res2["testAUC"], "Deterministic holdout")
        except AssertionError:
            # try once more if not different
            res2 = holdout(dc, self.space_x, self.space_y, 0.5)
            self.assertTrue(res1["testAUC"] != res2["testAUC"], "Deterministic holdout")


    @weight(1)
    @number("2.C1")
    def test_kfold1(self):
        dc = DecisionTreeClassifier()
        res = kfold_cv(dc, self.train_x, self.train_y, 2)
        self.assertIsInstance(res, dict)
        # make sure it has the following keys
        for k in self.assess_keys:
            self.assertTrue(k in res,
                "Missing the following key in the dictionary: " + k)
            self.assertIsInstance(res[k], float,
                "Not a float value associated with key" + k)

    @weight(1)
    @number("2.C2")
    def test_kfold2(self):
        dc = DecisionTreeClassifier()
        res = kfold_cv(dc, self.train_x, self.train_y, 3)
        for k in self.perf_keys:
            self.assertTrue(res[k] >= 0 and res[k] <= 1,
                "Range is not between 0 and 1 for key " + k)

    @weight(1)
    @visibility('after_published')
    @number("2.C3")
    def test_kfold3(self):
        dc = DecisionTreeClassifier()
        res = kfold_cv(dc, self.train_x, self.train_y, 3)
        #sanity check - AUC>0.5
        self.assertTrue(res["trainAUC"]>0.5 and res["testAUC"]>0.5,
            "Train/test AUC is lower than expected")

    @weight(1)
    @visibility('after_published')
    @number("2.C4")
    def test_kfold4(self):
        dc = DecisionTreeClassifier()
        res1 = kfold_cv(dc, self.space_x, self.space_y, 3)
        res2 = kfold_cv(dc, self.space_x, self.space_y, 3)
        # ensure sampling is different
        try:
            #sanity check the results should be different!
            self.assertTrue(res1["testAUC"] != res2["testAUC"], "Deterministic holdout")
        except AssertionError:
            # try once more if not different
            res2 = kfold_cv(dc, self.space_x, self.space_y, 3)
            self.assertTrue(res1["testAUC"] != res2["testAUC"], "Deterministic holdout")

    @weight(1)
    @visibility('after_published')
    @number("2.C5")
    def test_kfold5(self):
        dc = DecisionTreeClassifier()
        res = kfold_cv(dc, self.train_x, self.train_y, 5)
        #sanity check - AUC>0.5
        self.assertTrue(res["trainF1"]>0.5 and res["trainF1"]>0.5,
            "Train/test F1 is lower than expected")

    @weight(1)
    @visibility('after_published')
    @number("2.C6")
    def test_kfold6(self):
        dc = DecisionTreeClassifier()
        res = kfold_cv(dc, self.train_x, self.train_y, 5)
        #sanity check - AUC>0.5
        self.assertTrue(res["trainAUC"]>0.5 and res["testAUC"]>0.5,
            "Train/test AUC is lower than expected")

    @weight(1)
    @visibility('after_published')
    @number("2.C7")
    def test_kfold7(self):
        dc = DecisionTreeClassifier(max_depth=5)
        res = kfold_cv(dc, self.train_x, self.train_y, 10)
        #sanity check - AUC>0.5
        self.assertGreaterEqual(res["trainAUC"], res["testAUC"],
            "Train AUC is lower than test AUC")


    @weight(1)
    @number("2.D1")
    def test_mc1(self):
        dc = DecisionTreeClassifier()
        res = mc_cv(dc, self.train_x, self.train_y, 0.2, 5)
        self.assertIsInstance(res, dict)
        # make sure it has the following keys
        for k in self.assess_keys:
            self.assertTrue(k in res,
                "Missing the following key in the dictionary: " + k)
            self.assertIsInstance(res[k], float,
                "Not a float value associated with key" + k)

    @weight(1)
    @number("2.D2")
    def test_mc2(self):
        dc = DecisionTreeClassifier()
        res = mc_cv(dc, self.train_x, self.train_y, 0.3, 3)
        for k in self.perf_keys:
            self.assertTrue(res[k] >= 0 and res[k] <= 1,
                "Range is not between 0 and 1 for key " + k)


    @weight(1)
    @visibility('after_published')
    @number("2.D3")
    def test_mc3(self):
        dc = DecisionTreeClassifier()
        res = mc_cv(dc, self.train_x, self.train_y, 0.2, 5)
        #sanity check - AUC>0.5
        self.assertTrue(res["trainAUC"]>0.5 and res["testAUC"]>0.5,
                        "Train/test AUC is lower than expected")

    @weight(1)
    @visibility('after_published')
    @number("2.D4")
    def test_mc4(self):
        dc = DecisionTreeClassifier()
        res = mc_cv(dc, self.train_x, self.train_y, 0.2, 5)
        #sanity check - AUC>0.5
        self.assertTrue(res["trainF1"]>0.5 and res["trainF1"]>0.5,
            "Train/test F1 is lower than expected")

    @weight(1)
    @visibility('after_published')
    @number("2.D5")
    def test_mc5(self):
        dc = DecisionTreeClassifier()
        res = mc_cv(dc, self.train_x, self.train_y, 0.33, 3)
        #sanity check - AUC>0.5
        self.assertTrue(res["trainAUC"]>0.5 and res["testAUC"]>0.5,
            "Train/test AUC is lower than expected")

    @weight(1)
    @visibility('after_published')
    @number("2.D6")
    def test_mc6(self):
        dc = DecisionTreeClassifier()
        res1 = mc_cv(dc, self.space_x, self.space_y, 0.5, 2)
        dc2=DecisionTreeClassifier()
        res2 = mc_cv(dc, self.space_x, self.space_y, 0.5, 2)
        try:
            #sanity check the results should be different!
            self.assertTrue(res1["testAUC"] != res2["testAUC"], "Deterministic holdout")
        except AssertionError:
            # try once more if not different
            res2 = mc_cv(dc, self.space_x, self.space_y, 0.5, 2)
            self.assertTrue(res1["testAUC"] != res2["testAUC"], "Deterministic holdout")

    @weight(1)
    @visibility('after_published')
    @number("2.D7")
    def test_mc7(self):
        dc = DecisionTreeClassifier(max_depth=5)
        res = mc_cv(dc, self.train_x, self.train_y, 0.2, 5)
        #sanity check - AUC>0.5
        self.assertGreaterEqual(res["trainAUC"], res["testAUC"],
                                "Train AUC is lower than test AUC")
