import numpy as np
import numpy.testing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import unittest
from gradescope_utils.autograder_utils.decorators import weight, number, visibility

from compareFeat import evaluate_unreg, evaluate_lasso
from compareOpt import evaluate_sgd
from common import align_preprocess, simple_process


def true_unreg_lr(trainDF, testDF):
    lr = LinearRegression()
    lr.fit(trainDF.iloc[:, :-1], trainDF.iloc[:, -1])
    yhat = lr.predict(testDF.iloc[:, :-1])
    ytrue = testDF.iloc[:, -1]
    # calculate true r2 and mse
    return r2_score(ytrue, yhat), mean_squared_error(ytrue, yhat)


def true_lasso(trainDF, testDF, alpha):
    lr = Lasso(alpha=alpha)
    lr.fit(trainDF.iloc[:, :-1], trainDF.iloc[:, -1])
    yhat = lr.predict(testDF.iloc[:, :-1])
    ytrue = testDF.iloc[:, -1]
    # calculate true r2 and mse
    return r2_score(ytrue, yhat), mean_squared_error(ytrue, yhat)


class TestCompare(unittest.TestCase):

    @classmethod 
    def setUpClass(cls):
        cls.sim_train = simple_process(pd.read_csv("data/smalltrain.csv"))
        cls.sim_test = simple_process(pd.read_csv("data/smalltest.csv"))
        tmp1, tmp2 = align_preprocess(pd.read_csv("data/smalltrain.csv"),
                                      pd.read_csv("data/smalltest.csv"))
        # preprocess the 2 for scaling
        scaler = MinMaxScaler()
        scale_cols = ["LAST_PRICE",
                      "PX_VOLUME",
                      "VOLATILITY_10D",
                      "VOLATILITY_30D",
                      "LSTM_POLARITY",
                      "MONTH",
                      "DOW",
                      "YEAR"]
        scaler.fit(tmp1[scale_cols])
        tmp1[scale_cols] = scaler.transform(tmp1[scale_cols])
        tmp2[scale_cols] = scaler.transform(tmp2[scale_cols])
        cls.train = tmp1
        cls.test = tmp2

    def _check_output(self, result_dict):
        # make sure all these keys are there
        for k in ["r2", "mse", "time"]:
            self.assertTrue(k in result_dict, 
                            "function result does not have key " + k)
            self.assertIsInstance(result_dict[k], float,
                                  "Value for " + k + " is not an instance of float")
        self.assertTrue("nfeat" in result_dict,
                        "function result does not have key nfeat")
        try:
            self.assertIsInstance(result_dict["nfeat"], int)
        except Exception:
            self.assertTrue(np.issubdtype(result_dict["nfeat"], np.integer),
                            "nfeat does not return an integer")

    @weight(1)
    @number("4.A1")
    def test_evaluate_unreg1(self):
        res = evaluate_unreg(self.sim_train, self.sim_test)
        self.assertIsInstance(res, dict,
                              "Return type of evaluate_unreg is not a dictionary")
        self._check_output(res)

    @weight(1)
    @visibility('after_published')
    @number("4.A2")
    def test_evaluate_unreg2(self):
        # drop the non-numeric columns
        res = evaluate_unreg(self.sim_train, self.sim_test)
        true_r2, true_mse = true_unreg_lr(self.sim_train, self.sim_test)
        npt.assert_almost_equal(res["r2"], true_r2, decimal=2,
                                err_msg="R2 not the same")
        npt.assert_almost_equal(res["mse"], true_mse, decimal=2,
                                err_msg="mse not the same")

    @weight(1)
    @visibility('after_published')
    @number("4.A3")
    def test_evaluate_unreg3(self):
        # drop the non-numeric columns
        res = evaluate_unreg(self.train, self.test)
        true_r2, true_mse = true_unreg_lr(self.train, self.test)
        print(true_mse)
        npt.assert_almost_equal(res["r2"], true_r2, decimal=2,
                                err_msg="R2 not the same")
        npt.assert_almost_equal(res["mse"], true_mse, decimal=2,
                                err_msg="mse not the same")
    
    @weight(1)
    @number("4.B1")
    def test_evaluate_lasso1(self):
        res = evaluate_lasso(self.sim_train, self.sim_test, [0.0001, 0.01])
        self.assertIsInstance(res, dict,
                              "Return type of evaluate_lasso is not a dictionary")
        self._check_output(res)

    @weight(1.5)
    @visibility('after_published')
    @number("4.B2")
    def test_evaluate_lasso2(self):
        alpha = 0.01
        res = evaluate_lasso(self.train, self.test, [alpha])
        true_r2, true_mse = true_lasso(self.train, self.test, alpha)
        npt.assert_almost_equal(res["r2"], true_r2, decimal=2,
                                err_msg="R2 not the same")
        npt.assert_almost_equal(res["mse"], true_mse, decimal=2,
                                err_msg="mse not the same")     

    @weight(1.5)
    @visibility('after_published')
    @number("4.B3")
    def test_evaluate_lasso3(self):
        alpha = 0.5
        res = evaluate_lasso(self.train, self.test, [alpha])
        true_r2, true_mse = true_lasso(self.train, self.test, alpha)
        npt.assert_almost_equal(res["r2"], true_r2, decimal=2,
                                err_msg="R2 not the same")
        npt.assert_almost_equal(res["mse"], true_mse, decimal=2,
                                err_msg="mse not the same")

    @weight(1)
    @visibility('after_published')
    @number("4.B4")
    def test_evaluate_lasso4(self):
        res1 = evaluate_lasso(self.sim_train, self.sim_test, [0.001, 0.01])
        res2 = evaluate_lasso(self.sim_train, self.sim_test, [0.0001, 0.1])
        # these 2 should give different results since alphas are not the same
        self.assertNotEqual(res1["r2"], res2["r2"],
                            "R2 for different alpha parameter lists are same")  
        self.assertNotEqual(res1["mse"], res2["mse"],
                            "MSE for different alpha parameter lists are same")    


    @weight(1)
    @number("5.A1")
    def test_evaluate_sgd1(self):
        res = evaluate_sgd(self.sim_train, self.sim_test, 1)
        self.assertIsInstance(res, dict,
                              "Return type of evaluate_sgd is not a dictionary")
        self._check_output(res)

        
    @weight(1)
    @visibility('after_published')
    @number("5.A2")
    def test_evaluate_sgd2(self):
        res1 = evaluate_sgd(self.train, self.test, 1)
        res2 = evaluate_sgd(self.train, self.test, 1)
        # try 2 runs and they should be different
        try:
            self.assertNotEqual(res1["r2"], res2["r2"],
                                "R2 for different SGD runs are same")  
            self.assertNotEqual(res1["mse"], res2["mse"],
                                "MSE for different SGD runs are same")
        except Exception:
            res2 = evaluate_sgd(self.train, self.test, 1)
            self.assertNotEqual(res1["r2"], res2["r2"],
                                "R2 for different SGD runs are same")  
            self.assertNotEqual(res1["mse"], res2["mse"],
                                "MSE for different SGD runs are same")

    @weight(1)
    @visibility('after_published')
    @number("5.A3")
    def test_evaluate_sgd3(self):
        # run it long enough and it should yield simliar results
        mepoch = 50
        res = evaluate_sgd(self.train, self.test, mepoch)
        # compare against true
        lr = SGDRegressor(penalty=None, max_iter=mepoch)
        lr.fit(self.train.iloc[:, :-1], self.train.iloc[:, -1])
        # do the prediction
        yhat = lr.predict(self.test.iloc[:, :-1])
        ytrue = self.test.iloc[:, -1]
        true_mse = mean_squared_error(ytrue, yhat)
        # assert the range is reasonable
        self.assertLess(np.abs(res["mse"] - true_mse),
                        0.01,
                        "SGD results are not quite similar for convergence")



