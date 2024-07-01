import numpy as np
import numpy.testing as npt
import pandas as pd
import unittest
from gradescope_utils.autograder_utils.decorators import weight, number, visibility
from selectFeat import cal_corr, select_features
from common import align_preprocess, simple_process



class TestSelect(unittest.TestCase):

    @classmethod 
    def setUpClass(cls):
        cls.df = pd.read_csv("data/smallsample.csv")
        cls.sim_train = simple_process(pd.read_csv("data/smalltrain.csv"))
        cls.sim_test = simple_process(pd.read_csv("data/smalltest.csv"))
        cls.train, cls.test = align_preprocess(pd.read_csv("data/smalltrain.csv"),
                                               pd.read_csv("data/smalltest.csv"))

    @weight(1)
    @number("3.A1")
    def test_corr1(self):
        # drop the non-numeric columns
        tmp_df = self.sim_train.copy()
        corr_df = cal_corr(tmp_df)
        self.assertEqual(corr_df.shape[0], tmp_df.shape[1],
                         "Number of rows returned by cal_corr is inconsistent")
        self.assertEqual(corr_df.shape[1], tmp_df.shape[1],
                         "Number of columns returned by cal_corr is inconsistent")

    @weight(1)
    @number("3.A2")
    def test_corr2(self):
        # drop the non-numeric columns
        tmp_df = self.sim_train.copy()
        corr_df = cal_corr(tmp_df)
        self.assertLessEqual(np.max(corr_df), 1,
                             "Correlation value larger than 1")
        self.assertGreaterEqual(np.min(corr_df), -1,
                                "Correlation value smaller than -1")

    @weight(1)
    @visibility('after_published')
    @number("3.A3")
    def test_corr3(self):
        tmp_df = self.sim_train.copy()
        npt.assert_array_almost_equal(tmp_df.corr(),
                                      cal_corr(tmp_df),
                                      decimal=4,
                                      err_msg="Correlation calcuation is incorrect")

    @weight(1)
    @visibility('after_published')
    @number("3.A4")
    def test_corr4(self):
        tmp_df = self.train.copy()
        npt.assert_array_almost_equal(tmp_df.corr(),
                                      cal_corr(tmp_df),
                                      decimal=4,
                                      err_msg="Correlation calcuation is incorrect")

    @weight(1)
    @number("3.D1")
    def test_sel1(self):
        # simple preprocessing
        train_df = self.sim_train.copy()
        test_df = self.sim_test.copy()
        tmp1, tmp2 = select_features(train_df, test_df)
        self.assertIsInstance(tmp1, pd.DataFrame,
                              "select_features() does not return a pandas dataframe in first tuple")
        self.assertIsInstance(tmp2, pd.DataFrame,
                              "select_features() does not return a pandas dataframe in second tuple")

    @weight(1)
    @number("3.D2")
    def test_sel2(self):
        # simple preprocessing
        train_df = self.sim_train.copy()
        test_df = self.sim_test.copy()
        tmp1, tmp2 = select_features(train_df, test_df)
        self.assertEqual(tmp1.shape[0], train_df.shape[0],
                        "select_features() does not return equal rows for training data")
        self.assertEqual(tmp2.shape[0], test_df.shape[0],
                        "select_features() does not return equal rows for test data")

    @weight(1)
    @number("3.D3")
    def test_sel3(self):
        # simple preprocessing
        train_df = self.sim_train.copy()
        test_df = self.sim_test.copy()
        tmp1, tmp2 = select_features(train_df, test_df)
        self.assertLessEqual(tmp1.shape[1], train_df.shape[1],
                            "select_features() returns more columns than passed in")
        self.assertLessEqual(tmp2.shape[1], test_df.shape[1],
                            "select_features() returns more columns than passed in")

    @weight(1)
    @number("3.D4")
    def test_sel4(self):
        # simple preprocessing
        train_df = self.sim_train.copy()
        test_df = self.sim_test.copy()
        tmp1, tmp2 = select_features(train_df, test_df)
        self.assertEqual(tmp1.columns.tolist(), tmp2.columns.tolist(),
                         "Column names are not the same between train and test!")
 
    @weight(1)
    @visibility('after_published')
    @number("3.D5")
    def test_sel5(self):
        train_df, test_df = self.train.copy(), self.test.copy()
        tmp1, tmp2 = select_features(train_df, test_df)
        self.assertLessEqual(tmp1.shape[1], train_df.shape[1],
                            "select_features() returns more columns than passed in")
        self.assertLessEqual(tmp2.shape[1], test_df.shape[1],
                            "select_features() returns more columns than passed in")

    @weight(1)
    @visibility('after_published')
    @number("3.D6")
    def test_sel6(self):
        train_df, test_df = self.train.copy(), self.test.copy()
        tmp1, tmp2 = select_features(train_df, test_df)
        self.assertListEqual(tmp1.columns.tolist(), tmp2.columns.tolist(),
                             "Column names are not the same between train and test!")

    @weight(1)
    @visibility('after_published')
    @number("3.D7")
    def test_sel7(self):
        train_df = self.sim_train.copy()
        test_df = self.sim_test.copy()
        tmp1, tmp2 = select_features(train_df, test_df)
        self.assertLess(tmp1.shape[1], train_df.shape[1],
                        "select_features() does not return a subset of features")

    @weight(1)
    @visibility('after_published')
    @number("3.D8")
    def test_sel8(self):
        train_df, test_df = self.train.copy(), self.test.copy()
        tmp1, tmp2 = select_features(train_df, test_df)
        self.assertLess(tmp1.shape[1], train_df.shape[1],
                        "select_features() does not return a subset of feature")

    @weight(1)
    @visibility('after_published')
    @number("3.D9")
    def test_sel9(self):
        train_df, test_df = self.train.copy(), self.test.copy()
        tmp1, tmp2 = select_features(train_df, test_df)
        # calculate the correlation here
        corr_tmp = tmp1.corr()
        # calculate smallest of correlation
        min_target = np.min(np.abs(corr_tmp.iloc[:, -1]))
        self.assertGreaterEqual(min_target,
                                0.001,
                                "min feature to target correlation is very small!")
        np.fill_diagonal(corr_tmp.values, 0)
        max_feat = np.max(np.abs(corr_tmp.iloc[:-1, :-1]))
        self.assertLessEqual(max_feat, 0.95,
                             "max feature correlation is very large!")
        
