import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from gradescope_utils.autograder_utils.decorators import leaderboard

from extractFeat import extract_date, extract_company, preprocess_data
from common import reorder_columns


class TestLeaderboard(unittest.TestCase):

    @classmethod 
    def setUpClass(cls):
        cls.traindf = pd.read_csv("data/lb_train.csv")
        cls.testdf = pd.read_csv("data/lb_test.csv")


    @leaderboard("MSE", "asc")
    def test_leaderboard(self, set_leaderboard_value=None):
        """Sets a leaderboard value"""
        proc_train = extract_date(self.traindf)
        proc_test = extract_date(self.testdf)
        # extract company features for both/test
        proc_train = extract_company(proc_train)
        proc_test = extract_company(proc_test)
        # run pre-process
        proc_train, proc_test = preprocess_data(proc_train, proc_test)

        # drop the text data
        proc_train.drop(columns=["TWTOKEN", "TWEET"], inplace=True)
        proc_test.drop(columns=["TWTOKEN", "TWEET"], inplace=True)
        # do some book-keeping here to make sure columns are the same
        ptrain, ptest = proc_train.align(proc_test,
                                         join='outer',
                                         axis=1, fill_value=0)
        ptrain = reorder_columns(ptrain)
        ptest = reorder_columns(ptest)
        
        lr = LinearRegression()
        lr.fit(ptrain.iloc[:, :-1], ptrain.iloc[:, -1])
        yhat = lr.predict(ptest.iloc[:, :-1])
        ytrue = ptest.iloc[:, -1]

        set_leaderboard_value(mean_squared_error(ytrue, yhat))
