import numpy as np
import numpy.testing as npt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import unittest
from gradescope_utils.autograder_utils.decorators import weight, number, visibility

from extractFeat import extract_date, extract_company, preprocess_data
from extractFeat import extract_binary, extract_tfidf
from common import align_preprocess, simple_process


class TestExtract(unittest.TestCase):

    @classmethod 
    def setUpClass(cls):
        cls.df = pd.read_csv("data/smallsample.csv")
        cls.traindf = pd.read_csv("data/smalltrain.csv")
        cls.testdf = pd.read_csv("data/smalltest.csv")

    
    @weight(1)
    @number("2.B1")
    def test_date1(self):
        date_df = extract_date(self.df.copy())
        msg = "DATE column not dropped in extract_date()"
        self.assertFalse('DATE' in date_df.columns, msg)
    
    @weight(2)
    @visibility('after_published')
    @number("2.B2")
    def test_date2(self):
        date_df = extract_date(self.df.copy())
        msg = "extract_date() did not add at least 2 new columns"
        self.assertGreater(date_df.shape[1],
                           self.df.shape[1],
                           msg)

    @weight(1)
    @number("2.D1")
    def test_company1(self):
        company_df = extract_company(self.df.copy())
        msg = "STOCK column not dropped in extract_company()"
        self.assertFalse('STOCK' in company_df.columns, msg)
    
    @weight(1)
    @visibility('after_published')
    @number("2.D2")
    def test_company2(self):
        company_df = extract_company(self.df.copy())
        msg = "extract_company() did not create new features"
        self.assertGreater(company_df.shape[1],
                           self.df.shape[1],
                           msg)

    @weight(0.5)
    @number("2.F1")
    def test_pre1(self):
        proc_train = extract_date(self.traindf.copy())
        proc_test = extract_date(self.testdf.copy())
        # extract company features for both/test
        proc_train = extract_company(proc_train)
        proc_test = extract_company(proc_test)
        trainTr, testTr = preprocess_data(proc_train, proc_test)
        # preprocess should yield dataframes
        self.assertIsInstance(trainTr, pd.DataFrame,
                              "preprocess_data() does not return a pandas dataframe in first tuple")
        self.assertIsInstance(testTr, pd.DataFrame,
                              "preprocess_data() does not return a pandas dataframe in second tuple")

    @weight(0.5)
    @number("2.F2")
    def test_pre2(self):
        proc_train = extract_date(self.traindf.copy())
        proc_test = extract_date(self.testdf.copy())
        # extract company features for both/test
        proc_train = extract_company(proc_train)
        proc_test = extract_company(proc_test)
        trainTr, testTr = preprocess_data(proc_train, proc_test)
        self.assertEqual(trainTr.shape[0], self.traindf.shape[0],
                        "select_features() does not return equal rows for training data")
        self.assertEqual(testTr.shape[0], self.testdf.shape[0],
                        "select_features() does not return equal rows for test data")

    @weight(1)
    @visibility('after_published')
    @number("2.F3")
    def test_pre3(self):
        proc_train = extract_date(self.traindf.copy())
        proc_test = extract_date(self.testdf.copy())
        # extract company features for both/test
        proc_train = extract_company(proc_train)
        proc_test = extract_company(proc_test)
        trainTr, testTr = preprocess_data(proc_train.copy(), proc_test.copy())
        # preprocess the data
        msg = "did not preprocess data"
        self.assertFalse(trainTr.equals(proc_train), msg)
        self.assertFalse(testTr.equals(proc_test), msg)
    
    @weight(1)
    @number("2.G1")
    def test_bin1(self):
        trainTr, testTr = extract_binary(self.traindf["TWTOKEN"].copy(),
                                         self.testdf["TWTOKEN"].copy(), 10)
        # preprocess should yield dataframes
        self.assertIsInstance(trainTr, pd.DataFrame,
                              "extract_binary() does not return a pandas dataframe in first tuple")
        self.assertIsInstance(testTr, pd.DataFrame,
                              "extract_binary() does not return a pandas dataframe in second tuple")

    @weight(1)
    @number("2.G2")
    def test_bin2(self):
        trainTr, testTr = extract_binary(self.traindf["TWTOKEN"].copy(),
                                         self.testdf["TWTOKEN"].copy(), 10)
        # preprocess should yield dataframes
        self.assertEqual(trainTr.shape[0], self.traindf.shape[0],
                              "extract_binary() does not return same number of rows in train")
        self.assertEqual(testTr.shape[0], self.testdf.shape[0],
                              "extract_binary() does not return same number of rows in test")

    @weight(1)
    @number("2.G3")
    def test_bin3(self):
        k = 10
        trainTr, testTr = extract_binary(self.traindf["TWTOKEN"].copy(),
                                         self.testdf["TWTOKEN"].copy(), k)
        # preprocess should yield dataframes
        self.assertEqual(trainTr.shape[1], k,
                              "extract_binary() does not return expeted number of columns in train")
        self.assertEqual(testTr.shape[1], k,
                              "extract_binary() does not return expeted number of columns in test")

    @weight(1)
    @visibility('after_published')
    @number("2.G4")
    def test_bin4(self):
        k = 25
        msg = 'Dataset should only contain 0 or 1'
        trainTr, testTr = extract_binary(self.traindf["TWTOKEN"].copy(),
                                         self.testdf["TWTOKEN"].copy(), k)
        self.assertEqual(len(np.unique(trainTr)), 2, msg = msg)


    @weight(1)
    @visibility('after_published')
    @number("2.G5")
    def test_bin5(self):
        k = 25
        trainTr, testTr = extract_binary(self.traindf["TWTOKEN"].copy(),
                                         self.testdf["TWTOKEN"].copy(), k)
        # use train to calculate the words
        vect = CountVectorizer(stop_words='english',
                               max_features=k,
                               binary=True)
        tw_train = vect.fit_transform(self.traindf["TWTOKEN"])
        self.assertSetEqual(set(vect.get_feature_names_out()),
                            set(trainTr.columns.tolist()),
                            "Words are not the same")

    @weight(1)
    @number("2.H1")
    def test_tfidf1(self):
        trainTr, testTr = extract_tfidf(self.traindf["TWTOKEN"].copy(),
                                         self.testdf["TWTOKEN"].copy(), 10)
        # preprocess should yield dataframes
        self.assertIsInstance(trainTr, pd.DataFrame,
                              "extract_tfidf() does not return a pandas dataframe in first tuple")
        self.assertIsInstance(testTr, pd.DataFrame,
                              "extract_tfidf() does not return a pandas dataframe in second tuple")
    
    @weight(1)
    @number("2.H2")
    def test_tfidf2(self):
        trainTr, testTr = extract_tfidf(self.traindf["TWTOKEN"].copy(),
                                         self.testdf["TWTOKEN"].copy(), 10)
        # preprocess should yield dataframes
        self.assertEqual(trainTr.shape[0], self.traindf.shape[0],
                              "extract_tfidf() does not return same number of rows in train")
        self.assertEqual(testTr.shape[0], self.testdf.shape[0],
                              "extract_tfidf() does not return same number of rows in test")

    @weight(1)
    @number("2.H3")
    def test_tfidf3(self):
        k = 10
        trainTr, testTr = extract_tfidf(self.traindf["TWTOKEN"].copy(),
                                         self.testdf["TWTOKEN"].copy(), k)
        # preprocess should yield dataframes
        self.assertEqual(trainTr.shape[1], k,
                              "extract_tfidf() does not return expeted number of columns in train")
        self.assertEqual(testTr.shape[1], k,
                              "extract_tfidf() does not return expeted number of columns in test")

    @weight(1)
    @visibility('after_published')
    @number("2.H4")
    def test_tfidf4(self):
        k = 25
        msg = 'Dataset should only contain numeric value'
        trainTr, testTr = extract_tfidf(self.traindf["TWTOKEN"].copy(),
                                         self.testdf["TWTOKEN"].copy(), k)
        for x in trainTr.dtypes:
            self.assertTrue(np.issubdtype(x, np.floating),
                            msg = msg)


    @weight(1)
    @visibility('after_published')
    @number("2.H5")
    def test_tfidf5(self):
        k = 25
        trainTr, testTr = extract_tfidf(self.traindf["TWTOKEN"].copy(),
                                        self.testdf["TWTOKEN"].copy(), k)
        # use train to calculate the words
        vect = TfidfVectorizer(stop_words='english',
                               max_features=k)
        tw_train = vect.fit_transform(self.traindf["TWTOKEN"])
        self.assertSetEqual(set(vect.get_feature_names_out()),
                            set(trainTr.columns.tolist()),
                            "Words are not the same")
