import numpy as np
import numpy.testing as npt
import pandas as pd
import unittest
from gradescope_utils.autograder_utils.decorators import weight, number, visibility

from splitData import create_split


class TestSplit(unittest.TestCase):
    @classmethod 
    def setUpClass(cls):
        cls.df = pd.read_csv("data/smallsample.csv")

    @weight(1)
    @number("1.B1")
    def test_split1(self):
        tmp1, tmp2 = create_split(self.df)
        self.assertIsInstance(tmp1, pd.DataFrame,
                              "test_split() does not return a pandas dataframe in first tuple")
        self.assertIsInstance(tmp2, pd.DataFrame,
                              "test_split() does not return a pandas dataframe in second tuple")

    @weight(0.5)
    @number("1.B2")
    def test_split2(self):
        tmp1, tmp2 = create_split(self.df)
        self.assertEqual(tmp1.shape[1], self.df.shape[1],
                        "test_split() returns different columns than passed in data")
        self.assertEqual(tmp2.shape[1], self.df.shape[1],
                        "test_split() returns different columns than passed in data")


    @weight(0.5)
    @visibility('after_published')
    @number("1.B3")
    def test_split3(self):
        tmp1, tmp2 = create_split(self.df)
        self.assertLess(tmp1.shape[0], self.df.shape[0],
                        "test_split() returns more or equal number of rows than passed in data")
        self.assertLess(tmp2.shape[0], self.df.shape[0],
                        "test_split() returns more or equal number of rows than passed in data")

    @weight(0.5)
    @visibility('after_published')
    @number("1.B4")
    def test_split4(self):
        tmp1, tmp2 = create_split(self.df)
        self.assertEqual(tmp1.shape[0] + tmp2.shape[0],
                         self.df.shape[0],
                         "test_split() does not partition equally into train-test")

    @weight(0.5)
    @visibility('after_published')
    @number("1.B5")
    def test_split5(self):
        tmp1, tmp2 = create_split(self.df)
        tmp3, tmp4 = create_split(self.df)
        tmp1 = tmp1.reset_index(drop = True)
        tmp2 = tmp2.reset_index(drop = True)
        tmp3 = tmp3.reset_index(drop = True)
        tmp4 = tmp4.reset_index(drop = True)
        msg = 'Splits are not randomized'
        self.assertTrue(((tmp1 != tmp3).any()).any(), msg)
        self.assertTrue(((tmp2 != tmp4).any()).any(), msg)
