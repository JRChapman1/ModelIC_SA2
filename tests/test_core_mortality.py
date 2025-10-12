import unittest
import numpy as np
import pandas as pd

from modelic.core.mortality import MortalityTable


class TestMortalityTable(unittest.TestCase):

    df = pd.read_csv("data/mortality/AM92.csv")

    ages = df['x'].to_numpy(int)
    qx = df['q_x'].to_numpy(float)

    mort = MortalityTable(ages, qx, 'AM92')

    def test_tq_age_arr_3_full_path(self):

        expected = np.array([[0.00066   , 0.001802  , 0.034144  ],
                             [0.00068855, 0.00200438, 0.03661657],
                             [0.00072302, 0.00223247, 0.0390708 ]])

        actual = self.mort.tq([34, 47, 73], 3, True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_tq_age_arr_t1_full_path(self):

        expected = np.array([[0.00066   , 0.001802  , 0.034144  ]])

        actual = self.mort.tq([34, 47, 73], 1, True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_tq_age_arr_t3(self):

        expected = np.array([0.00072302, 0.00223247, 0.0390708])

        actual = self.mort.tq([34, 47, 73], 3, False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_tp_age_arr_t3_full_path(self):

        expected = np.array([[0.99934   , 0.998198  , 0.965856  ],
                             [0.99865145, 0.99619362, 0.92923943],
                             [0.99792843, 0.99396115, 0.89016863]])

        actual = self.mort.tp([34, 47, 73], 3, True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_tp_age_arr_t3(self):

        expected = np.array([0.99792843, 0.99396115, 0.89016863])

        actual = self.mort.tp([34, 47, 73], 3)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)



# ----------



    def test_tq_age_int_3_full_path(self):

        expected = np.array([[0.03414400],
                             [0.03661657],
                             [0.03907080]])

        actual = self.mort.tq(73, 3, True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_tq_age_int_t1_full_path(self):

        expected = np.array([[0.03414400]])

        actual = self.mort.tq(73, 1, True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_tq_age_int_t3(self):

        expected = np.array([0.0390708])

        actual = self.mort.tq(73, 3, False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_tp_age_int_t3_full_path(self):

        expected = np.array([[0.96585600],
                             [0.92923943],
                             [0.89016863]])

        actual = self.mort.tp(73, 3, True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_tp_age_int_t3(self):

        expected = np.array([0.89016863])

        actual = self.mort.tp(73, 3)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)