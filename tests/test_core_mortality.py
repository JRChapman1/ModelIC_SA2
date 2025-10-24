import unittest
import numpy as np
import pandas as pd

from modelic.core.mortality import MortalityTable


class TestMortalityTable(unittest.TestCase):

    df = pd.read_csv("data/mortality/AM92.csv")
    ages = df['x'].to_numpy(int)
    qx = df['q_x'].to_numpy(float)
    mort = MortalityTable(ages, qx, 'AM92')

    def test_npx_age_arr_t3(self):

        expected = np.array([0.99792843, 0.99396115, 0.89016863])

        actual = self.mort.npx([34, 47, 73], 3)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_npx_age_arr_t1(self):

        expected = np.array([0.99934, 0.998198, 0.965856])

        actual = self.mort.npx([34, 47, 73])

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_npx_age_arr_t0(self):

        expected = np.array([1, 1, 1])

        actual = self.mort.npx([34, 47, 73], 0)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_nqx_age_arr_t3(self):

        expected = np.array([0.00072302, 0.00223247, 0.0390708])

        actual = self.mort.nqx([34, 47, 73], 3)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_nqx_age_arr_t1(self):

        expected = np.array([0.00066, 0.001802, 0.034144])

        actual = self.mort.nqx([34, 47, 73], 1)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_nqx_age_arr_t0(self):

        expected = np.array([0, 0, 0])

        actual = self.mort.nqx([34, 47, 73], 0)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_npx_age_arr_t3(self):
        expected = np.array([0.99792843, 0.99396115, 0.89016863])

        actual = self.mort.npx([34, 47, 73], 3)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_npx_age_arr_t1(self):
        expected = np.array([0.99934, 0.998198, 0.965856])

        actual = self.mort.npx([34, 47, 73])

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_npx_age_arr_t0(self):
        expected = np.array([1, 1, 1])

        actual = self.mort.npx([34, 47, 73], 0)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_nqx_age_arr_t3(self):
        expected = np.array([0.00072302, 0.00223247, 0.0390708])

        actual = self.mort.nqx([34, 47, 73], 3)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_nqx_age_arr_t1(self):
        expected = 0.00066

        actual = self.mort.nqx(34, 1)

        assert np.allclose(actual, expected)

    def test_nqx_age_arr_t0(self):
        expected = 0

        actual = self.mort.nqx(34, 0)

        assert np.allclose(actual, expected)

    def test_npx_age_int_t3(self):
        expected = 0.99792843

        actual = self.mort.npx(34, 3)

        assert np.allclose(actual, expected)

    def test_npx_age_int_t1(self):
        expected = 0.99934

        actual = self.mort.npx(34)

        assert np.allclose(actual, expected)

    def test_npx_age_int_t0(self):
        expected = 1

        actual = self.mort.npx(34, 0)

        assert np.allclose(actual, expected)

    def test_nqx_age_int_t3(self):
        expected = 0.00072302

        actual = self.mort.nqx(34, 3)

        assert np.allclose(actual, expected)

    def test_nqx_age_int_t1(self):
        expected = 0.00066

        actual = self.mort.nqx(34, 1)

        assert np.allclose(actual, expected)

    def test_nqx_age_int_t0(self):
        expected = 0

        actual = self.mort.nqx(34, 0)

        assert np.allclose(actual, expected)



