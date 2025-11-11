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

        actual = self.mort.npx([34, 47, 73], 3, full_path=False)

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

    def test_npx_age_arr_t3_full_path(self):

        expected = np.array([[0.99934   , 0.998198  , 0.965856  ],
                             [0.99865145, 0.99619362, 0.92923943],
                             [0.99792843, 0.99396115, 0.89016863]])

        actual = self.mort.npx([34, 47, 73], 3, full_path=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_nqx_age_arr_3_full_path(self):

        expected = np.array([[0.00066   , 0.001802  , 0.034144  ],
                             [0.00068855, 0.00200438, 0.03661657],
                             [0.00072302, 0.00223247, 0.0390708 ]])

        actual = self.mort.nqx([34, 47, 73], 3, full_path=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_nqx_age_arr_t1_full_path(self):

        expected = np.array([[0.00066, 0.001802, 0.034144]])

        actual = self.mort.nqx([34, 47, 73], 1, full_path=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_nqx_age_arr_t_out_of_bounds_full_path(self):

        expected = np.array([[0.00066, 0.001802, 0.776648],
                             [0.000688545, 0.002004382, 0.178118083],
                             [0.000723024, 0.00223247, 0.036966288],
                             [0.000763415, 0.002492855, 0.008267629],
                             [0.000810695, 0.002785034, 0]])

        actual = self.mort.nqx([34, 47, 117], 5, full_path=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_nqx_age_int_3_full_path(self):

        expected = np.array([[0.00066], [0.00068855], [0.00072302]])

        actual = self.mort.nqx(34, 3, full_path=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_nqx_age_int_t1_full_path(self):

        expected = 0.00066

        actual = self.mort.nqx(34, 1)

        assert np.allclose(actual, expected)

    def test_nqx_age_int_t_out_of_bounds_full_path(self):

        expected = np.array([[0.776648  ],
                             [0.17811808],
                             [0.03696629],
                             [0.00826763],
                             [0.        ]])

        actual = self.mort.nqx(117, 5, full_path=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_npx_age_arr_t3_full_path2(self):

        expected = np.array([[0.99934   , 0.998198  , 0.965856  ],
                             [0.99865145, 0.99619362, 0.92923943],
                             [0.99792843, 0.        , 0.89016863]])

        actual = self.mort.npx([34, 47, 73], [3, 2, 3], full_path=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)



    def test_npx_age_arr_t3_2(self):

        expected = np.array([0.99792843, 0.99619362, 0.89016863])

        actual = self.mort.npx([34, 47, 73], [3, 2, 3])

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_nqx_age_arr_3_full_path2(self):

        expected = np.array([[0.00066   , 0.001802  , 0.034144  ],
                             [0.00068855, 0.00200438, 0.03661657],
                             [0.00072302, 1.        , 0.0390708 ]])

        actual = self.mort.nqx([34, 47, 73], [3, 2, 3], full_path=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)



    def test_nqx_age_arr_3_2(self):
        expected = np.array([0.00072302, 0.00200438, 0.0390708])


        actual = self.mort.nqx([34, 47, 73], [3, 2, 3])

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)