import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.products.annuity import Annuity


class TestMortalityTable(unittest.TestCase):

    mort_raw = pd.read_csv("data/mortality/AM92.csv")
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    disc_raw = pd.read_csv("data/curves/boe_spot_annual.csv")
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    age = [34, 47, 73]
    term = 3
    amount = [5, 10, 15]

    annuity = Annuity(age, term, amount, mortality, discount_curve)

    def test_project_cashflows(self):

        expected = np.array([[4.9967,9.98198,14.48784],
                             [4.9932572737,9.96193618416,13.93859149776],
                             [4.98964215543384,9.9396114851713,13.3525294796452]])

        actual = self.annuity.project_cashflows(False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_project_cashflows_aggregated(self):

        expected = np.array([29.46652   , 28.89378496, 28.28178312])

        actual = self.annuity.project_cashflows(True)

        assert np.allclose(actual, expected)

    def test_present_value(self):

        expected = np.array([14.76985138, 29.46534658, 41.20403608])

        actual = self.annuity.present_value(False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_present_value_aggregated(self):

        expected = 85.43923404

        actual = self.annuity.present_value(True)

        assert np.allclose(actual, expected)
