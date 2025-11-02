import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.products.life_assurance import LifeAssurance


class TestLifeAssurance(unittest.TestCase):

    mort_raw = pd.read_csv("data/mortality/AM92.csv")
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    disc_raw = pd.read_csv("data/curves/boe_spot_annual.csv")
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    age = [34, 47, 73]
    amount = [5, 10, 15]

    term_assurance = LifeAssurance(age, 3, amount, mortality, discount_curve)
    wol_assurance = LifeAssurance(age, None, amount, mortality, discount_curve)

    def test_project_cashflows_term_3(self):

        expected = np.array([[0.0033, 0.01802, 0.51216],
                             [0.003442726, 0.020043816, 0.549248502],
                             [0.003615118, 0.022324699, 0.586062018]])

        actual = self.term_assurance.project_cashflows(aggregate=False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_project_cashflows_aggregated_term_3(self):

        expected = np.array([0.53348, 0.572735044, 0.612001835])

        actual = self.term_assurance.project_cashflows(aggregate=True)

        assert np.allclose(actual, expected)

    def test_present_value(self):

        expected = np.array([3.125365639, 6.961391422, 13.05573419])

        actual = self.wol_assurance.present_value(aggregate=False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_present_value_aggregated(self):

        expected = 23.14249125

        actual = self.wol_assurance.present_value(aggregate=True)

        assert np.allclose(actual, expected)
