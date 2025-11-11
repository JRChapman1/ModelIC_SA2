import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.benefits.death_benefit import _DeathBenefit


class TestDeathBenefit(unittest.TestCase):

    mort_raw = pd.read_csv("../data/mortality/AM92.csv")
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    disc_raw = pd.read_csv("../data/curves/boe_spot_annual.csv")
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    wol_policies = PolicyPortfolio.from_csv(r'../data/policy_data/wol_test_data.csv')
    term_policies = PolicyPortfolio.from_csv(r'../data/policy_data/ta_test_data.csv')

    term_death_benefit = _DeathBenefit(term_policies, discount_curve, mortality)
    wol_death_benefit = _DeathBenefit(wol_policies, discount_curve, mortality)

    def test_project_cashflows_term_3(self):

        expected = np.array([[0.0033, 0.01802, 0.51216],
                             [0.003442726, 0.020043816, 0.549248502],
                             [0.003615118, 0.022324699, 0.586062018]])

        actual = self.term_death_benefit.project_cashflows(aggregate=False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_project_cashflows_aggregated_term_3(self):

        expected = np.array([[0.53348], [0.572735044], [0.612001835]])

        actual = self.term_death_benefit.project_cashflows(aggregate=True)

        assert np.allclose(actual, expected)

    def test_present_value(self):

        expected = np.array([3.125365639, 6.961391422, 13.05573419])

        actual = self.wol_death_benefit.present_value(aggregate=False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_present_value_aggregated(self):

        expected = 23.14249125

        actual = self.wol_death_benefit.present_value(aggregate=True)

        assert np.allclose(actual, expected)
