import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.benefits.survival_benefit import _SurvivalBenefit


class TestSurvivalBenefit(unittest.TestCase):

    mort_raw = pd.read_csv("../data/mortality/AM92.csv")
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    disc_raw = pd.read_csv("../data/curves/boe_spot_annual.csv")
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    policies_t3 = PolicyPortfolio.from_csv(r'../data/policy_data/annuity_test_data_term3.csv')
    policies_wol = PolicyPortfolio.from_csv(r'../data/policy_data/annuity_test_data.csv')

    annuity_t3 = _SurvivalBenefit(policies_t3, discount_curve, mortality)
    annuity = _SurvivalBenefit(policies_wol, discount_curve, mortality)

    def test_project_cashflows_term_3(self):

        expected = np.array([[4.9967,9.98198,14.48784],
                             [4.9932572737,9.96193618416,13.93859149776],
                             [4.98964215543384,9.9396114851713,13.3525294796452]])

        actual = self.annuity_t3.project_cashflows(aggregate=False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_project_cashflows_aggregated_term_3(self):

        expected = np.array([[29.46652], [28.89378496], [28.28178312]])

        actual = self.annuity_t3.project_cashflows(aggregate=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_present_value_term_3(self):

        expected = np.array([14.76985138, 29.46534658, 41.20403608])

        actual = self.annuity_t3.present_value(aggregate=False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_present_value_aggregated_term_3(self):

        expected = 85.43923404

        actual = self.annuity_t3.present_value(aggregate=True)

        assert np.allclose(actual, expected)

    def test_present_value(self):

        expected = np.array([172.8855834, 263.5107767, 152.4597913])

        actual = self.annuity.present_value(aggregate=False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_present_value_aggregated(self):

        expected = 588.8561514

        actual = self.annuity.present_value(aggregate=True)

        assert np.allclose(actual, expected)
