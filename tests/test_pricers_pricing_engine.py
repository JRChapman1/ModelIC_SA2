import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.pricers.pricing_engine import PricingEngine
from _data import data_path


class TestPricingEngine(unittest.TestCase):

    try:
        expense_spec = pd.read_csv('./../Parameters/expense_spec.csv')
    except FileNotFoundError:
        expense_spec = pd.read_csv('./Parameters/expense_spec.csv')

    policies_in = PolicyPortfolio.from_csv(data_path('policy_data', 'mixed_policies.csv'))

    mort_raw = pd.read_csv(data_path('mortality', 'AM92.csv'))
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    disc_raw = pd.read_csv(data_path('curves', 'boe_spot_annual.csv'))
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    expense_inflation_rate = 0.03

    eng = PricingEngine(mortality, discount_curve, expense_spec, expense_inflation_rate)

    def test_pricing_engine(self):

        expected_values = np.array([])

        actual = self.eng.price_policies(self.policies_in)

        assert actual.shape == expected_values.shape
        assert np.allclose(actual.values, expected_values)

