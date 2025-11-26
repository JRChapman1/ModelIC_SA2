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

    policies_in = pd.read_csv(data_path('policy_data', 'mixed_policies.csv'))

    # TODO: Should not need to do this
    policies_in['annual_premium'] = 1.0
    policies_in = PolicyPortfolio.from_df(policies_in)

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

        expected = np.array([294.4919, 321.0385, 490.7803, 1334.534, 438.1998, 1100.275, 12677.79, 24088.09,
                             4652.497, 1421.843, 12722.83, 11510.22, 959828.2, 130964.4, 116709.7])

        actual = self.eng.price_policy_portfolio(self.policies_in)

        actual = np.array([actual[i] for i in range(1, 1+ max(actual.keys()))])

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

