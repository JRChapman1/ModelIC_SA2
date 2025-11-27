import unittest
import numpy as np
import pandas as pd

from full_model import PricingEngine, PolicyPortfolio, MortalityTable, YieldCurve
from _data import data_path


class TestFullModel(unittest.TestCase):

    try:
        expense_spec = pd.read_csv('./../Parameters/expense_spec.csv')
    except FileNotFoundError:
        expense_spec = pd.read_csv('./Parameters/expense_spec.csv')

    policies_in = pd.read_csv(data_path('policy_data', 'mixed_policies.csv'))

    policies_in = policies_in.drop('premium', axis=1)
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

    expense_spec_per_pol_only = expense_spec.copy()[expense_spec['Basis'] == 'PER_POLICY']
    eng_per_pol_exp_only = PricingEngine(mortality, discount_curve, expense_spec_per_pol_only, expense_inflation_rate)

    expense_spec_pct_prem_only = expense_spec.copy()[expense_spec['Basis'] == 'PCT_PREMIUM']
    eng_pct_prem_exp_only = PricingEngine(mortality, discount_curve, expense_spec_pct_prem_only, expense_inflation_rate)


    def test_pricing_engine(self):

        expected = np.array([294.4919, 321.0385, 490.7803, 1334.534, 438.1998, 1100.275, 12677.79, 24088.09,
                             4652.497, 1421.843, 12722.83, 11510.22, 959828.2, 130964.4, 116709.7])

        actual = self.eng.price_policy_portfolio(self.policies_in)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_pricing_engine_exp_per_pol_only(self):

        expected = np.array([2.69949229e+02, 2.93884617e+02, 3.90801664e+02, 1.19709607e+03, 3.97496219e+02,
                             9.99924079e+02, 1.22557242e+04, 2.27890487e+04, 4.23665320e+03, 1.31912574e+03,
                             1.18573277e+04, 1.02531719e+04, 9.59828196e+05, 1.30964405e+05, 1.16709676e+05])

        actual = self.eng_per_pol_exp_only.price_policy_portfolio(self.policies_in)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_pricing_engine_exp_pct_prem_only(self):

        expected = np.array([2.57053967e+02, 2.87106400e+02, 4.48764192e+02, 1.25522536e+03, 3.70150807e+02,
                             1.03865257e+03, 1.26525768e+04, 2.40628078e+04, 4.62437872e+03, 1.36360368e+03,
                             1.26642493e+04, 1.14500860e+04, 9.57397667e+05, 1.29409078e+05, 1.15756967e+05])

        actual = self.eng_pct_prem_exp_only.price_policy_portfolio(self.policies_in)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

