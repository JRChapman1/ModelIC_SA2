import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.expenses.expense_engine import ExpenseEngine
from _data import data_path


class TestExpenseEngine(unittest.TestCase):

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

    eng = ExpenseEngine(expense_spec, discount_curve, mortality, expense_inflation_rate)


    def test_expense_engine_pv(self):

        expected_values = np.array(
            [[3.38523829e+02, 7.50000000e+01, 3.00000000e+01, 3.83137742e+02, 5.31583030e+02, 5.74706614e+02, 1.40814485e+01, 0.00000000e+00],
             [1.11619495e+03, 9.00000000e+01, 4.50000000e+01, 1.54672691e+03, 4.99746960e+03, 2.06230255e+03, 4.98998525e+02, 0.00000000e+00],
             [1.16685361e+04, 1.05000000e+02, 6.00000000e+01, 4.56106587e+02, 3.79683768e+03, 5.70133234e+02, 6.09408560e+00, 5.15268756e+01],
             [1.31971729e+04, 6.00000000e+01, 3.00000000e+01, 2.01608929e+02, 3.94156079e+03, 2.52011161e+02, 0.00000000e+00, 4.85249670e+01],
             [3.60000000e+02, 1.50000000e+02, 6.00000000e+01, 1.94434553e+03, 0.00000000e+00, 2.22210918e+03, 2.02109394e+02, 0.00000000e+00]])

        actual = self.eng.present_value(self.policies_in, aggregate=False)

        assert actual.shape == expected_values.shape
        assert np.allclose(actual.values, expected_values)


    def test_expense_engine_pv_aggregated(self):

        expected = 51687.40256545101
        actual = self.eng.present_value(self.policies_in, aggregate=True)

        assert np.allclose(actual, expected)


    def test_expense_engine_pv_term_assurance(self):

        expected_values = np.array([3.38523829e+02, 7.50000000e+01, 3.00000000e+01, 3.83137742e+02, 5.31583030e+02,
                                    5.74706614e+02, 1.40814485e+01])

        product_type = 'Term Assurance'
        ages = self.policies_in.get('ages', product_type)
        terms = self.policies_in.get('terms', product_type)
        premiums = self.policies_in.get('annual_premium', product_type)

        actual = self.eng.present_value_for_product(product_type, ages, terms, premiums, aggregate=False)

        assert actual.shape == expected_values.shape
        assert np.allclose(actual.values, expected_values)


    def test_expense_engine_pv_term_assurance_aggregated(self):

        expected = 1947.0326635

        product_type = 'Term Assurance'
        ages = self.policies_in.get('ages', product_type)
        terms = self.policies_in.get('terms', product_type)
        premiums = self.policies_in.get('annual_premium', product_type)

        actual = self.eng.present_value_for_product(product_type, ages, terms, premiums, aggregate=True)

        assert np.allclose(actual, expected)

