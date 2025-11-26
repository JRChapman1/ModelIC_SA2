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
            [[3.60000000e+02, 2.09434553e+03, 0.00000000e+00, 2.02109394e+02, 6.00e+01, 0.00000000e+00, 2.22210918e+03],
             [1.15447019e+04, 5.61106587e+02, 3.72583175e+03, 6.09408560e+00, 6.00e+01, 5.15268756e+01, 5.70133234e+02],
             [1.24255131e+04, 2.61608929e+02, 3.69441152e+03, 0.00000000e+00, 3.00e+01, 4.85249670e+01, 2.52011161e+02],
             [3.87208745e+02, 4.58137742e+02, 5.91642317e+02, 1.40814485e+01, 3.00e+01, 0.00000000e+00, 5.74706614e+02],
             [1.14920352e+03, 1.63672691e+03, 5.20709476e+03, 4.98998525e+02, 4.50e+01, 0.00000000e+00, 2.06230255e+03]])

        actual = self.eng.present_value(self.policies_in, group_by=['Product', 'Description'], unstack=True)

        assert actual.shape == expected_values.shape
        assert np.allclose(actual.values, expected_values)


    def test_expense_engine_pv_aggregated(self):

        expected = 50825.1368
        actual = self.eng.present_value(self.policies_in, group_by='*')

        assert np.allclose(actual, expected)


