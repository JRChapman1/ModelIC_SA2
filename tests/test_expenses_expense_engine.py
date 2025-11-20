import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.expenses.expense_engine import ExpenseEngine
from _data import data_path


class TestExpenseEngine(unittest.TestCase):

        expense_spec = pd.read_csv('../Parameters/expense_spec.csv')
        policies_in = PolicyPortfolio.from_csv(data_path('policy_data', 'ta_and_endowment_test_data.csv'))

        mort_raw = pd.read_csv(data_path('mortality', 'AM92.csv'))
        ages = mort_raw['x'].to_numpy(int)
        qx = mort_raw['q_x'].to_numpy(float)
        mortality = MortalityTable(ages, qx, 'AM92')

        disc_raw = pd.read_csv(data_path('curves', 'boe_spot_annual.csv'))
        times = disc_raw['year'].to_numpy(int)
        zeros = disc_raw['rate'].to_numpy(float)
        discount_curve = YieldCurve(times, zeros, 'BoE')

        expense_inflation_rate = 0.03

        def test_expense_engine(self):
                eng = ExpenseEngine(self.expense_spec, self.discount_curve, self.mortality, self.policies_in, self.expense_inflation_rate)
                foo = eng.project_cashflows(aggregate=False)
                pv = eng.present_value(aggregate=False)
                pv_agg = eng.present_value(aggregate=True)
                print(0)
