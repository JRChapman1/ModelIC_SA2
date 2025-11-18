import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.expenses.expense_component import ExpenseComponent, ExpenseTiming, ExpenseBasis
from _data import data_path


class TestExpenseComponent(unittest.TestCase):

    mort_raw = pd.read_csv(data_path("mortality", "AM92.csv"))
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    disc_raw = pd.read_csv(data_path("curves", "boe_spot_annual.csv"))
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    wol_policies = PolicyPortfolio.from_csv(data_path("policy_data", "wol_test_data.csv"))
    term_policies = PolicyPortfolio.from_csv(data_path("policy_data", "ta_test_data.csv"))

    expense_component = ExpenseComponent("name", ExpenseTiming.RENEWAL, ExpenseBasis.PCT_PREMIUM, 0.025, 0.03, True, wol_policies)