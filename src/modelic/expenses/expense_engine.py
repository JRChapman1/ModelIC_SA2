# modelic/expenses/expense_engine.py

import numpy as np
import pandas as pd

from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.expenses.expense_bases import ExpenseBasis
from modelic.expenses.expense_timings import ExpenseTiming
from modelic.expenses.expense_factor_factory import EXPENSE_FACTOR_FACTORY


class ExpenseEngine:

    def __init__(self, expense_spec: pd.DataFrame, yield_curve: YieldCurve, mortality_table: MortalityTable, expense_inflation_rate: float):
        self.expense_spec = expense_spec
        self.yield_curve = yield_curve
        self.mortality_table = mortality_table
        self.expense_inflation_rate = expense_inflation_rate


    def present_value(self, policy_data: PolicyPortfolio, *, group_by: str = None, unstack=False):

        num_cfs = self.mortality_table.ages.max() - self.mortality_table.ages.min() + 1
        times = np.arange(num_cfs)

        expenses_table = pd.merge(policy_data.data, self.expense_spec, how="inner", left_on="policy_type",
                                       right_on="Product")

        basis_col = expenses_table['Basis'].to_numpy()
        amount_col = expenses_table['Amount'].to_numpy()
        premium_col = expenses_table['premiums'].to_numpy()
        timing_col = expenses_table['Type'].to_numpy()
        ages_col = expenses_table['ages'].to_numpy()
        terms_col = expenses_table['terms'].to_numpy()

        amounts = np.where(basis_col==ExpenseBasis.PCT_PREMIUM, amount_col * premium_col, amount_col)

        factors = np.zeros_like(amounts)
        for expense_timing in np.unique(timing_col):
            factors[timing_col == expense_timing] = self._get_expense_factors(timing_col, ages_col, terms_col,
                                                                              expense_timing, times)

        expenses_table['Expense PV'] = (factors * amounts)

        if group_by is not None:
            
            if group_by == '*':
                
                if unstack:
                    raise ValueError("Cannot use unstack=True with group_by='*' - too many columns to unpack")
                expenses_table = float(expenses_table['Expense PV'].sum())
                
            else:
                
                expenses_table = expenses_table.groupby(group_by, as_index=False).sum()

                if unstack:
                    assert len(group_by) == 2, "Can only unstack a table with exactly 2 grouping categories."
                    multi_idx = pd.MultiIndex.from_arrays((expenses_table[group_by[0]], expenses_table[group_by[1]]))
                    expenses_table = pd.Series(expenses_table['Expense PV'].values, index=multi_idx)
                    expenses_table = expenses_table.unstack(fill_value=0.0)

        return expenses_table


    def _get_expense_factors(self, expense_timing_col: np.ndarray, ages: np.ndarray, terms: np.ndarray, expense_timing: ExpenseTiming, times: np.ndarray):

        if expense_timing == ExpenseTiming.INITIAL:
            return 1.0

        mask = (expense_timing_col == expense_timing)

        if mask.any():

            factory = EXPENSE_FACTOR_FACTORY[expense_timing]

            ages = ages[mask]
            terms = terms[mask] + factory['term_offset']
            class_constructor = factory['cls']
            class_kwargs = factory['kwargs']

            surv_obj = class_constructor(self.yield_curve, self.mortality_table, ages, terms, projection_steps=times,
                                         escalation=self.expense_inflation_rate, **class_kwargs)

            return surv_obj.present_value(aggregate=False)

        else:

            return []

