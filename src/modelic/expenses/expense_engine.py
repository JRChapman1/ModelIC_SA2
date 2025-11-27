# modelic/expenses/expense_engine.py

import numpy as np
import pandas as pd

from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.contingent_cashflows.survival_contingent_cashflow import SurvivalContingentCashflow
from modelic.core.contingent_cashflows.death_contingent_cashflow import DeathContingentCashflow
from modelic.expenses.expense_bases import ExpenseBasis
from modelic.expenses.expense_timings import ExpenseTiming


class ExpenseEngine:

    def __init__(self, expense_spec: pd.DataFrame, yield_curve: YieldCurve, mortality_table: MortalityTable, expense_inflation_rate: float):
        self.expense_spec = expense_spec
        self.yield_curve = yield_curve
        self.mortality_table = mortality_table
        self.expense_inflation_rate = expense_inflation_rate


    def present_value(self, policy_data: PolicyPortfolio, *, group_by: str = None, unstack=False):

        num_cfs = self.mortality_table.ages.max() - self.mortality_table.ages.min() + 1
        times = np.arange(num_cfs)

        policies_x_expenses = pd.merge(policy_data.data, self.expense_spec, how="outer", left_on="policy_type",
                                       right_on="Product")

        amounts = np.where(policies_x_expenses['Basis']==ExpenseBasis.PCT_PREMIUM,
                           policies_x_expenses['Amount'] * policies_x_expenses['annual_premium'],
                           policies_x_expenses['Amount'])

        factors = np.zeros_like(amounts)

        # Initial expense factors

        factors[policies_x_expenses['Type'] == ExpenseTiming.INITIAL] = 1

        # Renewal expense factors

        renewal_mask = policies_x_expenses['Type'] == ExpenseTiming.RENEWAL
        ages = policies_x_expenses['ages'][renewal_mask]
        terms = policies_x_expenses['terms'][renewal_mask]
        surv_obj = SurvivalContingentCashflow(self.yield_curve, self.mortality_table, ages, terms - 1, periodic_cf=1,
                                              projection_steps=times, escalation=self.expense_inflation_rate)

        factors[renewal_mask] = surv_obj.present_value(aggregate=False)

        # Maturity expense factors

        maturity_mask = policies_x_expenses['Type'] == ExpenseTiming.SURVIVAL
        ages = policies_x_expenses['ages'][maturity_mask]
        terms = policies_x_expenses['terms'][maturity_mask]
        surv_obj = SurvivalContingentCashflow(self.yield_curve, self.mortality_table, ages, terms, terminal_cf=1,
                                              projection_steps=times, escalation=self.expense_inflation_rate)

        factors[maturity_mask] = surv_obj.present_value(aggregate=False)

        # Death expense factors

        death_mask = policies_x_expenses['Type'] == ExpenseTiming.DEATH
        ages = policies_x_expenses['ages'][death_mask]
        terms = policies_x_expenses['terms'][death_mask]
        surv_obj = DeathContingentCashflow(self.yield_curve, self.mortality_table, ages, terms, death_contingent_cf=1,
                                           projection_steps=times, escalation=self.expense_inflation_rate)

        factors[death_mask] = surv_obj.present_value(aggregate=False)

        policies_x_expenses['Expense PV'] = (factors * amounts)

        if group_by is not None:
            if group_by == '*':
                policies_x_expenses = float(policies_x_expenses['Expense PV'].sum())
            else:
                policies_x_expenses = policies_x_expenses.groupby(group_by, as_index=False).sum()

        if unstack:
            assert len(group_by) == 2, "Can only unstack a table with exactly 2 grouping categories."
            multi_idx = pd.MultiIndex.from_arrays((policies_x_expenses[group_by[0]], policies_x_expenses[group_by[1]]))
            policies_x_expenses = pd.Series(policies_x_expenses['Expense PV'].values, index=multi_idx)
            policies_x_expenses = policies_x_expenses.unstack(fill_value=0.0)

        return policies_x_expenses




