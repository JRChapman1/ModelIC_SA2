# modelic/expenses/expense_engine.py

import numpy as np
import pandas as pd

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.contingent_cashflows.survival_contingent_cashflow import _SurvivalContingentCashflow
from modelic.core.contingent_cashflows.death_contingent_cashflow import _DeathContingentCashflow
from modelic.core.custom_types import ArrayLike, IntArrayLike
from modelic.core.compounding import zero_to_df


class ExpenseTiming:
    INITIAL = 'INITIAL'
    RENEWAL = 'RENEWAL'
    SURVIVAL = 'SURVIVAL'
    DEATH = 'DEATH'


class ExpenseBasis:
    PER_POLICY = 'PER_POLICY'
    PCT_PREMIUM = 'PCT_PREMIUM'


class ExpenseEngine:

    def __init__(self, expense_spec: pd.DataFrame, yield_curve: YieldCurve, mortality_table: MortalityTable, expense_inflation_rate: float):
        self.expense_spec = expense_spec
        self.yield_curve = yield_curve
        self.mortality_table = mortality_table
        self.expense_inflation_rate = expense_inflation_rate


    def _get_t0_expense_amount(self, basis, amount, policy_count=1, annual_premium=1):

        match basis:
            case ExpenseBasis.PER_POLICY:
                t0_amount = np.repeat(amount, policy_count)
            case ExpenseBasis.PCT_PREMIUM:
                t0_amount = np.atleast_1d(amount * annual_premium)
            case _:
                raise ValueError('Unrecognized expense basis', basis)

        return t0_amount


    def _project_expense_cashflow_single(self, basis, expense_type, amount, ages, terms, premiums, annual_inflation_rate=0):

        num_cfs = self.mortality_table.ages.max() - self.mortality_table.ages.min() + 1
        times = np.arange(num_cfs)

        t0_amount = self._get_t0_expense_amount(basis, amount, np.atleast_1d(ages).size, premiums)

        if np.atleast_1d(t0_amount).sum() == 0:
            return np.zeros((num_cfs, 1))

        match expense_type:
            case ExpenseTiming.INITIAL:
                results = np.zeros(num_cfs)
                results[0] = t0_amount.sum()
            case ExpenseTiming.RENEWAL:
                surv_obj = _SurvivalContingentCashflow(self.yield_curve, self.mortality_table, ages, terms-1, periodic_cf=t0_amount, projection_steps=times)
                results = surv_obj.project_cashflows(aggregate=True)
            case ExpenseTiming.SURVIVAL:
                surv_obj = _SurvivalContingentCashflow(self.yield_curve, self.mortality_table, ages, terms, terminal_cf=t0_amount, projection_steps=times)
                results = surv_obj.project_cashflows(aggregate=True)
            case ExpenseTiming.DEATH:
                dth_obj = _DeathContingentCashflow(self.yield_curve, self.mortality_table, ages, terms, t0_amount, projection_steps=times)
                results = dth_obj.project_cashflows(aggregate=True)
            case _:
                raise ValueError('Unrecognized expense basis', expense_type)

        results *= (1 + annual_inflation_rate) ** times

        return results


    def project_cashflows(self, ages: IntArrayLike, terms: IntArrayLike,
                          premiums: IntArrayLike, aggregate: bool=True):

        results = pd.DataFrame(columns=pd.MultiIndex.from_tuples([], names=['Product', 'Description']))

        for _, expense_component in self.expense_spec.iterrows():

            cfs = self._project_expense_cashflow_single(expense_component['Basis'], expense_component['Type'],
                                                        expense_component['Amount'], ages, terms, premiums,
                                                        self.expense_inflation_rate)
            if cfs.sum() > 0:
                results = results.reindex(range(np.atleast_1d(cfs).size), fill_value=0.0)
                results[(expense_component['Product'],
                         f"{expense_component['Description']} ({expense_component['Type']})")] = cfs
        return results.sum(axis=1).values if aggregate else results

    def discount_factors(self, times, spread: ArrayLike = 0) -> np.ndarray:
        zeros = self.yield_curve.zero(times)[:, None] + np.atleast_2d(spread)
        return zero_to_df(times, zeros)


    def present_value(self, policy_data: PolicyPortfolio, *, group_by: str = None, unstack=False):

        num_cfs = self.mortality_table.ages.max() - self.mortality_table.ages.min() + 1
        times = np.arange(num_cfs)

        policies_x_expenses = pd.merge(policy_data.data, self.expense_spec, how="outer", left_on="policy_type",
                                       right_on="Product")

        amounts = np.where(policies_x_expenses['Basis']=='PCT_PREMIUM',
                           policies_x_expenses['Amount'] * policies_x_expenses['annual_premium'],
                           policies_x_expenses['Amount'])

        factors = np.zeros_like(amounts)

        # Initial expense factors

        factors[policies_x_expenses['Type'] == ExpenseTiming.INITIAL] = 1

        # Renewal expense factors

        renewal_mask = policies_x_expenses['Type'] == ExpenseTiming.RENEWAL
        ages = policies_x_expenses['ages'][renewal_mask]
        terms = policies_x_expenses['terms'][renewal_mask]
        surv_obj = _SurvivalContingentCashflow(self.yield_curve, self.mortality_table, ages, terms - 1, periodic_cf=1,
                                               projection_steps=times, escalation=self.expense_inflation_rate)

        factors[renewal_mask] = surv_obj.present_value(aggregate=False)

        # Maturity expense factors

        maturity_mask = policies_x_expenses['Type'] == ExpenseTiming.SURVIVAL
        ages = policies_x_expenses['ages'][maturity_mask]
        terms = policies_x_expenses['terms'][maturity_mask]
        surv_obj = _SurvivalContingentCashflow(self.yield_curve, self.mortality_table, ages, terms, terminal_cf=1,
                                               projection_steps=times, escalation=self.expense_inflation_rate)

        factors[maturity_mask] = surv_obj.present_value(aggregate=False)

        # Death expense factors

        death_mask = policies_x_expenses['Type'] == ExpenseTiming.DEATH
        ages = policies_x_expenses['ages'][death_mask]
        terms = policies_x_expenses['terms'][death_mask]
        surv_obj = _DeathContingentCashflow(self.yield_curve, self.mortality_table, ages, terms, death_contingent_cf=1,
                                            projection_steps=times, escalation=self.expense_inflation_rate)

        factors[death_mask] = surv_obj.present_value(aggregate=False)

        policies_x_expenses['Expense PV'] = (factors * amounts)

        expense_pvs = policies_x_expenses[['policy_id', 'Product', 'Type', 'Description', 'Basis', 'Expense PV']]

        if group_by is not None:
            if group_by == '*':
                expense_pvs = float(expense_pvs['Expense PV'].sum())
            else:
                expense_pvs = expense_pvs.groupby(group_by, as_index=False).sum()

        if unstack:
            assert len(group_by) == 2, "Can only unstack a table with exactly 2 grouping categories."
            expense_pvs = pd.Series(expense_pvs['Expense PV'].values, index=pd.MultiIndex.from_arrays((expense_pvs[group_by[0]], expense_pvs[group_by[1]])))
            expense_pvs = expense_pvs.unstack(fill_value=0.0)

        return expense_pvs




