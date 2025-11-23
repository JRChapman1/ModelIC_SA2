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
                t0_amount = amount * annual_premium
            case _:
                raise ValueError('Unrecognized expense basis', basis)

        return t0_amount


    def _project_expense_cashflow_single(self, basis, expense_type, amount, ages, terms, premiums, annual_inflation_rate=0):

        policy_count = ages.size
        num_cfs = self.mortality_table.ages.max() - self.mortality_table.ages.min() + 1
        times = np.arange(num_cfs)

        t0_amount = self._get_t0_expense_amount(basis, amount, policy_count, premiums)

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


    def project_cashflows(self, policy_data: PolicyPortfolio, aggregate=True):

        results = pd.DataFrame(columns=pd.MultiIndex.from_tuples([], names=['Product', 'Description']))

        # TODO: Is this loop necessay?
        expense_spec = self.expense_spec.loc[self.expense_spec['Product'].isin(np.unique(policy_data.policy_type))]
        for _, expense_component in expense_spec.iterrows():

            premiums = policy_data.get('annual_premium', expense_component['Product'])
            ages = policy_data.get('ages', expense_component['Product'])
            terms = policy_data.get('terms', expense_component['Product'])

            cfs = self._project_expense_cashflow_single(expense_component['Basis'], expense_component['Type'],
                                                        expense_component['Amount'], ages, terms, premiums,
                                                        self.expense_inflation_rate)
            if cfs.sum() > 0:
                results = results.reindex(range(np.atleast_1d(cfs).size), fill_value=0.0)
                results[(expense_component['Product'],  f"{expense_component['Description']} ({expense_component['Type']})")] = cfs
        return results.sum(axis=1).values if aggregate else results


    def project_cashflows_for_product(self, product_type: str, ages: IntArrayLike, terms: IntArrayLike,
                                      premiums: IntArrayLike, aggregate=True):

        results = pd.DataFrame()

        # TODO: Is this loop necessay?
        expense_spec = self.expense_spec.loc[self.expense_spec['Product'] == product_type]
        for _, expense_component in expense_spec.iterrows():

            cfs = self._project_expense_cashflow_single(expense_component['Basis'], expense_component['Type'],
                                                        expense_component['Amount'], ages, terms, premiums,
                                                        self.expense_inflation_rate)
            if cfs.sum() > 0:
                results = results.reindex(range(np.atleast_1d(cfs).size), fill_value=0.0)
                results[f"{expense_component['Description']} ({expense_component['Type']})"] = cfs

        return results.sum(axis=1).values if aggregate else results


    def discount_factors(self, times, spread: ArrayLike = 0) -> np.ndarray:
        zeros = self.yield_curve.zero(times)[:, None] + np.atleast_2d(spread)
        return zero_to_df(times, zeros)


    def present_value(self, policy_data, spread: ArrayLike = 0.0, aggregate: bool = True):

        cf = self.project_cashflows(policy_data, aggregate=False)
        df = self.discount_factors(np.arange(cf.shape[0]), spread)
        pv = (df * cf).sum(axis=0)

        if type(pv) is pd.Series:
            return float(pv.sum()) if aggregate else pv.unstack(fill_value=0)
        else:
            return float(pv.sum()) if aggregate else pv


    def present_value_for_product(self, product_type: str, ages: IntArrayLike, terms: IntArrayLike,
                                  premiums: IntArrayLike, spread=0.0, aggregate=True):

        cf = self.project_cashflows_for_product(product_type, ages, terms, premiums, aggregate=False)
        df = self.discount_factors(np.arange(cf.shape[0]), spread)
        pv = (df * cf).sum(axis=0)

        return float(pv.sum()) if aggregate else pv



