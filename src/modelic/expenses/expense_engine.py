from enum import Enum, auto
from unittest import case

import numpy as np
import pandas as pd

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.contingent_cashflows.survival_contingent_cashflow import _SurvivalContingentCashflow
from modelic.core.contingent_cashflows.death_contingent_cashflow import _DeathContingentCashflow


class ExpenseTiming:
    INITIAL = 'INITIAL'
    RENEWAL = 'RENEWAL'
    SURVIVAL = 'SURVIVAL'
    DEATH = 'DEATH'


class ExpenseBasis:
    PER_POLICY = 'PER_POLICY'
    PCT_PREMIUM = 'PCT_PREMIUM'


class ExpenseEngine(BaseCashflowModel):

    def __init__(self, expense_spec: pd.DataFrame, yield_curve: YieldCurve, mortality_table: MortalityTable, policy_data: PolicyPortfolio, expense_inflation_rate: float):

        policy_terms = np.asarray(policy_data.terms)
        policy_terms[np.isnan(policy_terms)] = mortality_table.max_age - mortality_table.min_age
        max_term = int(policy_terms.max() + 1)
        super().__init__(np.arange(0, max_term), yield_curve)

        self.expense_spec = expense_spec
        self.yield_curve = yield_curve
        self.mortality_table = mortality_table
        self.policy_data = policy_data
        self.expense_inflation_rate = expense_inflation_rate
        self.num_cfs = max_term

    def _get_t0_expense_amount(self, basis, amount, product_type):

        match basis:
            case ExpenseBasis.PER_POLICY:
                t0_amount = np.repeat(amount, self.policy_data.get('count', product_type))
            case ExpenseBasis.PCT_PREMIUM:
                t0_amount = amount * self.policy_data.get('annual_premium', product_type)
            case _:
                raise ValueError('Unrecognized expense basis', basis)

        return t0_amount


    def _project_expense_cashflow_single(self, basis, expense_type, amount, product_type, annual_inflation_rate=0):

        t0_amount = self._get_t0_expense_amount(basis, amount, product_type)

        if np.atleast_1d(t0_amount).sum() == 0:
            return np.zeros((self.num_cfs, 1))

        ages = self.policy_data.get('ages', product_type)
        terms = self.policy_data.get('terms', product_type)
        match expense_type:
            case ExpenseTiming.INITIAL:
                results = np.zeros((self.num_cfs, 1))
                results[0, 0] = t0_amount.sum()
            case ExpenseTiming.RENEWAL:
                surv_obj = _SurvivalContingentCashflow(self.yield_curve, self.mortality_table, ages, terms-1, periodic_cf=t0_amount, projection_steps=self.times)
                results = surv_obj.project_cashflows(aggregate=True)
            case ExpenseTiming.SURVIVAL:
                surv_obj = _SurvivalContingentCashflow(self.yield_curve, self.mortality_table, ages, terms, terminal_cf=t0_amount, projection_steps=self.times)
                results = surv_obj.project_cashflows(aggregate=True)
            case ExpenseTiming.DEATH:
                dth_obj = _DeathContingentCashflow(self.yield_curve, self.mortality_table, ages, terms, t0_amount, projection_steps=self.times)
                results = dth_obj.project_cashflows(aggregate=True)
            case _:
                raise ValueError('Unrecognized expense basis', expense_type)

        results *= (1 + annual_inflation_rate) ** self.times.reshape(results.shape)
        return results


    def project_cashflows(self, aggregate=True):

        results = pd.DataFrame(columns=pd.MultiIndex.from_tuples([], names=['Product', 'Description']))
        for _, expense_component in self.expense_spec.loc[self.expense_spec['Product'].isin(np.unique(self.policy_data.policy_type))].iterrows():
            cfs = self._project_expense_cashflow_single(expense_component['Basis'], expense_component['Type'], expense_component['Amount'], expense_component['Product'], self.expense_inflation_rate)
            if cfs.sum() > 0:
                results = results.reindex(range(np.atleast_1d(cfs).size), fill_value=0.0)
                results[(expense_component['Product'], expense_component['Type'] + expense_component['Description'])] = cfs
        return results



