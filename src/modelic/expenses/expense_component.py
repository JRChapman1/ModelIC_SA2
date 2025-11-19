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

        super().__init__(np.arange(1, policy_terms.max() + 1), yield_curve)

        self.expense_spec = expense_spec
        self.yield_curve = yield_curve
        self.mortality_table = mortality_table
        self.policy_data = policy_data
        self.expense_inflation_rate = expense_inflation_rate

    def _get_t0_expense_amount(self, basis, amount):

        match basis:
            case ExpenseBasis.PER_POLICY:
                t0_amount = np.repeat(amount, self.policy_data.count)
            case ExpenseBasis.PCT_PREMIUM:
                t0_amount = amount * self.policy_data.annual_premium
            case _:
                raise ValueError('Unrecognized expense basis', basis)

        return t0_amount


    def _project_expense_cashflow_single(self, basis, expense_type, amount, annual_inflation_rate=0):

        t0_amount = self._get_t0_expense_amount(basis, amount)

        if np.atleast_1d(t0_amount).sum() == 0:
            return 0

        match expense_type:
            case ExpenseTiming.INITIAL:
                return t0_amount.sum()
            case ExpenseTiming.RENEWAL:
                surv_obj = _SurvivalContingentCashflow(self.yield_curve, self.mortality_table, self.policy_data.ages, self.policy_data.terms, periodic_cf=t0_amount)
                results = surv_obj.project_cashflows(aggregate=True)
            case ExpenseTiming.SURVIVAL:
                surv_obj = _SurvivalContingentCashflow(self.yield_curve, self.mortality_table, self.policy_data.ages, self.policy_data.terms, terminal_cf=t0_amount)
                results = surv_obj.project_cashflows(aggregate=True)
            case ExpenseTiming.DEATH:
                dth_obj = _DeathContingentCashflow(self.yield_curve, self.mortality_table, self.policy_data.ages, self.policy_data.terms, t0_amount)
                results = dth_obj.project_cashflows(aggregate=True)
            case _:
                raise ValueError('Unrecognized expense basis', expense_type)

        results *= (1 + annual_inflation_rate) ** np.arange(1, results.shape[0] + 1).reshape(results.shape)
        return results


    def project_cashflows(self):

        results = pd.DataFrame()
        for _, expense_component in expense_spec.loc[expense_spec['Product'] == 'Term Assurance'].iterrows():
            cfs = self._project_expense_cashflow_single(expense_component['Basis'], expense_component['Type'], expense_component['Amount'])
            cfs = np.atleast_1d(cfs)
            if np.atleast_1d(cfs).sum():
                results = results.reindex(range(np.atleast_1d(cfs).size), fill_value=0.0)
                results[expense_component['Description']] = cfs
        return results



if __name__ == '__main__':

    expense_spec = pd.read_csv('../../../Parameters/expense_spec.csv')
    policies_in = PolicyPortfolio.from_csv('../../../tests/data/policy_data/ta_and_endowment_test_data.csv')

    mort_raw = pd.read_csv(r'../../../tests/data/mortality/AM92.csv')
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    disc_raw = pd.read_csv('../../../tests/data/curves/boe_spot_annual.csv')
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    for pol_type in np.unique(policies_in.policy_type):
        filtered_pols = policies_in.filter_on_product(pol_type)

        eng = ExpenseEngine(expense_spec, discount_curve, mortality, filtered_pols, 0.03)
        print(eng.project_cashflows())

    eng = ExpenseEngine(expense_spec, discount_curve, mortality, policies_in, 0.03)
    print(eng.project_cashflows())