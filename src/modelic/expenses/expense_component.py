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


class ExpenseTiming(Enum):
    INITIAL = auto()
    RENEWAL = auto()
    CLAIM = auto()
    SURRENDER = auto()
    MATURITY = auto()
    INVESTMENT = auto()


class ExpenseBasis(Enum):
    PER_POLICY = auto()
    PCT_PREMIUM = auto()
    PCT_SUM_ASSURED = auto()
    PCT_FUND = auto()


class ExpenseEngine:
    def from_csv(self, path: str):
        data = pd.read_csv(path)
        return data


class ExpenseComponent(BaseCashflowModel):

    def __init__(self, name: str, timing: ExpenseTiming, basis: ExpenseBasis, expense_amount: float,
                 inflation_rate: float = 0.0, directly_attributable: bool = True, policy_data: PolicyPortfolio = None,
                 fund_value: float = 0.0):
        
        self.name = name
        self.timing = timing
        self.basis = basis
        self.expense_amount = expense_amount
        self.inflation_rate = inflation_rate
        self.directly_attributable = directly_attributable
        self.policy_data = policy_data
        self.fund_value = fund_value

    def project_cashflows(self, aggregate: bool = False) -> BaseCashflowModel:

        amount = self._get_base_expense_amount()

    def _get_base_expense_amount(self) -> float:

        match self.basis:
            case ExpenseBasis.PER_POLICY:
                amount = np.repeat(self.expense_amount, self.policy_data.count)# TODO: tile with self.num_policies
            case ExpenseBasis.PCT_PREMIUM:
                amount = self.expense_amount * self.policy_data.p
            case ExpenseBasis.PCT_SUM_ASSURED:
                amount = self.expense_amount * self.sum_assured
            case ExpenseBasis.PCT_FUND:
                amount = self.expense_amount * self.fund_value
            case '_':
                raise ValueError('Unrecognized expense basis')

def get_t0_expense_amount(basis, amount, policies):
    match basis:
        case 'PER_POLICY':
            t0_amount = np.repeat(amount, policies.count)
        case 'PCT_PREMIUM':
            t0_amount = amount * policies.annual_premium
        case _:
            raise ValueError('Unrecognized expense basis', basis)

    return t0_amount

expense_type_overrides = {
    'Term Assurance': {'CLAIM': 'DEATH'}
}

def project_expense_cashflow_single(basis, expense_type, amount, policies, yield_curve, mortality_table, policy_type, annual_inflation_rate=0):

    t0_amount = get_t0_expense_amount(basis, amount, policies)

    if np.atleast_1d(t0_amount).sum() == 0:
        return 0

    if expense_type in expense_type_overrides[policy_type]:
        expense_type = expense_type_overrides[policy_type][expense_type]

    match expense_type:
        case 'INITIAL':
            return t0_amount.sum()
        case 'RENEWAL':
            surv_obj = _SurvivalContingentCashflow(yield_curve, mortality_table, policies.ages, policies.terms, periodic_cf=t0_amount)
            results = surv_obj.project_cashflows(aggregate=True)
        case 'SURVIVAL':
            surv_obj = _SurvivalContingentCashflow(yield_curve, mortality_table, policies.ages, policies.terms, terminal_cf=t0_amount)
            results = surv_obj.project_cashflows(aggregate=True)
        case 'DEATH':
            dth_obj = _DeathContingentCashflow(yield_curve, mortality_table, policies.ages, policies.terms, t0_amount)
            results = dth_obj.project_cashflows(aggregate=True)
        case _:
            raise ValueError('Unrecognized expense basis', expense_type)

    results *= (1 + annual_inflation_rate) ** np.arange(1, 4).reshape(results.shape)
    return results


def project_expense_cashflows(expense_spec, discount_curve, mortality, policies, policy_type, annual_inflation_rate=0):

    results = pd.DataFrame()
    for _, expense_component in expense_spec.loc[expense_spec['Product'] == 'Term Assurance'].iterrows():
        cfs = project_expense_cashflow_single(expense_component['Basis'], expense_component['Type'], expense_component['Amount'], policies, discount_curve, mortality, policy_type, annual_inflation_rate)
        cfs = np.atleast_1d(cfs)
        if np.atleast_1d(cfs).sum():
            results = results.reindex(range(np.atleast_1d(cfs).size), fill_value=0.0)
            results[expense_component['Description']] = cfs
    return results



if __name__ == '__main__':

    expense_spec = ExpenseEngine().from_csv('../../../Parameters/expense_spec.csv')
    policies_in = PolicyPortfolio.from_csv('../../../tests/data/policy_data/ta_test_data.csv')

    mort_raw = pd.read_csv(r'../../../tests/data/mortality/AM92.csv')
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    disc_raw = pd.read_csv('../../../tests/data/curves/boe_spot_annual.csv')
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    foo = project_expense_cashflows(expense_spec.loc[expense_spec['Product'] == 'Term Assurance'], discount_curve, mortality, policies_in, 'Term Assurance', annual_inflation_rate=0.03)
    print(foo)
    print('done')