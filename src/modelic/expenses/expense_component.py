from enum import Enum, auto
import numpy as np
import pandas as pd

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.policy_portfolio import PolicyPortfolio


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

if __name__ == '__main__':

    data = ExpenseEngine().from_csv('../../../Parameters/expense_spec.csv')