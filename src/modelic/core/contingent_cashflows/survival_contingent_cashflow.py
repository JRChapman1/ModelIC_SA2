# modelic/core/contingent_cashflows/survival_contingent_cashflows.py

import numpy as np

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.custom_types import ArrayLike, IntArrayLike
from modelic.core.policy_portfolio import PolicyPortfolio


class SurvivalContingentCashflow(BaseCashflowModel):
    """ Projects cashflows and calculates present values for death contingent contingent_cashflows """

    def __init__(self, yield_curve: YieldCurve, mortality_table: MortalityTable, ph_age: IntArrayLike, term: IntArrayLike,
                 *, periodic_cf: ArrayLike = None, terminal_cf: ArrayLike = None, projection_steps: IntArrayLike = None,
                 escalation: float = 0.0):

        term = np.nan_to_num(np.asarray(term, dtype=np.float64), nan=mortality_table.max_age-mortality_table.min_age).astype(int)
        ph_age = np.asarray(ph_age)

        if projection_steps is None:
            projection_steps = np.arange(1, term.max() + 1)

        super().__init__(projection_steps, yield_curve)

        self.age = np.asarray(ph_age, dtype=int)
        self.term = term
        self.periodic_amount = None if periodic_cf is None else np.asarray(periodic_cf, dtype=float)
        self.terminal_amount = None if terminal_cf is None else np.asarray(terminal_cf, dtype=float)
        self.mortality = mortality_table
        self.discount_curve = yield_curve
        self.escalation = escalation

    @classmethod
    def from_policy_portfolio(cls, policy_data: PolicyPortfolio, yield_curve: YieldCurve,
                              mortality_table: MortalityTable, *, projection_steps: IntArrayLike = None,
                              policy_mask: bool = None) -> "SurvivalContingentCashflow":

        ages = policy_data.ages
        terms = policy_data.terms
        periodic_survival_contingent_benefits = policy_data.periodic_survival_contingent_benefits
        terminal_survival_contingent_benefits = policy_data.terminal_survival_contingent_benefits

        if policy_mask is not None:
            assert policy_mask.size == policy_data.count, "Policy mask shape does not match policy mask shape"
            ages = ages[policy_mask]
            terms = terms[policy_mask]
            periodic_survival_contingent_benefits = periodic_survival_contingent_benefits[policy_mask]
            terminal_survival_contingent_benefits = terminal_survival_contingent_benefits[policy_mask]

        return cls(yield_curve,
                   mortality_table,
                   ages,
                   terms,
                   periodic_cf=periodic_survival_contingent_benefits,
                   terminal_cf=terminal_survival_contingent_benefits,
                   projection_steps=projection_steps)

    def project_cashflows(self, aggregate: bool = True) -> ArrayLike:

        cfs = np.zeros((self.times.size, self.age.size))
        if self.periodic_amount is not None:
            survival_path = self.mortality.npx(self.age, self.term, full_path=True)
            cfs[(self.times >= 1) & (self.times <= self.term.max())] += self.periodic_amount * survival_path

        if self.terminal_amount is not None:
            survival_to_year = self.mortality.npx(self.age, self.term, full_path=False)
            cfs[self.term - self.times.min(), np.arange(self.age.size)] += self.terminal_amount * survival_to_year

        cfs *= (1 + self.escalation) ** np.tile(self.times.reshape(-1, 1), self.age.size)

        return cfs.sum(axis=1) if aggregate else cfs
