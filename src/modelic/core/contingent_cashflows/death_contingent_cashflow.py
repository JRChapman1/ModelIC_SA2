# modelic/core/contingent_cashflows/death_contingent_cashflows.py

import numpy as np

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.custom_types import ArrayLike, IntArrayLike


class _DeathContingentCashflow(BaseCashflowModel):
    """ Projects cashflows and calculates present values for death contingent contingent_cashflows """

    def __init__(self, yield_curve: YieldCurve, mortality_table: MortalityTable, ph_age: IntArrayLike, term: IntArrayLike,
                 death_contingent_cf: ArrayLike = 1, projection_steps: IntArrayLike = None):

        policy_terms = np.asarray(term)
        policy_terms[np.isnan(policy_terms)] = mortality_table.max_age - mortality_table.min_age
        policy_terms = policy_terms.astype(int)

        if projection_steps is None:
            projection_steps = np.arange(1, policy_terms.max() + 1)

        super().__init__(projection_steps, yield_curve)

        self.age = np.asarray(ph_age, dtype=int)
        self.term = policy_terms
        self.amount = np.asarray(death_contingent_cf, dtype=float)
        self.mortality = mortality_table
        self.discount_curve = yield_curve

    @classmethod
    def from_policy_portfolio(cls, policy_portfolio: PolicyPortfolio, yield_curve: YieldCurve,
                              mortality_table: MortalityTable) -> "_DeathContingentCashflow":
        return cls(yield_curve, mortality_table, policy_portfolio.ages, policy_portfolio.terms,
                   policy_portfolio.death_contingent_benefits)

    def project_cashflows(self, aggregate: bool = True) -> ArrayLike:

        cfs = np.zeros((self.times.size, self.age.size))

        death_in_year = self.mortality.nqx(self.age, self.term, full_path=True)
        cfs[(self.times >= 1) & (self.times <= self.term.max())] = self.amount * death_in_year

        return cfs.sum(axis=1) if aggregate else cfs