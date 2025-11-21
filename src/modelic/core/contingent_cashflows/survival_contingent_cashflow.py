import numpy as np

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.custom_types import ArrayLike, IntArrayLike
from modelic.core.policy_portfolio import PolicyPortfolio


class _SurvivalContingentCashflow(BaseCashflowModel):
    """ Projects cashflows and calculates present values for death contingent contingent_cashflows """

    def __init__(self, yield_curve: YieldCurve, mortality_table: MortalityTable, ph_age: IntArrayLike, term: IntArrayLike,
                 *, periodic_cf: ArrayLike = None, terminal_cf: ArrayLike = None):

        policy_terms = np.asarray(term)
        policy_terms[np.isnan(policy_terms)] = mortality_table.max_age - mortality_table.min_age
        policy_terms = policy_terms.astype(int)

        super().__init__(np.arange(1, policy_terms.max() + 1), yield_curve)

        self.age = np.asarray(ph_age, dtype=int)
        self.term = policy_terms
        self.periodic_amount = None if periodic_cf is None else np.asarray(periodic_cf, dtype=float)
        self.terminal_amount = None if terminal_cf is None else np.asarray(terminal_cf, dtype=float)
        self.mortality = mortality_table
        self.discount_curve = yield_curve

    @classmethod
    def from_policy_portfolio(cls, policy_portfolio: PolicyPortfolio, yield_curve: YieldCurve,
                              mortality_table: MortalityTable) -> "_SurvivalContingentCashflow":

        return cls(yield_curve,
                   mortality_table,
                   policy_portfolio.ages,
                   policy_portfolio.terms,
                   periodic_cf=policy_portfolio.periodic_survival_contingent_benefits,
                   terminal_cf=policy_portfolio.terminal_survival_contingent_benefits)

    def project_cashflows(self, aggregate: bool = True, proj_horizon = None) -> ArrayLike:

        if proj_horizon is None:
            proj_horizon = int(self.term.max())

        cfs = np.zeros((proj_horizon, self.age.size))
        if self.periodic_amount is not None:
            survival_path = self.mortality.npx(self.age, self.term, full_path=True)
            cfs += self.periodic_amount * survival_path

        if self.terminal_amount is not None:
            survival_to_year = self.mortality.npx(self.age, self.term, full_path=False)
            cfs[self.term - 1, np.arange(self.age.size)] += self.terminal_amount * survival_to_year

        return cfs.sum(axis=1).reshape(-1, 1) if aggregate else cfs