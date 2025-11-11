import numpy as np

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.custom_types import ArrayLike


class _SurvivalBenefit(BaseCashflowModel):
    """ Projects cashflows and calculates present values for death contingent benefits """

    def __init__(self, policy_data: PolicyPortfolio, yield_curve: YieldCurve, mortality_table: MortalityTable):

        policy_terms = policy_data.terms
        policy_terms[np.isnan(policy_data.terms)] = mortality_table.max_age - mortality_table.min_age

        super().__init__(np.arange(1, policy_terms.max() + 1), yield_curve)

        self.age = policy_data.ages
        self.term = policy_terms
        self.periodic_amount = policy_data.periodic_survival_contingent_benefits
        self.terminal_amount = policy_data.terminal_survival_contingent_benefits
        self.mortality = mortality_table
        self.discount_curve = yield_curve

    def project_cashflows(self, aggregate: bool = True) -> ArrayLike:

        survival_to_year = self.mortality.npx(self.age, self.term, full_path=True)

        cfs = 0
        if self.periodic_amount is not None:
            cfs += self.periodic_amount * survival_to_year
        if self.terminal_amount is not None:
            cfs += self.terminal_amount * survival_to_year[-1, :]

        return cfs.sum(axis=1).reshape(-1, 1) if aggregate else cfs