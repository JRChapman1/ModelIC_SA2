import numpy as np

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.custom_types import ArrayLike


class _DeathBenefit(BaseCashflowModel):
    """ Projects cashflows and calculates present values for death contingent benefits """

    def __init__(self, policy_data: PolicyPortfolio, yield_curve: YieldCurve, mortality_table: MortalityTable):

        policy_data = policy_data

        # TODO: Expects all policies to have same term
        if policy_data.terms is None:
            term = mortality_table.max_age - mortality_table.min_age
        else:
            term = int(policy_data.terms[0])


        super().__init__(np.arange(1, term + 1), yield_curve)

        self.age = policy_data.ages
        self.term = term
        self.amount = policy_data.death_contingent_benefits
        self.mortality = mortality_table
        self.discount_curve = yield_curve

    def project_cashflows(self, aggregate: bool = True) -> ArrayLike:

        death_in_year = self.mortality.nqx(self.age, self.term, True)
        cfs = self.amount * death_in_year

        return cfs.sum(axis=1).reshape(-1, 1) if aggregate else cfs