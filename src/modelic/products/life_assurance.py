# modelic/products/life_assurance.py

from modelic.core.cashflows import CompositeProduct
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.contingent_cashflows.death_contingent_cashflow import DeathContingentCashflow
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.custom_types import ArrayLike, IntArrayLike


class LifeAssurance(CompositeProduct):

    """ Projects cashflows and calculates present values for death contingent contingent_cashflows """

    def __init__(self, yield_curve: YieldCurve, mortality_table: MortalityTable, age: IntArrayLike, term: IntArrayLike, death_benefit: ArrayLike):
        components = [DeathContingentCashflow(yield_curve, mortality_table, age, term, death_benefit)]
        super().__init__(components, yield_curve)

    @classmethod
    def from_policy_portfolio(cls, policy_portfolio: PolicyPortfolio, yield_curve: YieldCurve, mortality_table: MortalityTable):
        return cls(yield_curve, mortality_table, policy_portfolio.ages, policy_portfolio.terms, policy_portfolio.death_contingent_benefits)
