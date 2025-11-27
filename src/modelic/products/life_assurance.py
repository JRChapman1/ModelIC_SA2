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
    def from_policy_portfolio(cls, policy_data: PolicyPortfolio, yield_curve: YieldCurve, mortality_table: MortalityTable,
                              *, policy_mask: ArrayLike = None):

        ages = policy_data.ages
        terms = policy_data.terms
        death_contingent_benefits = policy_data.death_contingent_benefits

        if policy_mask is not None:
            assert policy_mask.size == policy_data.count, "Policy mask shape does not match policy mask shape"
            ages = ages[policy_mask]
            terms = terms[policy_mask]
            death_contingent_benefits = death_contingent_benefits[policy_mask]


        return cls(yield_curve, mortality_table, ages, terms, death_contingent_benefits)

