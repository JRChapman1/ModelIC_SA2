# modelic/products/pure_endowment.py

from modelic.core.cashflows import CompositeProduct
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.contingent_cashflows.survival_contingent_cashflow import SurvivalContingentCashflow
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.custom_types import ArrayLike, IntArrayLike


class PureEndowment(CompositeProduct):

    """ Projects cashflows and calculates present values for death contingent contingent_cashflows """

    def __init__(self, yield_curve: YieldCurve, mortality_table: MortalityTable, age: IntArrayLike, term: IntArrayLike,
                 survival_benefit: ArrayLike):

        components = [SurvivalContingentCashflow(yield_curve, mortality_table, age, term, terminal_cf=survival_benefit)]

        super().__init__(components, yield_curve)

    @classmethod
    def from_policy_portfolio(cls, policy_data: PolicyPortfolio, yield_curve: YieldCurve, mortality_table: MortalityTable,
                              *, policy_mask: ArrayLike = None):

        ages = policy_data.ages
        terms = policy_data.terms
        terminal_survival_contingent_benefits = policy_data.terminal_survival_contingent_benefits

        if policy_mask is not None:
            assert policy_mask.size == policy_data.count, "Policy mask shape does not match policy mask shape"
            ages = ages[policy_mask]
            terms = terms[policy_mask]
            terminal_survival_contingent_benefits = terminal_survival_contingent_benefits[policy_mask]

        return cls(yield_curve, mortality_table, ages, terms, terminal_survival_contingent_benefits)


