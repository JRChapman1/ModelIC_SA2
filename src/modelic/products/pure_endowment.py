# modelic/products/pure_endowment.py

from modelic.core.cashflows import CompositeProduct
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.contingent_cashflows.survival_contingent_cashflow import _SurvivalContingentCashflow
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.custom_types import ArrayLike, IntArrayLike


class PureEndowment(CompositeProduct):

    """ Projects cashflows and calculates present values for death contingent contingent_cashflows """

    def __init__(self, yield_curve: YieldCurve, mortality_table: MortalityTable, age: IntArrayLike, term: IntArrayLike,
                 survival_benefit: ArrayLike):

        components = [_SurvivalContingentCashflow(yield_curve, mortality_table, age, term, terminal_cf=survival_benefit)]

        super().__init__(components, yield_curve)

    @classmethod
    def from_policy_portfolio(cls, policy_data: PolicyPortfolio, yield_curve: YieldCurve, mortality_table: MortalityTable):
        return cls(yield_curve, mortality_table, policy_data.ages, policy_data.terms,
                   policy_data.terminal_survival_contingent_benefits)


