# modelic/products/annuity.py

from modelic.core.cashflows import CompositeProduct
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.contingent_cashflows.survival_contingent_cashflow import _SurvivalContingentCashflow
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.custom_types import ArrayLike, IntArrayLike


class Annuity(CompositeProduct):

    """ Projects cashflows and calculates present values for death contingent contingent_cashflows """

    def __init__(self, yield_curve: YieldCurve, mortality_table: MortalityTable, age: IntArrayLike, term: IntArrayLike, annual_amount: ArrayLike):
        components = [_SurvivalContingentCashflow(yield_curve, mortality_table, age, term, periodic_cf=annual_amount)]
        super().__init__(components, yield_curve)

    @classmethod
    def from_policy_portfolio(cls, policy_data: PolicyPortfolio, yield_curve: YieldCurve, mortality_table: MortalityTable):
        return cls(yield_curve, mortality_table, policy_data.ages, policy_data.terms,
                   policy_data.periodic_survival_contingent_benefits)
