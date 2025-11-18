from modelic.core.cashflows import CompositeProduct
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.contingent_cashflows.survival_contingent_cashflow import _SurvivalContingentCashflow
from modelic.core.policy_portfolio import PolicyPortfolio


class PureEndowment(CompositeProduct):

    """ Projects cashflows and calculates present values for death contingent contingent_cashflows """

    def __init__(self, policy_data: PolicyPortfolio, yield_curve: YieldCurve, mortality_table: MortalityTable):
        components = [_SurvivalContingentCashflow.from_policy_portfolio(policy_data, yield_curve, mortality_table)]
        super().__init__(components, yield_curve)
