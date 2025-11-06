import numpy as np

from modelic.core.cashflows import CompositeProduct
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.benefits.death_benefit import _DeathBenefit
from modelic.core.policy_portfolio import PolicyPortfolio


class LifeAssurance(CompositeProduct):

    """ Projects cashflows and calculates present values for death contingent benefits """

    def __init__(self, policy_data: PolicyPortfolio, yield_curve: YieldCurve, mortality_table: MortalityTable):
        components = [_DeathBenefit(policy_data, yield_curve, mortality_table)]
        super().__init__(components, yield_curve)
