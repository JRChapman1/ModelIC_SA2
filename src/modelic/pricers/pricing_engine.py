# modelic/pricers/pricing_engine.py

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

from modelic.core.mortality import MortalityTable
from modelic.core.curves import YieldCurve
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.products.annuity import Annuity
from modelic.expenses.expense_engine import ExpenseEngine
from modelic.core.contingent_cashflows.survival_contingent_cashflow import _SurvivalContingentCashflow


# TODO: Move to seperate module
class PolicyType:
    Annuity = 'Annuity'
    Endowment = 'Endowment'
    PureEndowment = 'Pure Endowment'
    TermAssurance = 'Term Assurance'
    WholeOfLifeAssurance = 'Whole-of-Life Assurance'

class PricingEngine:

    def __init__(self, mortality_table: MortalityTable, yield_curves: YieldCurve, expense_spec: pd.DataFrame, expense_inflation_rate: float):
        self.mortality_table = mortality_table
        self.yield_curves = yield_curves
        self.expense_spec = expense_spec
        self.expense_inflation_rate = expense_inflation_rate

    def price_policies(self, policy_portfolio: PolicyPortfolio):

        # TODO: Ensure dict is best struct to accumulate results in
        results = {}

        for policy_type in np.unique(policy_portfolio.policy_type):

            policy_data = policy_portfolio.subset(policy_type)

            match policy_type:
                case PolicyType.Annuity:

                    for age, n in zip(policy_data.ages, policy_data.terms):
                        # TODO: Move this
                        def objective_function(premium: float):
                            product_engine = Annuity(policy_data, self.yield_curves, self.mortality_table)
                            expense_engine = ExpenseEngine(self.expense_spec, self.yield_curves, self.mortality_table,
                                                           policy_data, self.expense_inflation_rate)
                            pv_prems = _SurvivalContingentCashflow(self.yield_curves, self.mortality_table, age, n,
                                                                   periodic_cf=premium)

                            return product_engine.present_value() + expense_engine.present_value() - pv_prems.present_value()


                case PolicyType.Endowment:
                    pass
                case PolicyType.PureEndowment:
                    pass
                case PolicyType.TermAssurance:
                    pass
                case PolicyType.WholeOfLifeAssurance:
                    pass
