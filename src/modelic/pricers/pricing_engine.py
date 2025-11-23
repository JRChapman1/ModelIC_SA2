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
        results = []

        for policy_type in np.unique(policy_portfolio.policy_type):

            policy_data = policy_portfolio.subset(policy_type)

            match policy_type:
                case PolicyType.Annuity:

                    for age, term, annual_amount in zip(policy_data.ages, policy_data.terms, policy_data.periodic_survival_contingent_benefits):
                        product_engine = Annuity(self.yield_curves, self.mortality_table, age, term, annual_amount)
                        expense_engine = ExpenseEngine(self.expense_spec, self.yield_curves, self.mortality_table,
                                                       self.expense_inflation_rate)
                        prem_ann_fac_obj = _SurvivalContingentCashflow(self.yield_curves, self.mortality_table, age, term,
                                                               periodic_cf=1.0)

                        pv_bens = product_engine.present_value()
                        prem_ann_fac = prem_ann_fac_obj.present_value()

                        # TODO: Move this
                        def objective_function(premium: float):
                            pv_expenses = expense_engine.present_value_for_product(policy_type, age, term, premium)
                            return pv_bens + pv_expenses - premium * prem_ann_fac

                        results.append(root_scalar(objective_function, bracket=[0, pv_bens * 2]).root)

                case PolicyType.Endowment:
                    pass
                case PolicyType.PureEndowment:
                    pass
                case PolicyType.TermAssurance:
                    pass
                case PolicyType.WholeOfLifeAssurance:
                    pass

            return results