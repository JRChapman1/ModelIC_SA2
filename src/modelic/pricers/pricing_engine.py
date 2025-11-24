# modelic/pricers/pricing_engine.py

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

from modelic.core.mortality import MortalityTable
from modelic.core.curves import YieldCurve
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.products.annuity import Annuity
from modelic.products.life_assurance import LifeAssurance
from modelic.products.endowment import Endowment
from modelic.products.pure_endowment import PureEndowment
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
        self.expense_engine = ExpenseEngine(self.expense_spec, self.yield_curves, self.mortality_table,
                                            self.expense_inflation_rate)

    def price_policy_portfolio(self, policy_portfolio: PolicyPortfolio):

        # TODO: Ensure dict is best struct to accumulate results in
        results = []

        for idx, policy in policy_portfolio.data.iterrows():

            match policy['policy_type']:
                case PolicyType.Annuity:
                    product_engine = Annuity(self.yield_curves, self.mortality_table, policy['ages'], policy['terms'],
                                             policy['periodic_survival_contingent_benefits'])

                case PolicyType.Endowment:
                    product_engine = Endowment(self.yield_curves, self.mortality_table, policy['ages'], policy['terms'],
                                               policy['terminal_survival_contingent_benefits'], policy['death_contingent_benefits'])

                case PolicyType.PureEndowment:
                    product_engine = PureEndowment(self.yield_curves, self.mortality_table, policy['ages'],
                                                   policy['terms'], policy['terminal_survival_contingent_benefits'])

                case PolicyType.TermAssurance:
                    product_engine = LifeAssurance(self.yield_curves, self.mortality_table, policy['ages'],
                                                   policy['terms'], policy['death_contingent_benefits'])

                case PolicyType.WholeOfLifeAssurance:
                    product_engine = LifeAssurance(self.yield_curves, self.mortality_table, policy['ages'],
                                                   np.nan, policy['death_contingent_benefits'])

            term = np.nan if policy['terms'] == '' else policy['terms']
            results.append(self.price_policy(policy['policy_type'], policy['ages'], term,
                                                     policy['premium_type'], product_engine.present_value()))

        return results


    def price_policy(self, pol_type, ph_age, pol_term, premium_type, pv_bens):

        match premium_type:
            case 'Single':
                prem_ann_fac = 1
            case 'Regular':
                prem_ann_fac_obj = _SurvivalContingentCashflow(self.yield_curves, self.mortality_table,
                                                               ph_age, pol_term, periodic_cf=1.0)
                prem_ann_fac = prem_ann_fac_obj.present_value()
            case _:
                raise ValueError(f'premium_type {premium_type} not recognized')

        # TODO: Move this
        def objective_function(premium: float):
            pv_expenses = self.expense_engine.present_value_for_product(pol_type, ph_age, pol_term, premium)
            return pv_bens + pv_expenses - premium * prem_ann_fac

        # TODO: Fix bracket parameter
        return root_scalar(objective_function, bracket=[0, pv_bens * 2]).root