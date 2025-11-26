# modelic/pricers/pricing_engine.py

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

from modelic.core.mortality import MortalityTable
from modelic.core.curves import YieldCurve
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.products.annuity import Annuity
from modelic.products.product_types import ProductType
from modelic.products.life_assurance import LifeAssurance
from modelic.products.endowment import Endowment
from modelic.products.pure_endowment import PureEndowment
from modelic.expenses.expense_engine import ExpenseEngine
from modelic.core.contingent_cashflows.survival_contingent_cashflow import _SurvivalContingentCashflow
from modelic.products.product_factory import PRODUCT_FACTORY
from modelic.core.custom_types import ArrayLike


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
        results = {}


        for policy_type in np.unique(policy_portfolio.policy_type):

            filtered_policies = policy_portfolio.subset(policy_type)

            product_engine = PRODUCT_FACTORY[policy_type].from_policy_portfolio(filtered_policies, self.yield_curves, self.mortality_table)

            ben_pvs = product_engine.present_value(aggregate=False)

            prices = self.price_policy_group(filtered_policies, ben_pvs)
            prices.index = prices.index.astype(int)

            results |= prices.to_dict()

        return results


    def price_policy(self, pol_type, ph_age, pol_term, premium_type, pv_bens):

        match premium_type:
            case 'Single':
                prem_ann_fac = 1
            case 'Regular':
                prem_ann_fac_obj = _SurvivalContingentCashflow(self.yield_curves, self.mortality_table,
                                                               ph_age, pol_term - 1, periodic_cf=1.0)
                prem_ann_fac = prem_ann_fac_obj.present_value() + 1
            case _:
                raise ValueError(f'premium_type {premium_type} not recognized')

        # TODO: Move this
        def objective_function(premium: float):
            pv_expenses = self.expense_engine.present_value(ph_age, pol_term, premium, product_type=pol_type)
            return pv_bens + pv_expenses - premium * prem_ann_fac

        # TODO: Fix bracket parameter
        return root_scalar(objective_function, bracket=[0, pv_bens * 2]).root


    def price_policy_group(self, policy_data: PolicyPortfolio, pv_bens: ArrayLike):

        reg_prem_pols = policy_data.premium_type == 'Regular'
        prem_ann_fac = np.ones(policy_data.ages.size)

        if reg_prem_pols.any():

            prem_ann_fac_obj = _SurvivalContingentCashflow(self.yield_curves, self.mortality_table,
                                                           policy_data.ages[reg_prem_pols], policy_data.terms[reg_prem_pols] - 1,
                                                           periodic_cf=1.0)

            prem_ann_fac[reg_prem_pols] += prem_ann_fac_obj.present_value()

        pv_expenses = self.expense_engine.present_value(policy_data, group_by=['policy_id', 'Basis'], unstack=True)
        premium = (pv_bens + pv_expenses['PER_POLICY']) / (prem_ann_fac - pv_expenses['PCT_PREMIUM'])
        """
        premium * prem_ann_fac = pv_bens + pv_expenses['PER_POLICY'] - premium * pv_expenses['PCT_PREMIUM']
        """

        return premium