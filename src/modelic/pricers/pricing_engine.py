# modelic/pricers/pricing_engine.py

import numpy as np
import pandas as pd

from modelic.core.mortality import MortalityTable
from modelic.core.curves import YieldCurve
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.expenses.expense_engine import ExpenseEngine
from modelic.core.contingent_cashflows.survival_contingent_cashflow import SurvivalContingentCashflow
from modelic.products.product_factory import PRODUCT_FACTORY
from modelic.core.custom_types import ArrayLike


class PricingEngine:

    def __init__(self, mortality_table: MortalityTable, yield_curves: YieldCurve, expense_spec: pd.DataFrame,
                 expense_inflation_rate: float):
        self.mortality_table = mortality_table
        self.yield_curves = yield_curves
        self.expense_spec = expense_spec
        self.expense_inflation_rate = expense_inflation_rate
        self.expense_engine = ExpenseEngine(self.expense_spec, self.yield_curves, self.mortality_table,
                                            self.expense_inflation_rate)

    def price_policy_portfolio(self, policy_portfolio: PolicyPortfolio) -> np.ndarray:

        benefit_pvs = self._calculate_pv_of_benefits(policy_portfolio)

        prices = self.price_policy_group(policy_portfolio, benefit_pvs)

        return prices


    def _calculate_pv_of_benefits(self, policy_portfolio: PolicyPortfolio) -> np.ndarray:

        benefit_pvs = pd.Series(index=policy_portfolio.policy_id)

        for policy_type in np.unique(policy_portfolio.policy_type):

            filtered_policies = policy_portfolio.subset(policy_type)
            product_engine = PRODUCT_FACTORY[policy_type].from_policy_portfolio(filtered_policies, self.yield_curves, self.mortality_table)
            ben_pvs = product_engine.present_value(aggregate=False)
            benefit_pvs.loc[filtered_policies.policy_id] = ben_pvs

        return benefit_pvs.values


    def price_policy_group(self, policy_data: PolicyPortfolio, pv_bens: ArrayLike) -> np.ndarray:

        reg_prem_pols = policy_data.premium_type == 'Regular'
        prem_ann_fac = np.ones(policy_data.ages.size)

        if reg_prem_pols.any():

            prem_ann_fac_obj = SurvivalContingentCashflow(self.yield_curves, self.mortality_table,
                                                          policy_data.ages[reg_prem_pols], policy_data.terms[reg_prem_pols] - 1,
                                                          periodic_cf=1.0)

            prem_ann_fac[reg_prem_pols] += prem_ann_fac_obj.present_value(aggregate=False)

        pv_expenses = self.expense_engine.present_value(policy_data, group_by=['policy_id', 'Basis'], unstack=True)
        premium = (pv_bens + pv_expenses['PER_POLICY']) / (prem_ann_fac - pv_expenses['PCT_PREMIUM'])

        return premium.values