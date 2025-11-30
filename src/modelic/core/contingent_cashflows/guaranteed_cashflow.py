# modelic/core/contingent_cashflows/guaranteed_cashflow.py

import numpy as np

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.curves import YieldCurve
from modelic.core.asset_portfolio import AssetPortfolio
from modelic.core.custom_types import ArrayLike, IntArrayLike


class GuaranteedCashflow(BaseCashflowModel):
    """ Projects and calculates present values for guaranteed cashflows"""

    def __init__(self, yield_curve: YieldCurve, notional: ArrayLike, coupon_rate: ArrayLike, maturity: IntArrayLike, spread: ArrayLike, *,
                 projection_steps: IntArrayLike = None, escalation: float = 0.0):

        if projection_steps is None:
            projection_steps = np.arange(1, maturity.max() + 1)

        super().__init__(projection_steps, yield_curve)

        self.notional = notional
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.spread = spread
        self.escalation = escalation


    @classmethod
    def from_asset_portfolio(cls, asset_portfolio: AssetPortfolio, yield_curve: YieldCurve, *,
                             projection_steps: IntArrayLike = None):

        return cls(yield_curve=yield_curve,
                   notional=asset_portfolio.notional,
                   coupon_rate=asset_portfolio.coupon_rate,
                   maturity=asset_portfolio.maturity,
                   spread=asset_portfolio.spread,
                   projection_steps=projection_steps)


    def project_cashflows(self, aggregate: bool = True) -> ArrayLike:

        times = self.times.reshape(-1, 1)
        maturities = self.maturity.reshape(1, -1)

        cfs = (times <= maturities) * self.notional * self.coupon_rate
        cfs += (times == maturities) * self.notional

        return cfs.sum(axis=1) if aggregate else cfs


    def present_value(self, spread: ArrayLike = None, aggregate: bool = True) -> ArrayLike:
        if spread is None:
            spread = self.spread
        return super().present_value(spread=spread, aggregate=aggregate)
