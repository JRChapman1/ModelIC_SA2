# modelic/assets/bond.py

from modelic.core.cashflows import CompositeProduct
from modelic.core.contingent_cashflows.guaranteed_cashflow import GuaranteedCashflow
from modelic.core.curves import YieldCurve
from modelic.core.custom_types import ArrayLike, IntArrayLike
from modelic.core.asset_portfolio import AssetPortfolio

class Bond(CompositeProduct):

    """ Projects cashflows and calculates present values for bonds and bond-like assets """

    def __init__(self, yield_curve: YieldCurve, notional: ArrayLike, coupon_rate: ArrayLike, maturity: IntArrayLike,
                 spread: ArrayLike, *, projection_steps: IntArrayLike = None, escalation: float = 0.0):

        components = [GuaranteedCashflow(yield_curve, notional, coupon_rate, maturity, spread,
                                         projection_steps=projection_steps, escalation=escalation)]

        super().__init__(components, yield_curve)

    @classmethod
    def from_asset_portfolio(cls, asset_portfolio: AssetPortfolio, yield_curve: YieldCurve, *,
                             projection_steps: IntArrayLike = None):

        return cls(yield_curve=yield_curve,
                   notional=asset_portfolio.notional,
                   coupon_rate=asset_portfolio.coupon_rate,
                   maturity=asset_portfolio.maturity,
                   spread=asset_portfolio.spread,
                   projection_steps=projection_steps)
