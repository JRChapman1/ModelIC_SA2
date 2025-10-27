import numpy as np

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.curves import YieldCurve, IndexCurve
from modelic.core.custom_types import IntArrayLike, ArrayLike

# TODO: Implement indexation
class BondLike(BaseCashflowModel):

    def __init__(self,
                 coupon_rate: ArrayLike,
                 term: IntArrayLike,
                 notional: ArrayLike,
                 spreads: ArrayLike,
                 discount_curve: YieldCurve):

        coupon_rate = np.asarray(coupon_rate)
        term = np.asarray(term)
        notional = np.asarray(notional)

        super().__init__(np.arange(1, max(term) + 1), discount_curve)

        self.coupon_rate = coupon_rate
        self.term = term
        self.notional = notional
        self.spreads = spreads
        self.curve = discount_curve

    def project_cashflows(self, aggregate: bool = True) -> np.ndarray:

        cfs = np.zeros((self.term.max(), self.term.size))
        for j, (coupon, notional, dur) in enumerate(zip(self.coupon_rate, self.notional, self.term-1)):
            cfs[:dur, j] = coupon * notional
            cfs[dur, j] = (1 + coupon) * notional

        if aggregate:
            cfs = cfs.sum(axis=1)

        return cfs
