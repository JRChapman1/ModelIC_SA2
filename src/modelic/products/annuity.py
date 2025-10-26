import numpy as np

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.custom_types import IntArrayLike, ArrayLike


class Annuity(BaseCashflowModel):

    def __init__(self,
                 age: IntArrayLike,
                 term: IntArrayLike,
                 amount: ArrayLike,
                 mortality: MortalityTable,
                 discount_curve: YieldCurve):

        super().__init__(np.arange(1, term + 1), discount_curve)

        self.age = age
        self.term = term
        self.amount = amount
        self.mortality = mortality
        self.discount_curve = discount_curve

    def project_cashflows(self, aggregate: bool = True) -> ArrayLike:
        surv = self.mortality.npx(self.age, self.term, True)
        cfs = self.amount * surv
        if aggregate:
            cfs = cfs.sum(axis=1)
        return cfs