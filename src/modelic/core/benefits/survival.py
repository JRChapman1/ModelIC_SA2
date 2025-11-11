import numpy as np

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.mortality import MortalityTable
from modelic.core.custom_types import IntArrayLike, ArrayLike
from modelic.core.curves import YieldCurve

class _SurvivalBenefit(BaseCashflowModel):

    def __init__(self,
                 age: IntArrayLike,
                 term: IntArrayLike,
                 mortality: MortalityTable,
                 spot_curve: YieldCurve):

        if term is None:
            term = mortality.max_age - mortality.min_age

        self.age = age
        self.term = term
        self.mortality = mortality

        times = np.arange(1, term + 1)
        super().__init__(times, spot_curve)

    def project_cashflows(self, full_path: bool = False, aggregate: bool = False) -> ArrayLike:
        surv = self.mortality.npx(self.age, self.term, full_path)
        return surv
