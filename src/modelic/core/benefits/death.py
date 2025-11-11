import numpy as np

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.mortality import MortalityTable
from modelic.core.custom_types import IntArrayLike, ArrayLike


class _DeathBenefit(BaseCashflowModel):

    def __init__(self,
                 age: IntArrayLike,
                 term: IntArrayLike,
                 mortality: MortalityTable):

        if term is None:
            term = mortality.max_age - mortality.min_age

        self.age = age
        self.term = term
        self.mortality = mortality

    def project_cashflows(self, aggregate: bool = False) -> ArrayLike:
        surv = self.mortality.nqx(self.age, self.term, False)
        if aggregate:
            surv = surv.sum(axis=1)
        return surv