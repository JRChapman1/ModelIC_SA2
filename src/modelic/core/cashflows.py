from abc import ABC, abstractmethod
import numpy as np

from modelic.core.compounding import zero_to_df
from modelic.core.custom_types import ArrayLike, IntArrayLike
from modelic.core.curves import YieldCurve

class BaseCashflowModel(ABC):
    """Abstract base class for any models producing or valuing time-indexed cashflows."""

    def __init__(self, times: IntArrayLike, curve: YieldCurve):
        self.times = times
        self.curve = curve

    @abstractmethod
    def project_cashflows(self, aggregate: bool = True):
        pass

    def discount_factors(self, spread: ArrayLike = 0) -> np.ndarray:
        zeros = self.curve.zero(self.times)[:, None] + np.atleast_2d(spread)
        return zero_to_df(self.times, zeros)

    def present_value(self, spread: ArrayLike = 0, aggregate: bool = True):

        cf = self.project_cashflows(False)
        df = self.discount_factors(spread)
        pv = (df * cf).sum(axis=0)

        if aggregate:
            pv = pv.sum().sum()

        return pv

    def expected_value(self, spread: ArrayLike = 0, probabilities = None):

        cf = self.project_cashflows()
        if probabilities is not None:
            cf *= probabilities

        df = self.discount_factors()[:, None]

        return cf @ df
