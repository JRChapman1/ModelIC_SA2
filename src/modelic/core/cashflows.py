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
    def project_cashflows(self):
        pass

    def discount_factors(self, spread: ArrayLike = 0) -> np.ndarray:
        zeros = self.curve.zero(self.times)[:, None] + np.atleast_2d(spread)
        return zero_to_df(self.times, zeros)

    def present_value(self, spread: ArrayLike = 0, aggregate: bool = True):

        cf = self.project_cashflows(aggregate)
        df = self.discount_factors(spread)
        pv = (df * cf).sum(axis=0)

        return float(pv.sum()) if aggregate else pv


class CompositeProduct(BaseCashflowModel):
    """Aggregates multiple BaseCashflowModel components into one"""

    def __init__(self, components: list[BaseCashflowModel], yield_curve: YieldCurve):

        # Ensure at least one component and store as property
        assert components
        self.components = components

        times = np.unique(np.concatenate([c.times for c in components]))
        super().__init__(times, yield_curve)

    def project_cashflows(self, aggregate: bool = True) -> ArrayLike:

        cfs = sum([c.project_cashflows(aggregate) for c in self.components])
        return cfs

    def present_value(self, spread: ArrayLike = 0, aggregate: bool = True):

        pvs = sum(c.present_value(spread, aggregate) for c in self.components)

        return float(sum(pvs)) if aggregate and (type(pvs) is not float) else pvs
