import numpy as np

from modelic.core.curves import YieldCurve
from modelic.core.custom_types import ArrayLike


# --- Curve construction ---

def build_yield_curve(times: np.ndarray, zero_rates: np.ndarray, name: str = None) -> YieldCurve:
    pass

def build_index_curve(times: np.ndarray, index_levels: np.ndarray, name: str = None) -> YieldCurve:
    pass


# --- Curve adjustments ---

def apply_spread(yc: YieldCurve, spread: ArrayLike) -> YieldCurve:
    pass

def twist(yc: YieldCurve, shift_short: ArrayLike, shift_long: ArrayLike, pivot: float) -> ArrayLike:
    pass