from dataclasses import dataclass
import numpy as np
import yaml

from modelic.core.custom_types import ArrayLike


@dataclass(frozen=True)
class YieldCurve:
    times: np.ndarray
    zero_rates: np.ndarray
    name: str

    # --- Core queries ---

    def zero(self, t: ArrayLike) -> ArrayLike:
        pass

    def df(self, t: ArrayLike) -> ArrayLike:
        pass

    def fwd(self, t: ArrayLike, p: int) -> ArrayLike:
        pass


    # --- Transformations (return NEW YieldCurve objects) ---

    def with_spread(self, bp: float) -> "YieldCurve":
        pass

    def shifted(self, bp: float) -> "YieldCurve":   # Alias for with_spread
        pass

    def scaled(self, factor: float) -> "YieldCurve":
        pass


    # --- Utilities ---

    def to_json(self) -> dict:
        pass

    @classmethod
    def from_json(cls, data: dict) -> "YieldCurve":
        pass

    def validate(self) -> None:
        pass

    def __print__(self):
        return('foo') # TODO: This should override print(cls) functionality

@dataclass(frozen=True)
class IndexCurve:
    times: np.ndarray
    index_levels: np.ndarray
    name: str

    # --- Core queries ---
    def level(self, t: ArrayLike) -> ArrayLike:
        pass

    def ratio(self, t: ArrayLike, shift: int) -> ArrayLike:
        pass


    # --- Real discount rate helpers ---

    def real_df(self, nominal_curve: YieldCurve, t: ArrayLike) -> ArrayLike:
        pass


    # --- Transformations and utilities ---

    def with_wedge(self, wedge_bps: float) -> "IndexCurve":
        pass

    def to_json(self) -> dict:
        pass

    @classmethod
    def from_json(cls, data: dict) -> "IndexCurve":
        pass

    def validate(self) -> None:
        pass
