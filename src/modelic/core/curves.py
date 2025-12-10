# modelic/core/curves.py

from dataclasses import dataclass
import numpy as np
import pandas as pd

from modelic.core.compounding import zero_to_df
from modelic.core.custom_types import ArrayLike


@dataclass(frozen=True)
class YieldCurve:
    times: np.ndarray
    zero_rates: np.ndarray
    name: str


    # --- Properties ---

    @property
    def min_time(self):
        return self.times[0]

    @property
    def max_time(self):
        return self.times[-1]

    # --- Core queries ---

    def zero(self, t: ArrayLike) -> ArrayLike:
        idx = self._resolve_idx(t)
        return self.zero_rates[idx]

    def df(self, t: ArrayLike) -> ArrayLike:
        df = zero_to_df(self.times, self.zero_rates)
        idx = self._resolve_idx(t)
        return df[idx]

    def fwd(self, t: ArrayLike, p: int) -> ArrayLike:
        pass


    # --- Transformations (return NEW YieldCurve objects) ---

    def with_spread(self, bp: ArrayLike) -> "YieldCurve":
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

    def _resolve_idx(self, times: np.ndarray) -> np.ndarray:
        return (times - self.min_time).astype(int)


@dataclass(frozen=True)
class SpreadTable:
    spread_term_structures: pd.DataFrame
    name: str = None


    def resolve_spreads(self, asset_terms: ArrayLike, asset_ratings: ArrayLike) -> np.ndarray:
        term_idx = self.spread_term_structures.index.get_indexer(asset_terms)
        rating_idx = [self.spread_term_structures.columns.get_loc(r) for r in asset_ratings]
        return self.spread_term_structures.values[term_idx, rating_idx]


    @classmethod
    def from_df(cls, df: pd.DataFrame, name: str = None) -> "SpreadTable":

        return cls(df, name)


    @classmethod
    def from_csv(cls, path: str, name: str = None) -> "SpreadTable":
        data = pd.read_csv(path, index_col=0)
        return cls.from_df(data, name)



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
