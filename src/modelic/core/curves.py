# modelic/core/curves.py

from abc import ABC, abstractmethod
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

    def fwd(self, t: ArrayLike) -> ArrayLike:

        # Treat f(0) as forward rate applying between times 0 and 1.
        t += 1

        df = np.insert(self.df(t), 0, 1)
        return df[:-1] / df[1:] - 1


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
class EIOPATable(ABC):
    rate_term_structures: pd.DataFrame
    name: str = None

    @abstractmethod
    def resolve_values(self, asset_terms: ArrayLike, asset_ratings: ArrayLike) -> np.ndarray:
        pass


    @classmethod
    def from_df(cls, df: pd.DataFrame, name: str = None) -> "EIOPATable":

        return cls(df, name)


    @classmethod
    def from_csv(cls, path: str, name: str = None) -> "EIOPATable":
        data = pd.read_csv(path, index_col=0)
        return cls.from_df(data, name)


class FSTable(EIOPATable):

    def resolve_values(self, asset_terms: ArrayLike, asset_ratings: ArrayLike) -> np.ndarray:
        term_idx = self.rate_term_structures.index.get_indexer(asset_terms)
        rating_idx = [self.rate_term_structures.columns.get_loc(r) for r in asset_ratings]
        return self.rate_term_structures.values[term_idx, rating_idx]


class PDTable(EIOPATable):

    def resolve_values(self, asset_terms: ArrayLike, asset_ratings: ArrayLike, *, proj_term=None) -> np.ndarray:

        term_idx = self.rate_term_structures.index.get_indexer(asset_terms)
        rating_idx = [self.rate_term_structures.columns.get_loc(r) for r in asset_ratings]


        if proj_term is None:
            proj_term = term_idx.max() + 1

        pds = self.rate_term_structures.values[:proj_term, rating_idx]
        pds[np.arange(0, proj_term)[:, None] > term_idx[None, :]] = 0.0

        return pds


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
