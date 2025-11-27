# modelic/core/mortality.py

from dataclasses import dataclass
from functools import cached_property
import numpy as np

from modelic.core.custom_types import IntArrayLike, ArrayLike


@dataclass(frozen=True)
class MortalityTable:
    ages: np.ndarray
    qx: np.ndarray
    name: str


    def __post_init__(self):
        self._validate_inputs(self.qx, self.ages)


    @property
    def min_age(self) -> int:
        return int(self.ages[0])


    @property
    def max_age(self) -> int:
        return int(self.ages[-1])


    @cached_property
    def lx(self) -> np.ndarray:
        return np.concatenate(([1.0], (1 - self.qx).cumprod()))

    @cached_property
    def survival_table(self) -> np.ndarray:
        return self._hankel(self.lx[:-1]) / self.lx[:-1]

    @cached_property
    def death_table(self) -> np.ndarray:
        return self._hankel(self.qx) * self.survival_table

    @staticmethod
    def _hankel(arr_in):
        n = len(arr_in)
        out = np.zeros((n, n), dtype=float)
        for i in range(n):
            out[:(n - i), i] = arr_in[i:]
        return out

    def npx(self, age, term=1, *, proj_horizon = None, incl_t0 = False, full_path = False):

        age = np.asarray(age, dtype=int)

        if type(term) is int:
            term = np.repeat(term, age.size)
        else:
            term = np.asarray(term, dtype=int).copy()

        if full_path:
            term += 1
        surv = self._filter_table(self.survival_table, self._resolve_idx(age), term, num_rows=proj_horizon, fill_val=0.0, full_path=full_path)
        return surv[1:] if not incl_t0 and full_path else surv

    def nqx(self, age, term=1, *, proj_horizon = None, full_path = False):

        age = np.asarray(age, dtype=int)

        if type(term) is int:
            term = np.repeat(term, age.size)
        else:
            term = np.asarray(term, dtype=int).copy()

        if not full_path:
            term -= 1
        return self._filter_table(self.death_table, self._resolve_idx(age), term, num_rows=proj_horizon, fill_val=0.0, full_path=full_path)

    def _filter_table(self, array, col_filter, row_filter, *, num_rows = None, fill_val = np.nan, full_path = True):
        col_filter = np.asarray(col_filter, dtype=int)
        row_filter = np.asarray(row_filter, dtype=int)
        array = array.copy()
        if full_path:
            if num_rows is None:
                num_rows = row_filter.max()
            mask = (np.arange(num_rows)[:, None] >= row_filter)
            array = array[:num_rows, col_filter].reshape(mask.shape)
            array[mask] = fill_val
            return array
        else:
            return array[row_filter, col_filter]


    def _resolve_idx(self, age: IntArrayLike) -> IntArrayLike:
        return age.clip(self.min_age, self.max_age) - self.min_age



    @staticmethod
    def _validate_inputs(qx: ArrayLike, ages: IntArrayLike) -> None:

        if qx.ndim != 1:
            raise ValueError("qx must be 1-dimensional")
        if ages.ndim != 1:
            raise ValueError("ages must be 1-dimensional")
        if len(ages) != len(qx):
            raise ValueError("ages and qx must have same length")
        if qx[-1] != 1:
            raise ValueError("terminal qx value must be 1")
        if (qx < 0).any() or (qx > 1).any():
            raise ValueError("qx must be between 0 and 1")
        if not np.all(np.diff(ages) > 0):
            raise ValueError("ages must be increasing")