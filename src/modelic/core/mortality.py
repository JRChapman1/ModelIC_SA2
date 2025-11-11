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
        return (1 - self.qx).cumprod()


    def npx(self, age: IntArrayLike, n: int = 1, full_path: bool = False) -> ArrayLike:
        age = np.asarray(age.copy(), dtype=int)
        n = np.asarray(n.copy(), dtype=int)

        age -= 1
        idx = self._resolve_idx(age)

        if full_path:
            idx_end = idx.reshape(1, -1) + np.arange(n).reshape(-1, 1) + 1
        else:
            idx_end = idx + n

        idx_end = idx_end.clip(0, self.max_age - self.min_age)

        return self.lx[idx_end] / self.lx[idx]


    def nqx(self, age: IntArrayLike, n: int = 1, full_path: bool = False) -> ArrayLike:
        age = np.asarray(age.copy(), dtype=int)
        match n:
            case 0:
                return np.zeros(age.size)
            case 1:
                return self.qx[self._resolve_idx(age)]
            case _:
                npx = self.npx(age, n-1, full_path)
                idx = self._resolve_idx(age)

                if full_path:
                    npx = np.concatenate([np.ones([1, age.size]), npx])
                    idx = (idx.reshape(1, -1) + np.arange(n).reshape(-1, 1) + 1).clip(0, self.max_age - self.min_age)
                else:
                    idx += n

                qx = self.qx[idx]

                return npx * qx


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