from dataclasses import dataclass
import numpy as np

from modelic.core.custom_types import IntArrayLike, ArrayLike


# TODO: Update and implement _check_bounds()

@dataclass(frozen=True)
class MortalityTable:
    ages: np.ndarray
    qx: np.ndarray
    name: str

    def __post_init__(self):
        if self.qx.ndim != 1:
            raise ValueError("qx must be 1-dimensional")
        if self.ages.ndim != 1:
            raise ValueError("ages must be 1-dimensional")
        if (self.qx < 0).any() or (self.qx > 1).any():
            raise ValueError("qx must be between 0 and 1")


    # --- Properties ---

    @property
    def max_age(self) -> int:
        return np.max(self.ages)

    @property
    def min_age(self) -> int:
        return np.min(self.ages)


    # --- Public methods ---

    def q(self, age: IntArrayLike) -> ArrayLike:
        a = np.asarray(age, dtype=int)
        idx = a - self.min_age
        return self.qx[idx]

    def p(self, age: IntArrayLike) -> ArrayLike:
        return 1.0 - self.q(age)

    def tp(self, age: IntArrayLike, t: int, full_path: bool = False) -> ArrayLike:

        # Ensure that age and horizon are numpy arrays
        age = np.asarray(age, dtype=int)

        # Construct matrix of ages at for each policy at each positive projection step
        age_matrix = np.atleast_2d(age) + np.atleast_2d(np.arange(0, t)).T

        # Get one-year conditional survival probabilities for each age in age_matrix and construct the unconditional
        # survival probabilities to each age as the cumulative product of these.
        px = self.p(age_matrix)
        px_cumulative = np.cumprod(px, axis=0)

        # Return only the terminal values, unless the full path is requested
        if not full_path:
            px_cumulative = px_cumulative[-1]

        return px_cumulative

    def tq(self, age: IntArrayLike, t: int = 1, full_path: bool = 1) -> ArrayLike:

        # Get the number of ages
        if type(age) is int:
            num_ages = 1
        else:
            num_ages = len(age)

        # Construct matrix of ages at for each policy at each positive projection step
        age_matrix = self._construct_age_matrix(age, t)

        px_cumulative = np.concat([np.ones((1, num_ages)), self.tp(age, t-1, True)], axis=0)
        qx = self.q(age_matrix)
        qx_cumulative = qx * px_cumulative

        if not full_path:
            qx_cumulative = qx_cumulative[-1]

        return qx_cumulative


    # --- Private methods ---

    def _check_bounds(self, a: np.ndarray) -> None:

        a = np.atleast_1d(a)

        if a.min() < self.min_age or a.max() > self.max_age:
            raise ValueError(f"Age outside of bounds [{self.min_age}, {self.max_age}]")

    @staticmethod
    def _construct_age_matrix(age: np.ndarray, times: int) -> np.ndarray:

        # Ensure that age and horizon are numpy arrays
        age = np.asarray(age, dtype=int)
        horizon = np.arange(0, times)

        # Construct matrix of ages at for each policy at each positive projection step
        age_matrix = np.atleast_2d(age) + np.atleast_2d(horizon).T

        return age_matrix
