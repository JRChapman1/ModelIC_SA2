import numpy as np
from typing import Union, Sequence


ArrayLike = Union[float, np.ndarray, list, tuple]
IntArrayLike = int | Sequence[int] | np.ndarray