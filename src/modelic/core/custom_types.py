# modelic/core/custom_types.py

import numpy as np
from typing import Union, Sequence


ArrayLike = float | np.ndarray | list | tuple | None
IntArrayLike = int | Sequence[int] | np.ndarray | None

