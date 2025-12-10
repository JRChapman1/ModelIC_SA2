# modelic/core/utils

import numpy as np
import numpy_financial as npf


def calculate_irr(cfs: np.ndarray, pv: float) -> np.ndarray:
    return npf.irr(np.insert(cfs, 0, -pv))

