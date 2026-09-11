# modelic/core/utils

import numpy as np
import numpy_financial as npf


def calculate_irr(cfs: np.ndarray, pv: float) -> np.ndarray:
    return npf.irr(np.insert(cfs, 0, -pv))

def accumulate_cashflows(cfs: np.ndarray, forward_curve: np.ndarray) -> np.ndarray:

    results = [cfs[0]]
    for t in range(1, len(cfs)):
        results.append(cfs[t] + results[-1] * (1+forward_curve[t-1]))

    return np.array(results)

def match_length(array1: np.ndarray, array2: np.ndarray) -> [np.ndarray, np.ndarray]:
    max_len = max(len(array1), len(array2))
    array1 = np.pad(array1, (0, max_len - len(array1)), mode='constant')
    array2 = np.pad(array2, (0, max_len - len(array2)), mode='constant')
    return array1, array2
