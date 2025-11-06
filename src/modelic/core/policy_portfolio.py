from dataclasses import dataclass
import numpy as np
import pandas as pd

from modelic.core.custom_types import IntArrayLike, ArrayLike
from modelic.core.udf_globals import policy_data_csv_columns

@dataclass
class PolicyPortfolio:
    ages: IntArrayLike
    terms: IntArrayLike
    death_contingent_benefits: ArrayLike
    terminal_survival_contingent_benefits: ArrayLike
    periodic_survival_contingent_benefits: ArrayLike


    @classmethod
    def from_csv(cls, path: str):
        csv_data = pd.read_csv(path)
        columns_given = csv_data.columns
        return cls(*[csv_data[col].values if np.isin(col, columns_given) else None for col in policy_data_csv_columns])
