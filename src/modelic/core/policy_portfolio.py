from dataclasses import dataclass
import numpy as np
import pandas as pd

from modelic.core.custom_types import IntArrayLike, ArrayLike
from modelic.core.udf_globals import policy_data_csv_columns

@dataclass
class PolicyPortfolio:
    ages: IntArrayLike
    terms: IntArrayLike
    death_contingent_benefits: ArrayLike = None
    terminal_survival_contingent_benefits: ArrayLike = None
    periodic_survival_contingent_benefits: ArrayLike = None
    annual_premium: ArrayLike = None


    @classmethod
    def from_csv(cls, path: str):
        csv_data = pd.read_csv(path)
        columns_given = csv_data.columns
        return cls(*[csv_data[col].values if np.isin(col, columns_given) else None for col in policy_data_csv_columns])

    @property
    def count(self):
        return len(self.ages)

    @property
    def data(self):
        return pd.DataFrame({k: v for k, v in self.__dict__.items() if v is not None}).fillna('')
