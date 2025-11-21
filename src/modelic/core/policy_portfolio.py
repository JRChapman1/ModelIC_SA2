#modelic/core/policy_portfolio.py

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
    policy_type: ArrayLike = None


    @classmethod
    def from_csv(cls, path: str):
        csv_data = pd.read_csv(path)
        return cls.from_df(csv_data)


    @classmethod
    def from_df(cls, df: pd.DataFrame):
        columns_given = df.columns
        return cls(*[df[col].values if np.isin(col, columns_given) else None for col in policy_data_csv_columns])


    @property
    def count(self):
        return len(self.ages)

    @property
    def data(self):
        return pd.DataFrame({k: v for k, v in self.__dict__.items() if v is not None}).fillna('')

    def get(self, attr: str, product_type: str):
        if attr == 'count':
            return int(sum(self.policy_type == product_type))
        return getattr(self, attr)[self.policy_type == product_type]
