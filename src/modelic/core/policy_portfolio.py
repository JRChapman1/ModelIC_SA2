#modelic/core/policy_portfolio.py

from dataclasses import dataclass
import numpy as np
import pandas as pd

from modelic.core.custom_types import IntArrayLike, ArrayLike
from modelic.core.udf_globals import policy_data_csv_columns, wol_years

@dataclass(frozen=True)
class PolicyPortfolio:
    ages: IntArrayLike
    _terms: IntArrayLike
    death_contingent_benefits: ArrayLike = None
    terminal_survival_contingent_benefits: ArrayLike = None
    periodic_survival_contingent_benefits: ArrayLike = None
    policy_type: ArrayLike = None
    premium_type: ArrayLike = None
    _policy_id: IntArrayLike = None
    premiums: ArrayLike = None

    @classmethod
    def from_csv(cls, path: str):
        csv_data = pd.read_csv(path)
        return cls.from_df(csv_data)


    @classmethod
    def from_df(cls, df: pd.DataFrame):

        if 'premium' not in df.columns:
            df['premium'] = np.ones_like(df['age'])

        return cls(ages=df['age'].to_numpy(),
                   _terms=df['term'].to_numpy(),
                   death_contingent_benefits=df.get('death_contingent_benefit', None),
                   terminal_survival_contingent_benefits=df.get('terminal_survival_contingent_benefit', None),
                   periodic_survival_contingent_benefits=df.get('periodic_survival_contingent_benefit', None),
                   policy_type=df.get('policy_type', None),
                   premium_type=df.get('premium_type', None),
                   _policy_id=df.get('policy_id', None),
                   premiums=df['premium'].to_numpy())


    @property
    def policy_id(self):
        if self._policy_id is None:
            return np.arange(1, self.count+1)
        else:
            return self._policy_id


    @property
    def terms(self):
        policy_terms = self._terms.astype(float)
        policy_terms[np.isnan(policy_terms)] = wol_years
        return policy_terms.astype(int)


    @property
    def count(self):
        return len(self.ages)


    @property
    def data(self):
        return pd.DataFrame({k: getattr(self, k) for k in policy_data_csv_columns})


    def get(self, attr: str, product_type: str):
        if attr == 'count':
            return int(sum(self.policy_type == product_type))
        return getattr(self, attr)[self.policy_type == product_type]


    def is_type(self, product_type: str):
        return np.asarray(self.policy_type == product_type, dtype=bool)
