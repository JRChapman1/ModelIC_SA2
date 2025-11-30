# modelic/core/asset_portfolio.py

import pandas as pd
from dataclasses import dataclass

from modelic.core.custom_types import ArrayLike, IntArrayLike


@dataclass(frozen=True)
class AssetPortfolio:
    isin: ArrayLike
    notional: ArrayLike
    coupon_rate: ArrayLike
    rating: ArrayLike
    spread: ArrayLike
    maturity: IntArrayLike

    @classmethod
    def from_csv(cls, path: str) -> "AssetPortfolio":
        df = pd.read_csv(path)
        return cls.from_df(df)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> "AssetPortfolio":

        return cls(isin=df['ISIN'].to_numpy(),
                   notional=df['Notional'].to_numpy(),
                   coupon_rate=df['Coupon Rate'].to_numpy(),
                   rating=df['Rating'].to_numpy(),
                   spread=df['Spread'].to_numpy(),
                   maturity=df['Maturity'].to_numpy())