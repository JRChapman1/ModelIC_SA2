# modelic/SII/matching_adjustment

import numpy as np

from modelic.core.utils import calculate_irr
from modelic.core.curves import YieldCurve, SpreadTable


class MatchingAdjustment:

    def __init__(self, asset_cfs: np.ndarray, asset_mvs: np.ndarray, asset_terms: np.ndarray, asset_ratings: np.ndarray,
                 liability_cfs: np.ndarray, liability_pv: float, fs_table: SpreadTable):

        self.asset_cfs = asset_cfs
        self.asset_mvs = asset_mvs
        self.asset_terms = asset_terms
        self.asset_ratings = asset_ratings
        self.liability_cfs = liability_cfs
        self.liability_pv = liability_pv
        self.fs_table = fs_table

    def calculate_ma_spread(self):

        aggregate_asset_cfs = self.asset_cfs.sum(axis=1)
        aggregate_asset_mv = float(self.asset_mvs.sum())

        gry = calculate_irr(aggregate_asset_cfs, aggregate_asset_mv)
        fs = self._calculate_fs_deduction()
        rfr = calculate_irr(self.liability_cfs, self.liability_pv)

        return gry - fs - rfr


    def _calculate_fs_deduction(self) -> np.ndarray:

        fs_values = self.fs_table.resolve_spreads(self.asset_terms, self.asset_ratings)

        return (self.asset_mvs * fs_values).sum() / self.asset_mvs.sum()



