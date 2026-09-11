# modelic/SII/matching_adjustment

import numpy as np

from modelic.core.utils import calculate_irr, accumulate_cashflows, match_length
from modelic.core.curves import YieldCurve, EIOPATable
from modelic.core.udf_globals import (matching_test_1_threshold, matching_test_3_threshold_lower,
                                      matching_test_3_threshold_upper)


class MatchingAdjustment:

    def __init__(self, asset_cfs: np.ndarray, asset_mvs: np.ndarray, asset_terms: np.ndarray, asset_ratings: np.ndarray,
                 liability_cfs: np.ndarray, liability_pv: float, risk_free_curve: YieldCurve, fs_table: EIOPATable,
                 pd_table: EIOPATable, lgd: float, comp_a_cash: float = 0.0):

        self.asset_cfs = asset_cfs
        self.asset_mvs = asset_mvs
        self.asset_terms = asset_terms
        self.asset_ratings = asset_ratings
        self.liability_cfs = liability_cfs
        self.liability_pv = liability_pv
        self.risk_free_curve = risk_free_curve
        self.fs_table = fs_table
        self.pd_table = pd_table
        self.lgd = lgd
        self.comp_a_cash = comp_a_cash

    def calculate_ma_spread(self):

        aggregate_asset_cfs = self.asset_cfs.sum(axis=1)
        aggregate_asset_mv = float(self.asset_mvs.sum())

        gry = calculate_irr(aggregate_asset_cfs, aggregate_asset_mv)
        fs = self._calculate_fs_deduction()
        rfr = calculate_irr(self.liability_cfs, self.liability_pv)

        return gry - fs - rfr


    # Accumulated shortfall test
    def calculate_test_1_statistic(self) -> float:

        pd_adjusted_asset_cfs, liability_cfs = match_length(self._project_pd_adjusted_asset_cfs(aggregate=True),
                                                            self.liability_cfs)

        surplus = np.insert(pd_adjusted_asset_cfs - liability_cfs, 0, self.comp_a_cash)
        proj_times = np.arange(0, len(surplus))
        accumulated_surplus = accumulate_cashflows(surplus, self.risk_free_curve.fwd(proj_times))
        min_surplus = min(accumulated_surplus)

        return -min_surplus / self.liability_pv


    def calculate_test_2_statistic(self) -> float:
        raise NotImplementedError('Internal model needed to run test 2 (or standard formula derivation of SII shock '
                                  'magnitudes)')


    def calculate_test_3_statistic(self) -> float:

        pd_adjusted_asset_cfs = self._project_pd_adjusted_asset_cfs(aggregate=True)
        discount_curve = self.risk_free_curve.df(np.arange(1, len(pd_adjusted_asset_cfs) + 1))
        asset_pv = (pd_adjusted_asset_cfs * discount_curve).sum() + self.comp_a_cash

        return self.liability_pv / asset_pv


    def test_1_passes(self) -> bool:
        return self.calculate_test_1_statistic() <= matching_test_1_threshold


    def test_2_passes(self) -> bool:
        raise NotImplementedError('Internal model needed to run test 2 (or standard formula derivation of SII shock '
                                  'magnitudes)')


    def test_3_passes(self) -> bool:
        t3_stat = self.calculate_test_3_statistic()
        return (t3_stat >= matching_test_3_threshold_lower) and (t3_stat <= matching_test_3_threshold_upper)


    def _project_pd_adjusted_asset_cfs(self, aggregate: bool = True):
        pds = self.pd_table.resolve_values(self.asset_terms, self.asset_ratings)
        pd_adj_cfs = self.asset_cfs * (1 - pds * self.lgd)
        return pd_adj_cfs.sum(axis=1) if aggregate else pd_adj_cfs


    def _calculate_fs_deduction(self) -> np.ndarray:
        fs_values = self.fs_table.resolve_values(self.asset_terms, self.asset_ratings)
        return (self.asset_mvs * fs_values).sum() / self.asset_mvs.sum()




