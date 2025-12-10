import numpy as np
import pandas as pd
import unittest

from modelic.SII.matching_adjustment import MatchingAdjustment
from modelic.core.curves import YieldCurve, SpreadTable
from modelic.core.asset_portfolio import AssetPortfolio
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.mortality import MortalityTable
from modelic.products.annuity import Annuity
from modelic.assets.bond import Bond
from _data import data_path


disc_raw = pd.read_csv(data_path("curves", "boe_spot_annual.csv"))
times = disc_raw['year'].to_numpy(int)
zeros = disc_raw['rate'].to_numpy(float)
discount_curve = YieldCurve(times, zeros, 'BoE')

bond_portfolio = AssetPortfolio.from_csv(data_path("asset_data", "bond_portfolio.csv"))
bond_model = Bond.from_asset_portfolio(bond_portfolio, discount_curve)

mort_raw = pd.read_csv(data_path("mortality", "AM92.csv"))
ages = mort_raw['x'].to_numpy(int)
qx = mort_raw['q_x'].to_numpy(float)
mortality = MortalityTable(ages, qx, 'AM92')

annuity_policies = PolicyPortfolio.from_csv(data_path("policy_data", "annuity_test_data.csv"))
annuity_model = Annuity.from_policy_portfolio(annuity_policies, discount_curve, mortality)

fs_table = SpreadTable.from_csv(data_path("curves", "fs_table_gbp_non_fin.csv"))


class TestSIIMatchingAdjustment(unittest.TestCase):

    ma_engine = MatchingAdjustment(bond_model.project_cashflows(aggregate=False),
                                   bond_model.present_value(aggregate=False),
                                   bond_portfolio.maturity,
                                   bond_portfolio.rating,
                                   annuity_model.project_cashflows(aggregate=True),
                                   annuity_model.present_value(aggregate=True),
                                   fs_table)


    def test_calculate_fs_deduction(self):

        expected = 0.00304800563589877

        actual = self.ma_engine._calculate_fs_deduction()

        assert np.allclose(actual, expected)

    def test_calculate_ma_spread(self):

        expected = 0.0031088312924114764

        actual = self.ma_engine.calculate_ma_spread()

        assert np.allclose(actual, expected)



