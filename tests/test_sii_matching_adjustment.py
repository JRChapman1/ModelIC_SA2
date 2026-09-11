import numpy as np
import pandas as pd
import unittest

from modelic.SII.matching_adjustment import MatchingAdjustment
from modelic.core.curves import YieldCurve, FSTable, PDTable
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

fs_table = FSTable.from_csv(data_path("curves", "fs_table_gbp_non_fin.csv"))
pd_table = PDTable.from_csv(data_path("curves", "pd_table_gbp_non_fin.csv"))


class TestSIIMatchingAdjustment(unittest.TestCase):

    ma_engine = MatchingAdjustment(bond_model.project_cashflows(aggregate=False),
                                   bond_model.present_value(aggregate=False),
                                   bond_portfolio.maturity,
                                   bond_portfolio.rating,
                                   annuity_model.project_cashflows(aggregate=True),
                                   annuity_model.present_value(aggregate=True),
                                   discount_curve,
                                   fs_table,
                                   pd_table,
                                   0.7)


    def test_calculate_fs_deduction(self):

        expected = 0.00304800563589877

        actual = self.ma_engine._calculate_fs_deduction()

        assert np.allclose(actual, expected)

    def test_calculate_ma_spread(self):

        expected = 0.0031088312924114764

        actual = self.ma_engine.calculate_ma_spread()

        assert np.allclose(actual, expected)


    def test_project_pd_adjusted_asset_cfs(self):

        expected = np.array([13250.9961      , 13241.885459118 , 13229.7088867983, 13214.8232189946, 29479.5915773123,
                             12054.0382569626, 12049.2625127475, 12043.4777968079, 12036.5862015471, 65551.7148490474,
                             9881.95982800164, 9875.39382845576, 9867.79718437239, 9859.10453907495, 106119.972977338,
                             6857.96266709813, 6850.24275078487, 6841.67155922668, 6832.22118053714, 124910.560833326,
                             3275.6363554669 , 3271.68551861177, 3267.34143707904, 3262.59258637482, 3257.42917505814,
                             3251.84317040828, 3245.82830298082, 3239.38005204327, 3232.49561398596, 164483.866647881])

        actual = self.ma_engine._project_pd_adjusted_asset_cfs()

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_calculate_test_1_statistic(self):

        expected = 0.03

        actual = self.ma_engine.calculate_test_1_statistic()

        assert np.allclose(actual, expected)


    def test_calculate_test_3_statistic(self):

        expected = 1.0

        actual = self.ma_engine.calculate_test_3_statistic()

        assert np.allclose(actual, expected)


