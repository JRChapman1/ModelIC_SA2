import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.contingent_cashflows.guaranteed_cashflow import GuaranteedCashflow
from modelic.core.asset_portfolio import AssetPortfolio
from _data import data_path


class TestGuaranteedCashflow(unittest.TestCase):

    disc_raw = pd.read_csv(data_path("curves", "boe_spot_annual.csv"))
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    bond_portfolio = AssetPortfolio.from_csv(data_path("asset_data", "bond_portfolio.csv"))
    bond_model = GuaranteedCashflow.from_asset_portfolio(bond_portfolio, discount_curve)


    def test_present_value(self):

        expected = np.array([136110.4485, 107900.9637, 60806.65089, 18990.16939, 195387.6766]) # 519195.909

        actual = self.bond_model.present_value(aggregate=False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_present_value_aggregated(self):

        expected = 519195.909

        actual = self.bond_model.present_value(aggregate=True)

        assert np.allclose(actual, expected)