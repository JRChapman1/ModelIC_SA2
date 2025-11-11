import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.assets.bonds import BondLike
from _data import data_path


class TestMortalityTable(unittest.TestCase):

    disc_raw = pd.read_csv(data_path("curves", "boe_spot_annual.csv"))
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    coupon = [0.03, 0.05, 0.06]
    term = [10, 12, 7]
    notional = [100_000, 135_000, 97_000]
    spread = [0.01, 0.02, 0.03]

    bond_portfolio = BondLike(coupon, term, notional, spread, discount_curve)

    def test_project_cashflows(self):

        expected = np.array([[3000, 6750, 5820], [3000, 6750, 5820], [3000, 6750, 5820], [3000, 6750, 5820],
                             [3000, 6750, 5820], [3000, 6750, 5820], [3000, 6750, 102820], [3000, 6750, 0],
                             [3000, 6750, 0], [103000, 6750, 0], [0, 6750, 0], [0, 141750, 0]])

        actual = self.bond_portfolio.project_cashflows(False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_project_cashflows_aggregated(self):

        expected = np.array([[15570], [15570], [15570], [15570], [15570], [15570], [112570], [9750], [9750], [109750], [6750], [141750]])

        actual = self.bond_portfolio.project_cashflows(True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_present_value(self):

        expected = np.array([108396.5898, 160097.7912, 109132.8769])

        actual = self.bond_portfolio.present_value(self.spread, False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_present_value_aggregated(self):

        expected = 377627.25789999997

        actual = self.bond_portfolio.present_value(self.spread, True)

        assert np.allclose(actual, expected)

