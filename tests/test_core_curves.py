import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import FSTable, PDTable, YieldCurve
from _data import data_path


class TestFSTable(unittest.TestCase):

    fs_table = FSTable.from_csv(data_path("curves", "fs_table_gbp_non_fin.csv"))

    def test_resolve_spreads(self):

        expected = np.array([0.000900000, 0.014700000, 0.005900000])
        actual = self.fs_table.resolve_values(np.array([20, 30, 10]), np.array(['AAA', 'BB', 'BBB']))

        actual.shape == expected.shape
        assert np.allclose(actual, expected)

class TestPDTable(unittest.TestCase):

    pd_table = PDTable.from_csv(data_path("curves", "pd_table_gbp_non_fin.csv"))

    def test_resolve_spreads(self):

        expected = np.array([[0.              , 0.0063            , 0.0014            ],
                             [0.000003        , 0.01579351        , 0.00326536        ],
                             [0.000014672908  , 0.028333314399    , 0.005614326864    ],
                             [0.00004110474476, 0.0433851962119193, 0.0084770155238518],
                             [0.              , 0.0603403344297312, 0.0118749351603839],
                             [0.              , 0.                , 0.0158154573042729]])

        actual = self.pd_table.resolve_values(np.array([4, 5, 6]), np.array(['AAA', 'BB', 'BBB']))

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


class TestYieldCurve(unittest.TestCase):

    disc_raw = pd.read_csv(data_path("curves", "boe_spot_annual.csv"))
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    yield_curve = YieldCurve(times, zeros, 'BoE')

    def test_fwd(self):

        expected = np.array([0.004991184, 0.00918796024016966, 0.00921431257238936, 0.00945861076479004,
                             0.0100719450129438, 0.0109797524052513, 0.0120694136839334, 0.0132012167109825,
                             0.0142308261969024, 0.0150510534638155])

        actual = self.yield_curve.fwd(np.arange(0, 10))

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

