import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import SpreadTable
from _data import data_path
from tests.test_sii_matching_adjustment import fs_table


class TestSpreadTable(unittest.TestCase):

    fs_table = SpreadTable.from_csv(data_path("curves", "fs_table_gbp_non_fin.csv"))

    def test_resolve_spreads(self):

        expected = np.array([0.000900000, 0.014700000, 0.005900000])
        actual = fs_table.resolve_spreads(np.array([20, 30, 10]), np.array(['AAA', 'BB', 'BBB']))

        actual.shape == expected.shape
        assert np.allclose(actual, expected)


