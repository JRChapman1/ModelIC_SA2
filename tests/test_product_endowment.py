import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.products.endowment import Endowment
from _data import data_path


class TestPureEndowment(unittest.TestCase):

    mort_raw = pd.read_csv(data_path("mortality", "AM92.csv"))
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    disc_raw = pd.read_csv(data_path("curves", "boe_spot_annual.csv"))
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    policies = PolicyPortfolio.from_csv(data_path("policy_data", "endowment_test_data.csv"))

    endowments = Endowment(policies, discount_curve, mortality)


    def test_project_cashflows(self):

        expected = np.array([[0.00329999999999997, 0.0180199999999997, 0.512159999999999],
                              [0.00344272630000025, 0.0200438158400001, 0.54924850224],
                              [0.00361511826615853, 2.01024699602296, 0.586062018114817],
                              [0.00381707624890697, 0, 0.621934118102913],
                              [0.00405347578937712, 0, 3.07100661130299],
                              [0.00433414129495432, 0, 0],
                              [0.999218579541711, 0, 0]])

        actual = self.endowments.project_cashflows(aggregate=False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_project_cashflows_aggregated(self):

        expected = np.array([[0.533479999999999],
[0.572735044380001],
[2.59992413240394],
[0.62575119435182],
[3.07506008709236],
[0.00433414129495432],
[0.999218579541711]])

        actual = self.endowments.project_cashflows(aggregate=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_present_value(self):

        expected = np.array([0.95762289, 2.001648991, 5.16818587])
        actual = self.endowments.present_value(aggregate=False)

        assert expected.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_present_value_aggregated(self):

        expected = 8.127457751
        actual = self.endowments.present_value(aggregate=True)

        assert np.allclose(actual, expected)