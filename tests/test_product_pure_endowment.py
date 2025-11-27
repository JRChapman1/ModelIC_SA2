import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.products.pure_endowment import PureEndowment
from _data import data_path


class TestProductPureEndowment(unittest.TestCase):

    mort_raw = pd.read_csv(data_path("mortality", "AM92.csv"))
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    disc_raw = pd.read_csv(data_path("curves", "boe_spot_annual.csv"))
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    policies = PolicyPortfolio.from_csv(data_path("policy_data", "pure_endowment_test_data.csv"))

    pure_endowments = PureEndowment.from_policy_portfolio(policies, discount_curve, mortality)


    def test_present_value(self):

        expected = np.array([4.656797899, 9.710725896, 11.56938066])
        actual = self.pure_endowments.present_value(aggregate=False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)

    def test_cashflows(self):

        expected = np.array([[ 0.        ,  0.        ,  0.        ],
                             [ 0.        ,  0.        ,  0.        ],
                             [ 0.        ,  9.93961149,  0.        ],
                             [ 0.        ,  0.        ,  0.        ],
                             [ 0.        ,  0.        , 12.07448594],
                             [ 0.        ,  0.        ,  0.        ],
                             [ 4.9727736 ,  0.        ,  0.        ]])

        actual = self.pure_endowments.project_cashflows(aggregate=False)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)


    def test_present_value_aggregated(self):

        expected = 25.936904455
        actual = self.pure_endowments.present_value(aggregate=True)

        assert np.allclose(actual, expected)


    def test_cashflows_aggregated(self):

        expected = np.array([0., 0., 9.93961149, 0., 12.07448594, 0., 4.9727736])

        actual = self.pure_endowments.project_cashflows(aggregate=True)

        assert actual.shape == expected.shape
        assert np.allclose(actual, expected)
