import unittest
import numpy as np
import pandas as pd

from modelic.core.cashflows import BaseCashflowModel
from modelic.core.curves import YieldCurve
from _data import data_path


class BaseCashflowModelImplementation(BaseCashflowModel):
    def project_cashflows(self, aggregate: bool = True):
        return None

class TestBaseCashflowModelImplementation(unittest.TestCase):

    base_spot = pd.read_csv(data_path("curves", "boe_spot_annual.csv"))
    times = base_spot['year'].to_numpy(int)
    zeros = base_spot['rate'].to_numpy(float)
    yc = YieldCurve(times, zeros, 'BoE')

    cf_model = BaseCashflowModelImplementation(np.array([1, 2, 3, 4, 5]), yc)

    def test_discount_factors_no_spread(self):

        expected = np.array([[0.995033604], [0.98597451], [0.976972381], [0.967818165], [0.958167554]])
        actual = self.cf_model.discount_factors()

        assert expected.shape == actual.shape
        assert np.allclose(expected, actual)


    def test_discount_factors_with_spreads(self):

        expected = np.array([[0.966191804779663, 0.961546612495131, 0.947875219400885],
                             [0.929756655751588, 0.920855999692195, 0.894907646633322],
                             [0.894672718820515, 0.881864989697989, 0.844878789675949],
                             [0.860710311246354, 0.844327395784012, 0.797462798303495],
                             [0.82754885197482, 0.807913213071822, 0.752272381926347]])

        spreads = np.array([0.03, 0.035, 0.05])
        actual = self.cf_model.discount_factors(spreads)

        assert expected.shape == actual.shape
        assert np.allclose(expected, actual)

