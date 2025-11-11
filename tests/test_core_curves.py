import unittest
import pandas as pd

from modelic.core.curves import YieldCurve




class TestYieldCurve(unittest.TestCase):

    base_spot = pd.read_csv("data/curves/boe_spot_annual.csv")
    times = base_spot['year'].to_numpy(int)
    zeros = base_spot['rate'].to_numpy(float)
    yc = YieldCurve(times, zeros, 'BoE')


