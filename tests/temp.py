import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.products.endowment import Endowment
from modelic.products.annuity import Annuity
from _data import data_path


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
annuity = Annuity(policies_wol, discount_curve, mortality)