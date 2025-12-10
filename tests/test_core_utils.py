import numpy as np
import pandas as pd

from modelic.core.utils import calculate_irr
from modelic.core.curves import YieldCurve, SpreadTable
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

fs_table = SpreadTable.from_csv(data_path("curves", "fs_table_gbp_non_fin.csv"))


def test_calculate_portfolio_gry():

    expected = 0.0176137621027148

    asset_cfs = bond_model.project_cashflows(aggregate=True)
    bond_mvs = bond_model.present_value(aggregate=True)

    actual = calculate_irr(asset_cfs, bond_mvs)

    assert np.allclose(actual, expected)


def test_calculate_bel_rfr():

    expected = 0.011456925174404553

    actual = calculate_irr(annuity_model.project_cashflows(aggregate=True),
                                annuity_model.present_value(aggregate=True))

    assert np.allclose(actual, expected)


