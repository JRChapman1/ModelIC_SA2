import numpy as np
import pandas as pd

from modelic.core.utils import calculate_irr, accumulate_cashflows
from modelic.core.curves import YieldCurve, FSTable
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

fs_table = FSTable.from_csv(data_path("curves", "fs_table_gbp_non_fin.csv"))


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


def test_accumulate_cashflows():

    expected = np.array([12, 46.36, 97.84352, 148.2680432, 188.6056927552, 204.018286308877, 212.75088975707])

    cfs = [12, 34, 50, 47, 35, 9, 2]
    fwd = [0.03, 0.032, 0.035, 0.036, 0.034, 0.033, 0.031]
    actual = accumulate_cashflows(cfs, fwd)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)

