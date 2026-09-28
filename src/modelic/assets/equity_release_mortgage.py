import pandas as pd

from modelic.core import *


class ERM(CompositeProduct):

    """ Projects cashflows and calculates present values for ERMs """

    def __init__(self,
                 yield_curve: YieldCurve,
                 mortality_table: MortalityTable,
                 policyholder_age: IntArrayLike,
                 policy_accrual_rate: ArrayLike,
                 loan_balance: float = 1,
                 *,
                 projection_steps: IntArrayLike = None):

        components = [DeathContingentCashflow(yield_curve=yield_curve,
                                              mortality_table=mortality_table,
                                              ph_age=policyholder_age,
                                              death_contingent_cf=loan_balance,
                                              escalation=policy_accrual_rate)]

        super().__init__(components, yield_curve)

    @classmethod
    def from_asset_portfolio(cls, asset_portfolio: AssetPortfolio, yield_curve: YieldCurve, *,
                             projection_steps: IntArrayLike = None):

        return cls(yield_curve=yield_curve,
                   notional=asset_portfolio.notional,
                   coupon_rate=asset_portfolio.coupon_rate,
                   maturity=asset_portfolio.maturity,
                   spread=asset_portfolio.spread,
                   projection_steps=projection_steps)


if __name__ == '__main__':

    # Set up discounting assumptions
    disc_raw = pd.read_csv(r'/Users/joshchapman/PycharmProjects/ModelIC/tests/data/curves/boe_spot_annual.csv')
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    # Set up mortality assumptions
    mort_raw = pd.read_csv(r'/Users/joshchapman/PycharmProjects/ModelIC/tests/data/mortality/AM92.csv')
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    erm_1 = ERM(discount_curve, mortality, 72, 0.07, 50_000)
    print(erm_1.present_value())