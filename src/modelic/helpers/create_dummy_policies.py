# modelic/helpers/create_dummy_policies.py

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.core.contingent_cashflows.survival_contingent_cashflow import _SurvivalContingentCashflow
from modelic.products.endowment import Endowment
from modelic.products.annuity import Annuity
from modelic.products.life_assurance import LifeAssurance
from modelic.products.pure_endowment import PureEndowment
from modelic.core.custom_types import IntArrayLike, ArrayLike
from modelic.core.udf_globals import policy_data_csv_columns


mort_raw = pd.read_csv(r"../../../tests/data/mortality/AM92.csv")
ages = mort_raw['x'].to_numpy(int)
qx = mort_raw['q_x'].to_numpy(float)
mortality = MortalityTable(ages, qx, 'AM92')

disc_raw = pd.read_csv(r"../../../tests/data/curves/boe_spot_annual.csv")
times = disc_raw['year'].to_numpy(int)
zeros = disc_raw['rate'].to_numpy(float)
discount_curve = YieldCurve(times, zeros, 'BoE')


def create_dummy_assurance_policies(ph_age: IntArrayLike, term: IntArrayLike, sum_assured: ArrayLike, loading: float = 0):

    policies = PolicyPortfolio(ph_age, term, death_contingent_benefits=sum_assured)
    life_assurance = LifeAssurance(policies, discount_curve, mortality)
    pvs = life_assurance.present_value(aggregate=False)
    prems = _goalseek_regular_premium(pvs, ph_age, term)

    policies.annual_premium = prems * (1 + loading)

    return policies


def create_dummy_pure_endowment_policies(ph_age: IntArrayLike, term: IntArrayLike, sum_assured: ArrayLike, loading: float = 0):

    policies = PolicyPortfolio(ph_age, term, terminal_survival_contingent_benefits=sum_assured)
    pure_endowment = PureEndowment(policies, discount_curve, mortality)
    pvs = pure_endowment.present_value(aggregate=False)
    prems = _goalseek_regular_premium(pvs, ph_age, term)

    policies.annual_premium = prems * (1 + loading)

    return policies


def create_dummy_endowment_policies(ph_age: IntArrayLike, term: IntArrayLike, death_ben: ArrayLike, surv_ben: ArrayLike, loading: float = 0):

    policies = PolicyPortfolio(ph_age, term, death_contingent_benefits=death_ben, terminal_survival_contingent_benefits=surv_ben)
    pure_endowment = Endowment(policies, discount_curve, mortality)
    pvs = pure_endowment.present_value(aggregate=False)
    prems = _goalseek_regular_premium(pvs, ph_age, term)

    policies.annual_premium = prems * (1 + loading)

    return policies


def create_dummy_annuity_policies(ph_age: IntArrayLike, term: IntArrayLike, amount: ArrayLike, loading: float = 0):

    policies = PolicyPortfolio(ph_age, term, periodic_survival_contingent_benefits=amount)
    pure_endowment = PureEndowment(policies, discount_curve, mortality)
    pvs = pure_endowment.present_value(aggregate=False)

    policies.annual_premium = pvs * (1 + loading)

    return policies


def _goalseek_regular_premium(target: ArrayLike, ph_age: IntArrayLike, term: IntArrayLike):

    results = []

    for i in range(ph_age.size):

        age = ph_age[i]
        n = term[i]

        def _objective_function(x):
            cf = _SurvivalContingentCashflow(discount_curve, mortality, age, n, periodic_cf=x)
            return cf.present_value(aggregate=False) - target[i]

        results.append(root_scalar(_objective_function, bracket=[0, target[i]]).root)

    return np.asarray(results, dtype=np.float64)

if __name__ == '__main__':

    pols = {
        'Term Assurance': create_dummy_assurance_policies(np.array([34, 47, 73]), np.array([21, 13, 2]), [150_000, 70_000, 10_000], 0.1),
        'Whole-of-Life Assurance': create_dummy_assurance_policies(np.array([43, 55, 69]), np.array([np.nan, np.nan, np.nan]), [50_000, 10_000, 15_000], 0.1),
        'Pure Endowment': create_dummy_pure_endowment_policies(np.array([40, 53, 76]), np.array([15, 7, 4]), [210_000, 170_000, 20_000], 0.1),
        'Endowment': create_dummy_endowment_policies(np.array([41, 45, 53]), np.array([9, 10, 5]), [12_000, 125_000, 52_000], [12_000, 125_000, 52_000], 0.1),
        'Annuity': create_dummy_annuity_policies(np.array([58, 68, 78]), np.array([np.nan, np.nan, np.nan]), [50_000, 10_000, 15_000], 0.1)
    }

    pol_data_out = pd.DataFrame(columns=policy_data_csv_columns)

    for k, v in pols.items():
        data = v.data
        data['policy_type'] = k
        pol_data_out = pd.concat([pol_data_out, data], axis=0, sort=False)

    pol_data_out = pol_data_out.fillna('')

    pol_data_out.to_csv(r'/Users/joshchapman/PycharmProjects/ModelIC/tests/data/policy_data/mixed_policies.csv')