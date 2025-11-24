import unittest
import numpy as np
import pandas as pd

from modelic.core.curves import YieldCurve
from modelic.core.mortality import MortalityTable
from modelic.core.policy_portfolio import PolicyPortfolio
from modelic.expenses.expense_engine import ExpenseEngine
from _data import data_path


class TestExpenseEngine(unittest.TestCase):

    try:
        expense_spec = pd.read_csv('./../Parameters/expense_spec.csv')
    except FileNotFoundError:
        expense_spec = pd.read_csv('./Parameters/expense_spec.csv')

    policies_in = PolicyPortfolio.from_csv(data_path('policy_data', 'mixed_policies.csv'))

    mort_raw = pd.read_csv(data_path('mortality', 'AM92.csv'))
    ages = mort_raw['x'].to_numpy(int)
    qx = mort_raw['q_x'].to_numpy(float)
    mortality = MortalityTable(ages, qx, 'AM92')

    disc_raw = pd.read_csv(data_path('curves', 'boe_spot_annual.csv'))
    times = disc_raw['year'].to_numpy(int)
    zeros = disc_raw['rate'].to_numpy(float)
    discount_curve = YieldCurve(times, zeros, 'BoE')

    expense_inflation_rate = 0.03

    eng = ExpenseEngine(expense_spec, discount_curve, mortality, expense_inflation_rate)


    def test_expense_engine_pv(self):

        expected_values = np.array(
            [[387.208762534007, 75, 30, 383.137742354004, 591.642335554445, 574.706613531006, 14.0814484556628, 0],
             [1149.20352282716, 90, 45, 1546.72691063214, 5207.09478477757, 2062.30254750952, 498.998525285899, 0],
             [11544.7052131714, 105, 60, 456.106586901951, 3725.83287880425, 570.133233627438, 6.09408560375387, 51.526875582443],
             [12425.5138964623, 60, 30, 201.608929100455, 3694.41168074955, 252.011161375569, 0, 48.5249669644056],
             [360, 150, 60, 1944.34553040531, 0, 2222.10917760607, 202.109394144322, 0]])

        actual = self.eng.present_value(self.policies_in, aggregate=False)

        assert actual.shape == expected_values.shape
        assert np.allclose(actual.values, expected_values)


    def test_expense_engine_pv_aggregated(self):

        expected = 50825.1368
        actual = self.eng.present_value(self.policies_in, aggregate=True)

        assert np.allclose(actual, expected)


    def test_expense_engine_pv_term_assurance(self):

        expected_values = np.array([387.208762534007, 75, 30, 383.137742354004, 591.642335554445, 574.706613531006,
                                    14.0814484556628])

        product_type = 'Term Assurance'
        ages = self.policies_in.get('ages', product_type)
        terms = self.policies_in.get('terms', product_type)
        premiums = self.policies_in.get('annual_premium', product_type)

        actual = self.eng.present_value_for_product(product_type, ages, terms, premiums, aggregate=False)

        assert actual.shape == expected_values.shape
        assert np.allclose(actual.values, expected_values)


    def test_expense_engine_pv_term_assurance_aggregated(self):

        expected = 2055.776902

        product_type = 'Term Assurance'
        ages = self.policies_in.get('ages', product_type)
        terms = self.policies_in.get('terms', product_type)
        premiums = self.policies_in.get('annual_premium', product_type)

        actual = self.eng.present_value_for_product(product_type, ages, terms, premiums, aggregate=True)

        assert np.allclose(actual, expected)

