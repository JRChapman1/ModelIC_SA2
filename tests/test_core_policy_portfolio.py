import unittest
import numpy as np

from modelic.core.policy_portfolio import PolicyPortfolio


class TestPolicyPortfolio(unittest.TestCase):

    policy_wol_csv = PolicyPortfolio.from_csv(r'../data/policy_data/wol_test_data.csv')

    def test_policy_portfolio_wol_csv(self):

        expected_ages = np.array([34, 47, 73])
        expected_death_contingent_benefits = np.array([5, 10, 15])

        assert self.policy_wol_csv.ages.shape == expected_ages.shape
        assert np.allclose(self.policy_wol_csv.ages, expected_ages)

        assert self.policy_wol_csv.death_contingent_benefits.shape == expected_death_contingent_benefits.shape
        assert np.allclose(self.policy_wol_csv.death_contingent_benefits, expected_death_contingent_benefits)

        assert self.policy_wol_csv.terms is None
        assert self.policy_wol_csv.terminal_survival_contingent_benefits is None
        assert self.policy_wol_csv.periodic_survival_contingent_benefits is None