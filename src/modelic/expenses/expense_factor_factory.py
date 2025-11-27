# modelic/expenses/expense_factor_factory.py

from modelic.expenses.expense_timings import ExpenseTiming
from modelic.core.contingent_cashflows.death_contingent_cashflow import DeathContingentCashflow
from modelic.core.contingent_cashflows.survival_contingent_cashflow import SurvivalContingentCashflow


EXPENSE_FACTOR_FACTORY = {
    ExpenseTiming.RENEWAL: {'cls': SurvivalContingentCashflow, 'kwargs': {'periodic_cf': 1}, 'term_offset': -1},
    ExpenseTiming.SURVIVAL: {'cls': SurvivalContingentCashflow, 'kwargs': {'terminal_cf': 1}, 'term_offset': 0},
    ExpenseTiming.DEATH: {'cls': DeathContingentCashflow, 'kwargs': {'death_contingent_cf': 1}, 'term_offset': 0}
}

