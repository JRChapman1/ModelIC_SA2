from __future__ import annotations

from typing import Any, Dict

from modelic.balance_sheets.ifrs17 import IFRS17BalanceSheet
from modelic.balance_sheets.sii import SIIBalanceSheet


class BalanceSheetFactory:
    """Factory for creating balance-sheet outputs by regime."""

    @staticmethod
    def create(regime: str, **kwargs: Any):
        key = (regime or "").strip().lower()

        if key in {"sii", "solvency_ii"}:
            return SIIBalanceSheet(**kwargs)
        if key in {"ifrs17", "ifrs_17", "ifrs"}:
            return IFRS17BalanceSheet(**kwargs)

        valid = ["SII", "IFRS17"]
        raise ValueError(f"Unsupported regime '{regime}'. Expected one of: {valid}")

