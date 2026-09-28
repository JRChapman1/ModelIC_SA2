from __future__ import annotations

from typing import Any, Dict, Optional

from modelic.balance_sheets.base import BalanceSheetResult, BaseBalanceSheet


class SIIBalanceSheet(BaseBalanceSheet):
    """SII balance sheet assembly.

    This is intentionally kept lightweight at the skeleton stage so it can be
    upgraded with the actual standards and legacy outputs as the model grows.
    """

    def __init__(
        self,
        asset_value: float,
        liability_value: float,
        matching_adjustment: float = 0.0,
        risk_margin: float = 0.0,
        bscr: float = 0.0,
        own_funds: Optional[float] = None,
    ):
        super().__init__(asset_value=asset_value, liability_value=liability_value)
        self.matching_adjustment = float(matching_adjustment)
        self.risk_margin = float(risk_margin)
        self.bscr = float(bscr)
        self.own_funds = float(own_funds) if own_funds is not None else None

    def build(self) -> BalanceSheetResult:
        """Construct an SII balance-sheet result.

        The exact treatment of own funds, BSCR and risk margin should be aligned to
        the relevant legacy output once the parity work is complete.
        """
        if self.own_funds is None:
            own_funds = self.asset_value - (self.liability_value - self.matching_adjustment)
        else:
            own_funds = float(self.own_funds)

        excess_own_funds = own_funds - self.bscr

        return BalanceSheetResult(
            regime="SII",
            assets=self.asset_value,
            liabilities=self.liability_value,
            surplus=self.surplus,
            details={
                "matching_adjustment": self.matching_adjustment,
                "risk_margin": self.risk_margin,
                "bscr": self.bscr,
                "own_funds": own_funds,
                "excess_own_funds": excess_own_funds,
            },
        )

    def validate(self) -> None:
        """Placeholder validation hooks for SII-specific checks."""
        return None

