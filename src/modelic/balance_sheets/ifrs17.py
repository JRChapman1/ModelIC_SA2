from __future__ import annotations

from typing import Optional

from modelic.balance_sheets.base import BalanceSheetResult, BaseBalanceSheet


class IFRS17BalanceSheet(BaseBalanceSheet):
    """IFRS 17 balance sheet assembly.

    This is intentionally a skeleton. The exact IFRS 17 structure can vary by
    product book and reporting basis; this class provides the common result object
    and a reasonable starting point for the fuller implementation.
    """

    def __init__(
        self,
        asset_value: float,
        liability_value: float,
        csm: float = 0.0,
        risk_adjustment: float = 0.0,
        fulfilment_cashflows: float = 0.0,
        other_liabilities: float = 0.0,
        vir_adjustment: Optional[float] = None,
    ):
        super().__init__(asset_value=asset_value, liability_value=liability_value)
        self.csm = float(csm)
        self.risk_adjustment = float(risk_adjustment)
        self.fulfilment_cashflows = float(fulfilment_cashflows)
        self.other_liabilities = float(other_liabilities)
        self.vir_adjustment = float(vir_adjustment) if vir_adjustment is not None else None

    def build(self) -> BalanceSheetResult:
        """Construct an IFRS 17 balance-sheet result."""
        if self.vir_adjustment is None:
            net_liability = self.liability_value
            vir_adjustment = 0.0
        else:
            net_liability = self.liability_value + self.vir_adjustment
            vir_adjustment = float(self.vir_adjustment)

        equity = self.asset_value - net_liability

        return BalanceSheetResult(
            regime="IFRS17",
            assets=self.asset_value,
            liabilities=net_liability,
            surplus=equity,
            details={
                "contractual_service_margin": self.csm,
                "risk_adjustment": self.risk_adjustment,
                "fulfilment_cashflows": self.fulfilment_cashflows,
                "other_liabilities": self.other_liabilities,
                "vir_adjustment": vir_adjustment,
                "net_liability": net_liability,
                "equity": equity,
            },
        )

    def validate(self) -> None:
        """Placeholder validation hooks for IFRS 17-specific checks."""
        return None

