from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BalanceSheetResult:
    """Standard result object returned by all balance-sheet builders.

    The idea is to keep regime-specific outputs in one place while still exposing a
    common structure for reporting, tests, and UI layers.
    """

    regime: str
    assets: float
    liabilities: float
    surplus: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a flat dictionary for reporting or JSON export."""
        out = {
            "regime": self.regime,
            "assets": self.assets,
            "liabilities": self.liabilities,
            "surplus": self.surplus,
        }
        out.update(self.details)
        return out


class BaseBalanceSheet(ABC):
    """Common interface for regime-specific balance-sheet builders."""

    def __init__(self, asset_value: float, liability_value: float):
        self.asset_value = float(asset_value)
        self.liability_value = float(liability_value)

    @property
    def surplus(self) -> float:
        return self.asset_value - self.liability_value

    @abstractmethod
    def build(self) -> BalanceSheetResult:
        """Construct and return the reporting result for a given regime."""
        raise NotImplementedError

    def validate(self) -> None:
        """Hook for regime-specific validation checks."""
        return None

