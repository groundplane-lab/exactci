"""exactci: exact confidence intervals for odds ratios and relative risk (free core)."""

from ._api import ci_odds_ratio, ci_relative_risk

__version__ = "0.1.0"

__all__ = ["ci_odds_ratio", "ci_relative_risk", "__version__"]
