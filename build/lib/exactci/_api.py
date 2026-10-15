"""Public API: ci_odds_ratio and ci_relative_risk (free core, single tables).

Free methods: "cornfield" and "mid_p" for the odds ratio, "koopman" for
the relative risk. Premium methods (baptista_pike, agresti_min,
chan_zhang) refuse loudly with the paid URL instead of a plain method
error. One 2x2 table per call; batch input is refused loudly.
"""

import numpy as np

from ._validate import (
    _PAID_URL,
    _check_alpha,
    _check_method,
    _as_single_table,
    _OR_FREE,
    _RR_FREE,
    _OR_PREMIUM,
    _RR_PREMIUM,
)
from ._conditional import cornfield_ci, mid_p_ci
from ._score import koopman_ci

_OR_FUNCS = {
    "cornfield": cornfield_ci,
    "mid_p": mid_p_ci,
}

_RR_FUNCS = {
    "koopman": koopman_ci,
}


def _paid_only(method):
    raise ValueError(
        f"method {method!r} is not in the free core. The full version "
        "($30, one-time) adds the Baptista-Pike and Chan-Zhang exact "
        "unconditional intervals, the Agresti-Min unconditional score "
        f"interval, batch mode, and CLI: {_PAID_URL}"
    )


def ci_odds_ratio(table, alpha=0.05, method="cornfield"):
    """Confidence interval for the odds ratio of a 2x2 table.

    table: [[a, b], [c, d]] with row sums n1 = a+b, n2 = c+d.
    method: "cornfield" (exact conditional, default) or "mid_p"
      (conditional mid-p). The premium methods "baptista_pike" and
      "agresti_min" refuse loudly with the paid URL.
    Returns an array [lo, hi]. One table per call.
    """
    alpha = _check_alpha(alpha)
    if method in _OR_PREMIUM:
        _paid_only(method)
    method = _check_method(method, _OR_FREE)
    t = _as_single_table(table)
    a, b, c, d = (int(v) for v in t.flat)
    lo, hi = _OR_FUNCS[method](a, b, c, d, alpha)
    return np.array([lo, hi])


def ci_relative_risk(table, alpha=0.05, method="koopman"):
    """Confidence interval for the relative risk of a 2x2 table.

    table: [[a, b], [c, d]]; RR = (a/(a+b)) / (c/(c+d)).
    method: "koopman" (score, default). The premium method
      "chan_zhang" refuses loudly with the paid URL.
    Returns an array [lo, hi]. One table per call.
    """
    alpha = _check_alpha(alpha)
    if method in _RR_PREMIUM:
        _paid_only(method)
    method = _check_method(method, _RR_FREE)
    t = _as_single_table(table)
    a, b, c, d = (int(v) for v in t.flat)
    lo, hi = _RR_FUNCS[method](a, b, c, d, alpha)
    return np.array([lo, hi])
