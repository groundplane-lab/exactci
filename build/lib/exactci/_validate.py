"""Input validation for the exactci free core.

Every public entry point funnels through here. The policy is simple:
a bad input raises loudly (ValueError or TypeError with a plain message).
Nothing ever silently computes on garbage, and a degenerate table
(a zero row or column margin) is rejected instead of producing a
meaningless interval.

Free core: single 2x2 tables only. Batch input is refused loudly.
"""

import numpy as np

_PAID_URL = "https://groundplane0.gumroad.com/l/verudy"

# Methods shipped in this free core.
_OR_FREE = ("cornfield", "mid_p")
_RR_FREE = ("koopman",)

# Premium methods live in the paid product; requesting one raises the
# paid upsell instead of a plain method error (see _api._paid_only).
_OR_PREMIUM = ("baptista_pike", "agresti_min")
_RR_PREMIUM = ("chan_zhang",)


def _check_alpha(alpha):
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise TypeError(f"alpha must be a number in (0, 1), got {alpha!r}")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly inside (0, 1), got {alpha!r}")
    return alpha


def _check_method(method, allowed):
    if not isinstance(method, str) or method not in allowed:
        raise ValueError(f"method must be one of {list(allowed)}, got {method!r}")
    return method


def _as_int_table(table):
    """Validate one 2x2 table and return it as an int64 array."""
    arr = np.asarray(table)
    if arr.dtype == object:
        raise TypeError(f"table entries must be numbers, got {table!r}")
    if arr.dtype.kind == "b":
        raise TypeError(f"table entries must be counts, got boolean array {table!r}")
    if arr.dtype.kind == "c":
        raise TypeError(f"table entries must be real counts, got complex {table!r}")
    if arr.shape != (2, 2):
        raise ValueError(f"table must have shape (2, 2), got shape {arr.shape}")
    try:
        vals = arr.astype(np.float64)
    except (ValueError, TypeError):
        raise TypeError(f"table entries must be numbers, got {table!r}")
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"table entries must be finite, got {table!r}")
    if np.any(vals < 0):
        raise ValueError(f"table entries must be non-negative counts, got {table!r}")
    if not np.all(vals == np.floor(vals)):
        raise ValueError(f"table entries must be whole-number counts, got {table!r}")
    if np.any(vals > np.iinfo(np.int64).max):
        raise ValueError(
            f"table entries are absurdly large (>{np.iinfo(np.int64).max}); "
            "a confidence interval is undefined"
        )
    t = vals.astype(np.int64)
    n1, n2 = t[0].sum(), t[1].sum()
    m1, m2 = t[:, 0].sum(), t[:, 1].sum()
    if n1 == 0 or n2 == 0:
        raise ValueError(
            f"degenerate table: a whole row is zero (row sums {n1}, {n2}); "
            "a confidence interval is undefined"
        )
    if m1 == 0 or m2 == 0:
        raise ValueError(
            f"degenerate table: a whole column is zero (column sums {m1}, {m2}); "
            "a confidence interval is undefined"
        )
    return t


def _as_single_table(table):
    """Validate one 2x2 table; refuse batch input loudly.

    The free core handles one table per call. A 3-d array or a list of
    tables is a batch, which is a paid feature.
    """
    arr = np.asarray(table)
    if arr.ndim == 3:
        raise ValueError(
            "this free core handles one 2x2 table per call, not a batch. "
            f"The full version ($30, one-time) adds batch mode: {_PAID_URL}"
        )
    if isinstance(table, (list, tuple)) and len(table) > 0:
        try:
            stacked = np.asarray(table)
        except (ValueError, TypeError):
            raise TypeError(f"could not interpret input {table!r} as a table")
        if stacked.ndim == 3:
            raise ValueError(
                "this free core handles one 2x2 table per call, not a batch. "
                f"The full version ($30, one-time) adds batch mode: {_PAID_URL}"
            )
    return _as_int_table(table)
