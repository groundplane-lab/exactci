"""Koopman's (1984) score confidence interval for the relative risk.

The statistic is
    T(psi) = (p1_hat - psi*p2_hat) / sqrt(p1t q1t/n1 + psi^2 p2t q2t/n2)
with (p1_tilde, p2_tilde) the constrained MLEs under p1 = psi*p2
(closed-form quadratic). T decreases in psi, so the limits solve
T(lo) = z, T(hi) = -z.

Zero cells are handled: with a zero cell the point estimate is 0 or
infinity and the interval simply extends to that boundary.
"""

import numpy as np
from scipy.stats import norm

from ._conditional import _bisect


def _koopman_mle(a, b, c, d, psi):
    """Constrained MLEs (p1, p2) under p1 = psi*p2 (relative risk = psi).

    The profile score equation is quadratic in p2:
    A = N*psi, B = -((a+c)(1+psi) + b*psi + d), C = a+c,
    and the valid root is the smaller one (it satisfies p2 <= 1/psi).
    Solved with the stable form of the quadratic formula.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    n1 = a + b
    n2 = c + d
    A = (n1 + n2) * psi
    B = -((a + c) * (1.0 + psi) + b * psi + d)
    C = a + c
    disc = B * B - 4.0 * A * C
    # disc >= 0 mathematically; a slightly negative value is fp noise.
    tol = 1e-9 * np.maximum(B * B, 1e-300)
    if np.any(disc < -tol):
        raise RuntimeError("negative discriminant in constrained MLE")
    sd = np.sqrt(np.maximum(disc, 0.0))
    # B < 0 always here, so q = 0.5*(|B| + sd) > 0 and p2 = C/q is the
    # smaller root, no cancellation. Boundary tables can land on 0/1
    # up to floating-point noise, so clip instead of rejecting.
    q = -0.5 * (B - sd)
    with np.errstate(all="ignore"):
        p2 = C / np.where(q == 0.0, np.nan, q)
    # the valid root satisfies p2 <= 1/psi (so p1 = psi*p2 <= 1)
    p2 = np.clip(p2, 0.0, min(1.0, 1.0 / psi))
    p1 = psi * p2
    if np.any(~np.isfinite(p2)):
        raise RuntimeError("constrained MLE is not finite")
    return p1, p2


def _score_rr(a, b, c, d, psi):
    """Koopman score statistic for H0: RR = psi (decreasing in psi)."""
    n1 = a + b
    n2 = c + d
    p1t, p2t = _koopman_mle(a, b, c, d, psi)
    p1h = a / n1
    p2h = c / n2
    with np.errstate(all="ignore"):
        v = p1t * (1.0 - p1t) / n1 + psi * psi * p2t * (1.0 - p2t) / n2
    v = np.where(v <= 0, np.nan, v)
    t = (p1h - psi * p2h) / np.sqrt(v)
    if np.any(~np.isfinite(t)):
        raise RuntimeError("non-finite score statistic")
    return t


def _solve_decreasing(f, target, lo_start, hi_start,
                      lo_floor=1e-300, hi_ceil=1e300):
    """Solve f(x) = target for decreasing f, expanding the bracket as needed."""
    lo, hi = lo_start, hi_start
    flo = f(lo)
    while not (np.all(np.isfinite(flo)) and np.all(flo > target)):
        lo = lo / 2.0
        if lo < lo_floor:
            raise RuntimeError("could not bracket interval limit (lower side)")
        flo = f(lo)
    fhi = f(hi)
    while not (np.all(np.isfinite(fhi)) and np.all(fhi < target)):
        hi = hi * 2.0
        if hi > hi_ceil:
            raise RuntimeError("could not bracket interval limit (upper side)")
        fhi = f(hi)
    return _bisect(f, lo, hi, target, increasing=False)


def _rr_hat(a, b, c, d):
    n1 = a + b
    n2 = c + d
    p1h = a / n1
    p2h = c / n2
    if p1h == 0.0:
        return 0.0
    if p2h == 0.0:
        return np.inf
    return p1h / p2h


def koopman_ci(a, b, c, d, alpha):
    z = float(norm.ppf(1.0 - alpha / 2.0))
    est = _rr_hat(a, b, c, d)
    T = lambda psi: float(_score_rr(a, b, c, d, psi))
    if est == 0.0:
        lo = 0.0
    elif np.isinf(est):
        lo = _solve_decreasing(T, z, 1e-8, 1.0)
    else:
        lo = _solve_decreasing(T, z, est / 2.0, est)
    if np.isinf(est):
        hi = np.inf
    else:
        hi = _solve_decreasing(T, -z, max(est, 1e-8), max(2.0 * est, 1.0))
    return lo, hi
