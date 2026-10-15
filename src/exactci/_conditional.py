"""Exact conditional confidence intervals for the odds ratio.

Both methods condition on the table margins and work with Fisher's
noncentral hypergeometric law for cell (1,1) given the margins and a
candidate odds ratio psi. The distribution is evaluated in log space
from scratch (log-combinatorial weights via gammaln), so there is no
dependence on any library's noncentral-hypergeometric implementation
and no numerical cliff at extreme odds ratios. Tails are inverted by
bisection on the log-odds-ratio scale, so zero cells are handled
naturally (a limit becomes 0 or infinity instead of breaking).

cornfield: the classical exact conditional interval (what SAS PROC FREQ
  reports as the exact odds-ratio limits and what R's fisher.test
  returns). Lower limit solves P(X >= a | psi) = alpha/2, upper limit
  solves P(X <= a | psi) = alpha/2.
mid_p: the same construction with mid-p tail probabilities,
  P(X > a) + 0.5*P(X = a) and P(X < a) + 0.5*P(X = a). Shorter on
  average, but without the strict coverage guarantee.
"""

import numpy as np
from scipy.special import gammaln, logsumexp

# Largest support size we will enumerate exactly; beyond this the
# bisection would take too long, so we refuse loudly instead of hanging.
MAX_SUPPORT = 2000000

_LOG_LO = -700.0
_LOG_HI = 700.0


class _NoncentralHypergeometric:
    """Fisher's noncentral hypergeometric law for cell a given the margins."""

    def __init__(self, a, b, c, d):
        n1 = a + b
        m1 = a + c
        m2 = b + d
        N = n1 + c + d
        self.kmin = max(0, n1 + m1 - N)
        self.kmax = min(n1, m1)
        size = self.kmax - self.kmin + 1
        if size > MAX_SUPPORT:
            raise ValueError(
                f"exact conditional interval refuses tables whose (1,1) cell "
                f"has {size} possible values (> {MAX_SUPPORT}): enumeration "
                "would take too long. Large tables like this are where "
                "asymptotic intervals are reliable anyway."
            )
        ks = np.arange(self.kmin, self.kmax + 1, dtype=np.float64)
        # log C(m1,k) C(m2,n1-k); the psi-dependent term k*log(psi) is
        # added per evaluation.
        self._ks = ks
        self._base = (
            gammaln(m1 + 1) - gammaln(ks + 1) - gammaln(m1 - ks + 1)
            + gammaln(m2 + 1) - gammaln(n1 - ks + 1) - gammaln(m2 - n1 + ks + 1)
        )

    def tails(self, a, log_psi):
        """Return (P(X >= a), P(X <= a), P(X == a)) at odds ratio exp(log_psi)."""
        lw = self._base + self._ks * log_psi
        logp = lw - logsumexp(lw)
        ks = self._ks
        p_ge = float(np.exp(logsumexp(logp[ks >= a])))
        p_le = float(np.exp(logsumexp(logp[ks <= a])))
        p_eq = float(np.exp(logp[ks == a][0]))
        return p_ge, p_le, p_eq


def _bisect(f, lo, hi, target, increasing, n_iter=100, xtol=1e-13):
    """Bisect f(x) = target on [lo, hi]; f monotone (increasing flag)."""
    flo = f(lo) - target
    fhi = f(hi) - target
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        raise RuntimeError("non-finite tail probability while bracketing")
    ok = (flo <= 0 <= fhi) if increasing else (flo >= 0 >= fhi)
    if not ok:
        raise RuntimeError(
            "could not bracket the interval limit "
            f"(f(lo)-target={flo}, f(hi)-target={fhi})"
        )
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid) - target
        if not np.isfinite(fm):
            raise RuntimeError("non-finite tail probability during bisection")
        if increasing:
            if fm <= 0:
                lo = mid
            else:
                hi = mid
        else:
            if fm >= 0:
                lo = mid
            else:
                hi = mid
        if hi - lo < xtol:
            break
    return 0.5 * (lo + hi)


def cornfield_ci(a, b, c, d, alpha):
    nhg = _NoncentralHypergeometric(a, b, c, d)
    half = alpha / 2.0
    if a == nhg.kmin:
        lo = 0.0
    else:
        f = lambda lp: nhg.tails(a, lp)[0]  # P(X >= a), increasing in psi
        lo = float(np.exp(_bisect(f, _LOG_LO, _LOG_HI, half, increasing=True)))
    if a == nhg.kmax:
        hi = np.inf
    else:
        f = lambda lp: nhg.tails(a, lp)[1]  # P(X <= a), decreasing in psi
        hi = float(np.exp(_bisect(f, _LOG_LO, _LOG_HI, half, increasing=False)))
    return lo, hi


def mid_p_ci(a, b, c, d, alpha):
    nhg = _NoncentralHypergeometric(a, b, c, d)
    half = alpha / 2.0
    if a == nhg.kmin:
        lo = 0.0
    else:
        def f(lp):
            p_ge, _, p_eq = nhg.tails(a, lp)
            return (p_ge - p_eq) + 0.5 * p_eq  # P(X > a) + .5 P(X = a)
        lo = float(np.exp(_bisect(f, _LOG_LO, _LOG_HI, half, increasing=True)))
    if a == nhg.kmax:
        hi = np.inf
    else:
        def f(lp):
            _, p_le, p_eq = nhg.tails(a, lp)
            return (p_le - p_eq) + 0.5 * p_eq  # P(X < a) + .5 P(X = a)
        hi = float(np.exp(_bisect(f, _LOG_LO, _LOG_HI, half, increasing=False)))
    return lo, hi
