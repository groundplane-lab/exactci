"""Tests for the exactci free core.

Correctness is checked against the paid package (imported from its own
source tree, never copied) and against the published R exact2x2
vignette values for Cornfield.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import exactci

HERE = Path(__file__).resolve().parent
PAID_SRC = HERE.parent.parent.parent / "exact-ci" / "src" / "exactci"

_PAID_URL = "https://groundplane0.gumroad.com/l/verudy"


def _load_paid():
    spec = importlib.util.spec_from_file_location(
        "paid_exactci",
        PAID_SRC / "__init__.py",
        submodule_search_locations=[str(PAID_SRC)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paid_exactci"] = mod
    spec.loader.exec_module(mod)
    return mod


paid = _load_paid()

TABLES = [
    [[11, 4], [2, 6]],
    [[6, 12], [12, 5]],
    [[8, 2], [1, 5]],
    [[0, 5], [5, 3]],
    [[3, 0], [2, 7]],
    [[20, 15], [10, 25]],
]


def test_or_matches_paid_cornfield():
    for t in TABLES:
        free = exactci.ci_odds_ratio(t, method="cornfield")
        ref = paid.ci_odds_ratio(t, method="cornfield")
        assert free == pytest.approx(ref, rel=1e-9), t


def test_or_matches_paid_mid_p():
    for t in TABLES:
        free = exactci.ci_odds_ratio(t, method="mid_p")
        ref = paid.ci_odds_ratio(t, method="mid_p")
        assert free == pytest.approx(ref, rel=1e-9), t


def test_rr_matches_paid_koopman():
    for t in TABLES:
        free = exactci.ci_relative_risk(t, method="koopman")
        ref = paid.ci_relative_risk(t, method="koopman")
        assert free == pytest.approx(ref, rel=1e-9), t


def test_cornfield_matches_r_exact2x2():
    # R exact2x2 vignette: matrix(c(6,12,12,5), 2, 2) gives
    # 95% CI 0.03888003 to 1.05649145 (R's solver tol is 1e-5).
    lo, hi = exactci.ci_odds_ratio([[6, 12], [12, 5]], method="cornfield")
    assert lo == pytest.approx(0.03888003, rel=1e-4)
    assert hi == pytest.approx(1.05649145, rel=1e-4)


def _refuses_loudly(fn, *args, **kwargs):
    t0 = time.perf_counter()
    with pytest.raises(ValueError, match="groundplane0.gumroad.com/l/verudy"):
        fn(*args, **kwargs)
    assert time.perf_counter() - t0 < 1.0


def test_premium_or_methods_refuse():
    t = [[11, 4], [2, 6]]
    _refuses_loudly(exactci.ci_odds_ratio, t, method="baptista_pike")
    _refuses_loudly(exactci.ci_odds_ratio, t, method="agresti_min")


def test_premium_rr_method_refuses():
    t = [[11, 4], [2, 6]]
    _refuses_loudly(exactci.ci_relative_risk, t, method="chan_zhang")


def test_batch_refuses_loudly():
    batch = [[[11, 4], [2, 6]], [[6, 12], [12, 5]]]
    _refuses_loudly(exactci.ci_odds_ratio, batch)
    _refuses_loudly(exactci.ci_relative_risk, np.array(batch))


def test_premium_symbols_absent():
    import exactci._api as api

    for name in ("baptista_pike", "agresti_min", "chan_zhang"):
        assert not hasattr(api, name), name
    assert "baptista_pike" not in api._OR_FUNCS
    assert "agresti_min" not in api._OR_FUNCS
    assert "chan_zhang" not in api._RR_FUNCS
    with pytest.raises(ImportError):
        import exactci._unconditional  # noqa
    with pytest.raises(ImportError):
        import exactci._cli  # noqa


def test_garbage_inputs_raise():
    t = [[11, 4], [2, 6]]
    with pytest.raises((ValueError, TypeError)):
        exactci.ci_odds_ratio(t, alpha=0.0)
    with pytest.raises((ValueError, TypeError)):
        exactci.ci_odds_ratio(t, alpha=1.5)
    with pytest.raises(ValueError):
        exactci.ci_odds_ratio(t, method="nope")
    with pytest.raises((ValueError, TypeError)):
        exactci.ci_odds_ratio([[0, 0], [2, 6]])
    with pytest.raises((ValueError, TypeError)):
        exactci.ci_odds_ratio([[1, 2, 3], [4, 5, 6]])
    with pytest.raises((ValueError, TypeError)):
        exactci.ci_relative_risk(t, method="bogus")


def test_interval_sanity():
    lo, hi = exactci.ci_odds_ratio([[11, 4], [2, 6]])
    assert 0.0 <= lo <= hi
    lo, hi = exactci.ci_relative_risk([[11, 4], [2, 6]])
    assert 0.0 <= lo <= hi
    # default methods are the free ones
    assert exactci.ci_odds_ratio([[11, 4], [2, 6]])[0] == pytest.approx(
        exactci.ci_odds_ratio([[11, 4], [2, 6]], method="cornfield")[0]
    )
