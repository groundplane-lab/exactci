# exactci

Free core of ExactCI2x2: exact confidence intervals for odds ratios and relative risk in 2x2 tables.

**The problem**
Your 2x2 table shows an effect and you need a confidence interval for the odds ratio or the relative risk. scipy has no interval for either. Your options are: quote a Wald interval and hope the coverage holds, learn R, or pip install this.

**What it does**
`ci_odds_ratio(table)` gives the Cornfield exact conditional interval (the one R's fisher.test and SAS report), and `method="mid_p"` gives the shorter mid-p version. `ci_relative_risk(table)` gives Koopman's score interval. On 2x2 tables the Cornfield limits match R's fisher.test to 1e-4. Search terms: odds ratio confidence interval Python, relative risk confidence interval Python, Cornfield exact CI, Koopman interval.

**Limits**
One 2x2 table per call. Cornfield and mid-p condition on the table margins; Koopman is a score interval, not an exact one. The exact unconditional intervals (Baptista-Pike, Chan-Zhang) and the Agresti-Min score interval are not in this free core. No batch mode, no CLI.

**Install**

```bash
pip install exactci
```

Needs Python 3.9 or newer, NumPy, and SciPy.

```python
from exactci import ci_odds_ratio, ci_relative_risk
print(ci_odds_ratio([[11, 4], [2, 6]]))
print(ci_relative_risk([[11, 4], [2, 6]]))
```

This is the free core: exact conditional and score confidence intervals for odds ratios and relative risk in 2x2 tables. The full version ($30, one-time) adds the Baptista-Pike and Chan-Zhang exact unconditional intervals, the Agresti-Min unconditional score interval, batch mode, and CLI: https://groundplane0.gumroad.com/l/verudy
