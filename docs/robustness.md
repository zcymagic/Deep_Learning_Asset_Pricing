# Robustness to rule thresholds and knots

`scripts/run_knot_threshold_robustness.py` repeats the main checks on both panels: the CPZ panel
([cpz_results.md](cpz_results.md)) and the CRSP + OSAP panel, 1967–2016 bridge splits
([osap_results.md](osap_results.md)). Discovery uses 1967–1986; confirmation uses 1987–2016. Tables are
in [`robustness/`](robustness/). The state-variable robustness check is in `cpz_results.md`.

## Part A: audit rule thresholds

The fitted SDFs are re-audited under five rule grids. The fixed rule class of the global test changes
with each grid.

| grid | characteristic cut-points | state quantiles | global-test class |
|---|---|---|---|
| baseline | ±0.1, ±0.3 | 0.2 / 0.5 / 0.8 | ±0.3 × 0.2 / 0.8 |
| coarse | ±0.3 | 0.2 / 0.8 | same |
| fine | ±0.1, ±0.2, ±0.3, ±0.4 | 0.1 / 0.25 / 0.5 / 0.75 / 0.9 | ±0.2, ±0.4 × 0.1 / 0.9 |
| extreme | ±0.2, ±0.4 | 0.1 / 0.5 / 0.9 | ±0.4 × 0.1 / 0.9 |
| min_stocks 50 | baseline | baseline | baseline, but at least 50 stocks per rule portfolio |

**Does each SDF have a confirmed short-term-reversal failure under each grid?** ✓ = confirmed (max-t
p < 0.05), followed by the hold-out t of the top confirmed reversal rule. ✗ = no confirmed reversal
rule.

| SDF | panel | baseline | coarse | fine | extreme | min_stocks 50 |
|---|---|---|---|---|---|---|
| CAPM | CPZ | ✓ −9.8 | ✓ −10.4 | ✓ −10.1 | ✓ −10.1 | ✓ −9.8 |
| CAPM | OSAP | ✓ −9.6 | ✓ −10.2 | ✓ −9.6 | ✓ −7.9 | ✓ −8.5 |
| FF3 | CPZ | ✓ −10.3 | ✓ −9.7 | ✓ −9.3 | ✓ −9.7 | ✓ −10.3 |
| FF3 | OSAP | ✓ −11.0 | ✓ −11.0 | ✓ −11.1 | ✓ −10.8 | ✓ −10.3 |
| FF5 + UMD | CPZ | ✓ −8.5 | ✓ (weak, −1.8) | ✓ −7.3 | ✓ (weak, −2.7) | ✓ −8.5 |
| FF5 + UMD | OSAP | ✓ −8.3 | ✓ −8.4 | ✓ −8.3 | ✓ −8.2 | ✓ (top rule is NI × NOA, −7.8) |
| additive glass-box | CPZ | ✓ −2.9 | ✓ −1.7 | ✓ −3.0 | ✗ | ✓ −3.1 |
| additive glass-box | OSAP | ✗ | ✗ | ✗ | ✓ −1.7 | ✗ |
| CPZ GAN | CPZ | ✗ | ✗ | ✗ | ✗ | ✗ |

* **The reversal failure of the factor models is robust to the rule grid.** CAPM, FF3 and FF5 + UMD
  have a confirmed reversal failure under every grid on both panels. It is almost always the top
  confirmed failure, with hold-out t from −7 to −11.

  In the two weak FF5 + UMD cases on the CPZ panel, the confirmed reversal rule is a broader variant:
  `ST_REV > 0.3` alone in the coarse grid, and `ST_REV > 0.2` with high residual variance in the
  extreme grid. These grids lack the Rel2High < −0.1 cut that defines the baseline rule.
* **The glass-box and GAN results are also stable:**
  * the static additive glass-box SDF has at most a small reversal failure (|t| ≤ 3.1), and none in
    most OSAP grids;
  * the CPZ GAN has no confirmed failure under any grid.
* **Global tests change little.** The largest hold-out |t| moves by at most 1.5 across grids (CAPM on
  the CPZ panel), and usually by less than 0.9. The ordering of models is unchanged: factor models
  score 7–12, the additive glass-box SDF 5.0–5.3. Every SDF is rejected at 5% under every grid. The closest
  call is the linear SDF on OSAP (p = 0.048 in three grids, 0.006 in the other two).
* **The failure class is not robust,** as found in the state-variable check. The same reversal rule is
  labelled characteristic, state-dependent or mixed depending on the grid (whether Rel2High or r12_2
  enters, which state cut is available). The economic description should rest on the rule, not on the
  automatic class.

## Part B: knots of the additive SDF

The additive SDF is refit with 5 and 13 knots per shape function, against the baseline of 9, using
9 seeds each. Paper 2 is then repeated: shrinkage chosen on validation, identification at ε = 1% and by
block bootstrap (100 refits).

| knots | parameters (CPZ / OSAP) | test SR, CPZ | test SR, OSAP | CPZ audit: max \|t\| (top rule t) | OSAP audit: max \|t\| (top rule t) |
|---|---|---|---|---|---|
| 5 | 231 / 211 | 0.54 | 0.93 | 5.9 (reversal −3.5) | 3.7, p = 0.07 (none confirmed) |
| 9 | 415 / 379 | 0.64 | **0.96** | 5.2 (reversal −2.9) | 5.0 (none confirmed) |
| 13 | 599 / 547 | **0.67** | 0.88 | 5.2 (reversal −2.7) | 4.3 (none discovered) |

| knots | panel | ST_REV direction identified (ε = 1% / bootstrap) | value group identified (ε = 1% / bootstrap) | characteristics identified at ε = 1% |
|---|---|---|---|---|
| 5 | CPZ | ✓ / ✓ | ✓ / ✓ | 3 |
| 9 | CPZ | ✓ / ✓ | ✓ / ✓ | 3 |
| 13 | CPZ | ✓ / ✓ | ✓ / ✓ | 3 |
| 5 | OSAP | ✓ / ✓ | ✓ / ✓ | 4 |
| 9 | OSAP | ✓ / ✓ | ✓ / ✓ | 6 |
| 13 | OSAP | ✓ / ✓ | ✓ / ✓ | 4 |

* **Fit is not sensitive to the knot count.** The test Sharpe ratio moves within about ±0.1, with no
  consistent direction: it rises with knots on the CPZ panel and peaks at 9 on OSAP. Nine knots is a
  reasonable default and not a tuned choice.
* **The Paper 2 conclusions do not depend on knots.** On both panels and at every knot count:
  * the direction of short-term reversal is identified, at ε = 1% and by bootstrap;
  * the value group is identified;
  * only a handful of individual characteristics (3–6) are identified.
* **The audit conclusion does not depend on knots either.** On the CPZ panel the additive SDF's only
  confirmed failure stays the small reversal rule (t from −2.7 to −3.5). On OSAP the additive SDF has no
  confirmed failure at any knot count. With 5 knots it is not even rejected globally at 5%
  (p = 0.07); with 9 and 13 knots it is (p = 0.002 and 0.014).

## Reproducing

```bash
python scripts/run_knot_threshold_robustness.py --config config_glassbox/cpz.json --models results/paper1_cpz_neural --out results/knot_threshold_cpz
python scripts/run_knot_threshold_robustness.py --config config_glassbox/osap.json --models results/paper1_osap --out results/knot_threshold_osap
```

Runtime on one CPU container: 30 minutes for CPZ and 45 minutes for OSAP.
