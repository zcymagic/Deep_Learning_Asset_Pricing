# Robustness to rule thresholds, knots, the discovery window and the friction list

`scripts/run_knot_threshold_robustness.py` (Parts A and B) and `scripts/run_window_friction_robustness.py`
(Parts C and D) repeat the main checks on both panels: the CPZ panel ([cpz_results.md](cpz_results.md))
and the CRSP + OSAP panel, 1967–2016 bridge splits ([osap_results.md](osap_results.md)). Unless Part C
says otherwise, discovery uses 1967–1986 and confirmation uses 1987–2016. Tables are
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

## Part C: discovery window

`scripts/run_window_friction_robustness.py` holds every SDF fixed (estimated on the training split) and
moves only the audit's split of 1967–2016 into a discovery and a confirmation sample.

| window | discovery | confirmation |
|---|---|---|
| baseline | 1967–1986 | 1987–2016 |
| late | 1977–1986 | 1987–2016 |
| extended | 1967–1991 | 1992–2016 |
| within-holdout | 1987–2001 | 2002–2016 |

The within-holdout window uses no 1967–86 data at all, so the rules are found and confirmed entirely
inside the CPZ validation and test period.

**Does each SDF have a confirmed short-term-reversal failure in each window?** ✓ = a reversal rule is
confirmed (max-t p < 0.05), followed by the hold-out t of the top confirmed rule when that rule is a
reversal rule. When it is not, the top rule is named. ✗ = no confirmed reversal rule. The hold-out t
values are not comparable across windows, because the confirmation sample differs.

| SDF | panel | baseline | late | extended | within-holdout |
|---|---|---|---|---|---|
| CAPM | CPZ | ✓ −9.8 | ✓ −10.2 | ✓ −8.6 | ✓ −7.1 |
| CAPM | OSAP | ✓ −9.6 | ✓ −9.6 | ✓ −8.5 | ✓ (top rule ROA × SUV, +5.0) |
| FF3 | CPZ | ✓ −10.3 | ✓ −9.6 | ✓ −8.8 | ✓ −5.8 |
| FF3 | OSAP | ✓ −11.0 | ✓ −10.3 | ✓ −9.2 | ✓ −5.9 |
| FF5 + UMD | CPZ | ✓ −8.5 | ✓ −7.0 | ✓ −5.7 | ✓ (top rule SUV × NI, −4.7) |
| FF5 + UMD | OSAP | ✓ −8.3 | ✓ −8.3 | ✓ −7.1 | ✓ (top rule Investment × ROE, −5.2) |
| additive glass-box | CPZ | ✓ −2.9 | ✓ (top rule D2A × SUV, −3.9) | ✓ (top rule SUV × CF2P, +2.7) | ✗ (top rule SUV × D2P, −6.0) |
| additive glass-box | OSAP | ✗ | ✗ | ✗ | ✗ |
| additive + state | CPZ | ✓ −7.1 | ✓ −6.6 | ✓ −6.3 | ✗ (top rule r12_2 × SUV, −7.4) |
| additive + state | OSAP | ✗ | ✗ | ✗ | ✗ (top rule Investment × ROA, +5.1) |
| CPZ GAN | CPZ | ✗ | ✗ | ✗ | ✗ |

* **The factor models' reversal failure does not depend on the discovery window.** CAPM, FF3 and
  FF5 + UMD have a confirmed reversal failure in every window on both panels. It is the top confirmed
  failure in 21 of 24 cases.
  * It is confirmed even when discovery and confirmation both lie inside 1987–2016, so it is not an
    artefact of finding the rule in the high-reversal 1970s and early 1980s.
  * The hold-out t shrinks in the within-holdout window (−5.8 to −7.1 where reversal is the top rule).
    This matches the gradual decline of the reversal alpha in [osap_results.md](osap_results.md).
* **The reversal rule changes form across windows, but not its content.** The defining condition is
  always a recent winner (ST_REV > 0.1 or > 0.3). The second condition alternates between far below the
  52-week high (Rel2High < −0.1) and weak momentum (r12_2 < −0.1 or < −0.3). These are two descriptions
  of the same stocks: recent winners after a poor year.
* **The additive glass-box SDF on OSAP has no confirmed failure in any window.** The CPZ GAN has none
  in any window either. Both are still rejected by the global test in every window: max |t| from 4.8 to
  6.0, p ≤ 0.009.
* **On the CPZ panel the additive SDFs' failures depend on the window.** Reversal is their top failure in
  the baseline window. In the within-holdout window it is replaced by rules involving SUV (abnormal
  volume), at t = −6.0 and −7.4. These SDFs' small reversal failure is therefore not a stable finding;
  the factor models' is.
* **The linear SDF on OSAP is borderline.** It is rejected at 5% in the baseline, late and within-holdout
  windows (p = 0.048, 0.049, 0.001) but not in the extended window (p = 0.069), and it never has a
  confirmed failure.

## Part D: friction list

Confirmed failures from the baseline window are reclassified with four lists of limits-to-arbitrage
(friction) characteristics. In `audit.classify`:

* a rule is **limits-to-arbitrage** only if all its characteristics are on the friction list and its
  alpha does not depend on the state;
* it is **state-dependent** if it has a state condition or a significant state slope and no friction
  characteristic;
* it is **characteristic** if it has neither friction characteristics nor state dependence;
* everything else is **mixed**.

| list | characteristics |
|---|---|
| baseline | LME, Spread, IdioVol, Resid_Var, LTurnover, SUV, Variance, Rel2High, AT |
| without Rel2High | baseline minus Rel2High |
| narrow liquidity | LME, Spread, LTurnover |
| reversal as liquidity | baseline plus ST_REV and r2_1 (month t−1 and t−2 returns), treating reversal as compensation for liquidity provision |

**Class of the top confirmed reversal rule under each list**

| SDF | panel | rule | state slope t | baseline | without Rel2High | narrow liquidity | reversal as liquidity |
|---|---|---|---|---|---|---|---|
| CAPM | CPZ | ST_REV > 0.1, Rel2High < −0.1 | −1.4 (n.s.) | mixed | characteristic | characteristic | limits-to-arbitrage |
| FF3 | CPZ | ST_REV > 0.1, Rel2High < −0.1 | −2.5 (n.s.) | mixed | characteristic | characteristic | limits-to-arbitrage |
| FF5 + UMD | CPZ | ST_REV > 0.1, Rel2High < −0.1 | −3.5 | mixed | state-dependent | state-dependent | mixed |
| CAPM | OSAP | ST_REV > 0.3, Rel2High < −0.1 | −3.9 (svar) | mixed | state-dependent | state-dependent | mixed |
| FF3 | OSAP | ST_REV > 0.3, r12_2 < −0.3 | −3.1 (tbl) | state-dependent | state-dependent | state-dependent | mixed |
| FF5 + UMD | OSAP | ST_REV > 0.3, Rel2High < −0.1 | −4.1 (svar) | mixed | state-dependent | state-dependent | mixed |

The state slope is significant when it passes the Bonferroni bound over the state variables tried. The
largest slopes come from the CPZ dividend yield and from OSAP stock variance (svar) and the T-bill rate
(tbl).

* **The class is decided by the list, not by the data.** The same rule with the same hold-out t moves
  through all four classes as the list changes. Rel2High alone accounts for the baseline "mixed" label:
  dropping it gives "characteristic" or "state-dependent".
* **The reversal failure is "limits-to-arbitrage" only by assumption.** It gets that label only when
  short-term reversal is declared a friction characteristic, and then only for the two CPZ rules whose
  alpha does not vary with the state.
* **What is stable is the state dependence.** Under every list, the reversal alpha of FF5 + UMD on both
  panels, and of CAPM and FF3 on OSAP, loads significantly on a macro state. The slope is negative, so
  the failure is larger when market variance or short rates are high.
  * This fits a liquidity-provision reading, in which reversal profits rise when providing liquidity is
    riskier. It does not establish one.
* Together with Part A and the state-variable check in [cpz_results.md](cpz_results.md), this settles
  how the automatic class should be used. It is a description of how a rule is built and not evidence
  for risk versus mispricing. The paper should report the rule, its alpha and its state slope, and treat
  the class as a convenience label.

## Reproducing

```bash
python scripts/run_knot_threshold_robustness.py --config config_glassbox/cpz.json --models results/paper1_cpz_neural --out results/knot_threshold_cpz
python scripts/run_knot_threshold_robustness.py --config config_glassbox/osap.json --models results/paper1_osap --out results/knot_threshold_osap
python scripts/run_window_friction_robustness.py --config config_glassbox/cpz.json --models results/paper1_cpz_neural --out results/window_friction_cpz
python scripts/run_window_friction_robustness.py --config config_glassbox/osap.json --models results/paper1_osap --out results/window_friction_osap
```

Runtime on one CPU container: knots and thresholds take 30 minutes for CPZ and 45 minutes for OSAP;
window and friction take about 12 minutes per panel. Tables are in `robustness/{cpz,osap}_windows.csv`,
`robustness/{cpz,osap}_friction_rules.csv` and `robustness/{cpz,osap}_friction_shares.csv`.
