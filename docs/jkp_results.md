# Results on the Jensen–Kelly–Pedersen characteristics (153 characteristics and 13 themes)

This applies Paper 1 and Paper 2 to the most comprehensive characteristic set in the recent literature:
the 153 characteristics of Jensen, Kelly & Pedersen (2023, "Is There a Replication Crisis in Finance?",
JKP). It runs them in two forms:
* **all 153 characteristics** as separate inputs;
* **JKP's 13 themes**, each a composite of its member characteristics.

Earlier panels: [cpz_results.md](cpz_results.md) (46 CPZ characteristics) and [osap_results.md](osap_results.md)
(42 characteristics rebuilt from CRSP and OSAP). Tables: [`jkp_results/`](jkp_results/).

## Data

`scripts/build_jkp_panel.py` builds the panel from JKP's US stock-month file.

**Characteristics.** The 153 characteristics, lagged one month as in CPZ, and grouped into JKP's 13 themes
(`glassbox/jkp_themes.py`). The theme assignments come from the authors' cluster file.

**Universe.**
* CRSP stocks only. JKP's file also holds about 2,000 Compustat-only securities (2.2% of rows), whose
  returns are unreliable: monthly sd about 60, against 0.19 for CRSP stocks.
* A stock-month needs at least two-thirds of the 153 characteristics observed.
* Each month, characteristics are re-ranked to [−0.5, 0.5]; missing values get the median (0).

**Splits.** As the OSAP panel, except that JKP's file runs only from 1970 to 2022.

| split | months | stocks per month (median) |
|---|---|---|
| train | 1970–1986 | 3,143 |
| valid | 1987–1991 | 4,559 |
| test | 1992–2016 | 4,905 |
| post | 2017–2022 | 3,903 |

**Theme panel.** `scripts/build_jkp_theme_panel.py` collapses the 153 characteristics to 13 theme scores.
* Each characteristic is first signed by JKP's direction, so that a high value is the side the original
  study found to earn more.
* A stock's score for a theme is the mean of its observed members' signed ranks, re-ranked each month.
* The theme panel therefore holds exactly the same stock-months and returns as the 153-characteristic
  panel.

**Settings.**
* Configs: `config_glassbox/jkp*.json`. Same training schedule as before, with 9 seeds per rung.
* State variables: dfy, tms, svar. Audit state variables: the same eight as on the OSAP panel.
* **State clipping.** On these panels the state variables are clipped at ±3 training standard deviations.
  Without clipping, svar reaches 50 sd in October 1987, which falls in the validation split. It multiplies
  the state-varying weights by the same factor and dominates the validation Sharpe ratio used to pick the
  stopping point.
  * Unclipped, 6 of 9 seeds of the state-varying additive SDF on 153 characteristics ended near zero.
    The ensemble's Sharpe ratio was 0.19 even in training (`ladder_jkp153_unclipped_states.csv`).
  * Clipped, it trains normally.
  * The static rungs do not use the states and are unaffected.

## Paper 1: complexity ladder

Sharpe ratios (monthly mean/sd of the SDF portfolio, 9-seed ensembles).

| SDF | 153 characteristics: parameters | train | valid | test 1992–2016 | post 2017–2022 | 13 themes: parameters | train | valid | test 1992–2016 | post 2017–2022 |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 154 | 2.71 | 1.94 | 0.66 | 0.09 | 14 | 1.50 | 1.70 | 0.61 | 0.19 |
| linear + state | 616 | 2.31 | 1.54 | 0.35 | 0.15 | 56 | 1.38 | 1.32 | 0.47 | 0.21 |
| **additive** | 1,378 | 3.03 | 1.68 | 0.39 | 0.01 | **118** | 1.77 | 1.85 | **0.78** | **0.24** |
| additive + state | 5,512 | 3.25 | 1.54 | 0.19 | 0.03 | 472 | 2.27 | 1.32 | 0.35 | 0.08 |
| additive + state + pairs | 5,637 | 2.29 | 1.60 | 0.19 | 0.01 | 597 | 2.87 | 1.37 | 0.44 | 0.17 |

The additive SDF across panels (test Sharpe ratio):

| panel | characteristics | test period | additive SDF | best glass-box SDF |
|---|---|---|---|---|
| CPZ | 46 | 1992–2016 | 0.64 | 0.64 (additive) |
| OSAP | 42 | 1992–2016 | 0.96 | 0.96 (additive) |
| JKP | 153 | 1992–2016 | 0.39 | 0.66 (linear) |
| **JKP themes** | **13** | 1992–2016 | **0.78** | **0.78 (additive)** |

The universes differ, so compare orderings rather than levels. The CPZ panel is complete-case; the OSAP
and JKP panels are CRSP stocks with two-thirds coverage and median imputation, and are closest to each
other.

* **More characteristics do not give a better SDF out of sample.**
  * With 153 characteristics every rung overfits. Training Sharpe ratios are 2.3–3.3, but test Sharpe
    ratios are 0.19–0.66.
  * The additive SDF, which was the best glass-box rung on both earlier panels, falls to 0.39, below the
    linear SDF.
  * Validation-chosen shrinkage does not rescue it (Paper 2 below: test Sharpe ratio 0.38).
* **JKP's own theme structure recovers the signal.** Collapsing the 153 characteristics to 13 signed
  theme composites gives the best JKP glass-box SDF: test Sharpe ratio 0.78 with 118 parameters, against
  0.39 with 1,378. Its training, validation and test Sharpe ratios are close (1.77, 1.85, 0.78).
  * Most of what the extra characteristics add is noise around a low-dimensional structure, which is
    JKP's own conclusion about the factor zoo.
  * The interpretable model is also the better one here, so the "price of interpretability" is negative.
* **State-varying weights hurt out of sample on every panel**, as on the CPZ and OSAP panels: test
  Sharpe ratio 0.35 against 0.78 for the themes, and 0.19 against 0.39 for the 153 characteristics.
* **After 2016 every SDF loses most of its Sharpe ratio**, as on the OSAP panel. The theme additive SDF
  remains the best glass-box SDF (0.24), and the 153-characteristic additive SDF falls to zero (0.01).

## Paper 1: audit

Discovery on 1970–1986. Confirmation and the global test on 1987–2016, or on 2017–2022 for the
post-sample columns. Global test: largest hold-out |t| over the fixed rule class (p-value).

| SDF | 153 characteristics, 1987–2016 | 153 characteristics, 2017–2022 | themes, 1987–2016 | themes, 2017–2022 |
|---|---|---|---|---|
| linear | 8.8 (< 0.001) | 4.2 (0.010) | 6.0 (< 0.001) | 3.3 (0.044) |
| additive | 7.6 (< 0.001) | 4.3 (0.004) | **3.5 (0.073)** | 3.7 (0.012) |
| additive + state | 8.7 (< 0.001) | 4.7 (0.002) | 8.5 (< 0.001) | 3.5 (0.030) |
| CAPM | 11.6 (< 0.001) | 4.0 (0.020) | 8.9 (< 0.001) | 3.3 (0.048) |
| FF3 | 11.2 (< 0.001) | 5.6 (< 0.001) | 8.6 (< 0.001) | 3.9 (0.007) |
| FF5 + UMD | 9.6 (< 0.001) | 5.0 (0.001) | 7.3 (< 0.001) | 3.9 (0.008) |

**Confirmed failures of the factor models** (hold-out t, max-t p < 0.05):

| rule | CAPM | FF3 | FF5 + UMD |
|---|---|---|---|
| 153 characteristics: recent winners far below their 52-week high (`ret_1_0` > 0.3, `prc_highprc_252d` < −0.1) | −11.1 | −12.0 | −9.5 |
| same rule, 2017–2022 | −1.6 (not confirmed) | −2.9 | −2.3 |
| 153 characteristics: cheap and profitable (`ebitda_mev` > 0.3, `qmj_prof` > 0.3) | +4.5 | +5.0 | +2.9 |
| themes: recent winners with weak momentum (Short-Term Reversal < −0.3, Momentum < −0.1) | −10.0 | −11.4 | −9.4 |
| same rule, 2017–2022 | −1.5 (not confirmed) | −3.0 | −2.1 |
| themes: cheap with improving profits (Value > 0.3, Profit Growth > 0.1) | +6.6 | +8.9 | — |
| themes: cheap recent losers (Value > 0.3, Short-Term Reversal > 0.3) | — | +6.9 | +4.1 |

The themes are signed by JKP's direction, so a low Short-Term Reversal score is a recent winner and a
high Value score is a cheap stock.

* **Short-term reversal is the factor models' dominant failure on the richest characteristic set too.**
  * The rule found is the same as on the CPZ and OSAP panels: recent winners that are far below their
    52-week high, or have weak 12-month momentum. It is the top failure of all three factor models.
  * It is found in both the 153-characteristic and the theme rule spaces, with t from −9.4 to −12.0.
  * After 2016 it remains confirmed for FF3 and FF5 + UMD (t ≈ −2 to −3), as on the OSAP panel.
  * Adding momentum, seasonality, skewness and liquidity characteristics does not absorb it.
* **New on this panel: value combined with profitability.** CAPM and FF3 underprice cheap firms that are
  profitable or improving (t from +4.5 to +8.9). FF5 + UMD, which contains a profitability factor, still
  misses cheap profitable firms on the 153-characteristic panel (t = +2.9) and cheap recent losers in
  theme space (t = +4.1). After 2016 most of these are not confirmed. The exception is FF3 in theme space,
  which still misses cheap recent losers (t = +3.7) and cheap firms with improving profits (t = +2.3).
* **The theme additive SDF is the one SDF not rejected on 1987–2016** (p = 0.073), and the audit discovers
  no candidate failure for it. Two checks qualify this:
  * Against the richer 153-characteristic rule space it is rejected (largest |t| 5.0, p = 0.006). That is
    weaker than for the 153-characteristic additive SDF itself (7.6) or any factor model (9.6–11.6), and
    no failure is confirmed (`scripts/audit_cross_panel.py`, `audit_summary_themes_on_jkp153_rules.csv`).
  * The state-varying theme SDF, by contrast, fails there on the same reversal rule (t = −6.8).
* After 2016 every SDF is rejected at 5% but with much smaller |t| (3.3–5.6), consistent with weaker
  anomaly returns in the recent sample.
* **Glass-box failures on 153 characteristics.**
  * The linear SDF has one confirmed failure: illiquid recent winners (`bidaskhl_21d` > 0.3,
    `ret_1_0` > 0.3, t = −5.2). It is not confirmed after 2016.
  * The additive and state-varying additive SDFs have no confirmed failure.
  * The pairs rung has one small confirmed rule (t = −2.6).

## Paper 2: identification

The additive SDF is refit in closed form, with L2 shrinkage chosen on validation Sharpe ratio. Importance
is the share of SDF-weight variance from each characteristic or theme. "Identified" means it is bounded
away from zero over the Rashomon set of near-optimal SDFs at ε (here 1%), or by the bootstrap.

**153 characteristics** (shrinkage 0.001, test Sharpe ratio 0.38; 0.36–0.41 over the ε = 1% set).
* **No single characteristic's importance is identified.** The direction of the SDF weight is identified
  for two: short-term reversal (`ret_1_0`, negative) and Dimson beta.
* **Themes as groups.** Nine are identified:

  | theme | importance |
  |---|---|
  | Low Risk | 0.29 |
  | Value | 0.20 |
  | Investment | 0.08 |
  | Low Leverage | 0.07 |
  | Short-Term Reversal | 0.07 |
  | Quality | 0.07 |
  | Profit Growth | 0.06 |
  | Profitability | 0.05 |
  | Momentum | 0.03 |

  Size, Debt Issuance, Seasonality and Accruals are not identified.

**13 themes** (shrinkage 0.01, test Sharpe ratio 0.71; raw 0.65, and 0.59–0.72 raw over the ε = 1% set).

| theme | importance | identified (ε = 0.2% / 1% / 5% / bootstrap) | sign of slope identified at ε = 1% |
|---|---|---|---|
| **Short-Term Reversal** | **0.22** | ✓ / ✓ / ✓ / ✓ | ✓ (positive: long recent losers) |
| Value | 0.12 | ✓ / ✓ / ✗ / ✓ | ✓ (positive: long cheap stocks) |
| Profitability | 0.08 | ✓ / ✓ / ✗ / ✓ | ✗ |
| Low Leverage | 0.07 | ✓ / ✓ / ✗ / ✗ | ✓ (positive) |
| Low Risk | 0.06 | ✓ / ✓ / ✗ / ✓ | ✓ (positive) |
| Profit Growth | 0.04 | ✓ / ✗ / ✗ / ✗ | ✗ |
| Size | 0.04 | ✓ / ✗ / ✗ / ✓ | ✓ (positive: long small stocks) |
| Investment, Quality, Debt Issuance, Seasonality, Accruals, Momentum | ≤ 0.03 | ✗ at ε = 1% | ✗ |

* **Short-term reversal is the best-identified component of the SDF on every panel.** On the theme panel
  it carries the largest share of SDF-weight variance (22%). It is the only theme still identified at
  ε = 5%, and its sign is identified everywhere. On the CPZ and OSAP panels it was the characteristic
  whose direction was identified everywhere.
* **Where identified, signs agree with JKP's directions.** The SDF goes long the side of each theme that
  the original studies found to earn more.
* **Most premia sit in one tail** (`jkp_results/themes_shape_bands.png`).
  * The Value, Low Leverage, Size and Momentum shape functions rise only in the top 20–30% of the theme
    score: the cheapest, least levered, smallest and strongest-momentum stocks.
  * Low Risk and Profitability mainly penalize the riskiest and least profitable stocks.
  * Short-Term Reversal is the steepest shape and is monotone over the whole range.
* **Stock-level weights are still not identified.** The weight sign is ambiguous for 78% of test
  stock-months with themes, and 90% with 153 characteristics.
* **Ambiguity is not priced under the theme SDF.**
  * Fama–MacBeth: t = 1.1, with characteristic, squared-characteristic and SDF-weight controls.
  * High-minus-low ambiguity sort: alpha t = −0.9.
  * Under the 153-characteristic SDF ambiguity appears priced (Fama–MacBeth t = 3.3, sort alpha t = 3.0,
    raw spread t = 0.9). Since that reference SDF prices poorly out of sample, this is most likely an
    artefact of the poor reference model. The CPZ and OSAP panels also found no pricing of ambiguity.

## Summary

1. **Breadth without structure overfits.** Moving from 42–46 characteristics to JKP's 153 lowers the
   glass-box SDF's test Sharpe ratio from 0.64–0.96 to 0.39–0.66. Imposing JKP's 13-theme structure
   restores it to 0.78 with 118 parameters.
2. **Short-term reversal survives the richest characteristic set.** It is the factor models' dominant
   confirmed failure (t ≈ −10 to −12), with the same rule as on the CPZ and OSAP panels, and it remains
   confirmed for FF3 and FF5 + UMD after 2016. It is also the most important and best-identified
   component of the glass-box SDF.
3. **New failure: value combined with profitability or profit growth.** CAPM and FF3 miss it, and FF5 +
   UMD partly misses it, before 2017. After 2016 only FF3's version in theme space is still confirmed.
4. **State-varying weights hurt out of sample on all four panels.**
5. **Identification is at the level of themes, not characteristics.** No single characteristic's
   importance is identified among the 153, but nine themes are. With themes as inputs, five themes are
   identified at ε = 1%, and the direction of five.

## Reproducing

```bash
tar -xjOf kelly.tar.bz2 kelly.csv | python scripts/build_jkp_panel.py --out datasets/jkp_panel --cache jkp_raw.npz
python scripts/build_jkp_theme_panel.py --src datasets/jkp_panel --out datasets/jkp_themes
python scripts/run_paper1.py --config config_glassbox/jkp.json --out results/paper1_jkp --skip_neural
python scripts/run_paper1.py --config config_glassbox/jkp_post.json --out results/paper1_jkp_post --skip_neural --reuse results/paper1_jkp --holdout test
python scripts/run_paper2.py --config config_glassbox/jkp.json --out results/paper2_jkp_additive --rung additive
python scripts/run_paper1.py --config config_glassbox/jkp_themes.json --out results/paper1_jkp_themes --skip_neural
python scripts/run_paper1.py --config config_glassbox/jkp_themes_post.json --out results/paper1_jkp_themes_post --skip_neural --reuse results/paper1_jkp_themes --holdout test
python scripts/run_paper2.py --config config_glassbox/jkp_themes.json --out results/paper2_jkp_themes_additive --rung additive
python scripts/audit_cross_panel.py --models results/paper1_jkp_themes --model_config config_glassbox/jkp_themes.json --audit_config config_glassbox/jkp.json --out results/cross_audit_themes_on_jkp153
```

Runtime on one 4-CPU container, with two runs sharing it:
* Paper 1: about 95 minutes for 153 characteristics alone, and about 3–4 hours each with the theme run in
  parallel.
* Paper 2: 16 minutes (153 characteristics).
* Post-sample runs: about 15–45 minutes each.

JKP's data are licensed through WRDS; no data files are committed.
