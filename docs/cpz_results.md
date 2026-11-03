# Results on the Chen-Pelger-Zhu data

First run of both research pipelines on the real CPZ panel. Design documents:
[Paper 1](paper1_glassbox_sdf_auditor.md), [Paper 2](paper2_rashomon_sdf.md). The tables and figures
behind every number here are in [`cpz_results/`](cpz_results/).

**Data.** CPZ panel from the repository's Google Drive: 46 rank-normalized firm characteristics and 178
macro series (FRED-MD plus Welch-Goyal and aggregate characteristic series).

| split | months | max stocks |
|---|---|---|
| train | 1967-01 to 1986-12 (240) | 3,686 |
| valid | 1987-01 to 1991-12 (60) | 3,347 |
| test | 1992-01 to 2016-12 (300) | 7,141 |

Benchmarks:
* the published CPZ GAN ensemble (`sample_checkpoints/sdf_normalized_ensemble.npy`, nine trials);
* CAPM, FF3 and FF5 + UMD from the Kenneth French library, aligned with
  `scripts/download_ff_factors.py`. Their SDF coefficients are estimated on the training split only.

**Settings.** `config_glassbox/cpz.json` with the following state variables:
* SDF state variables: BAA − fed funds spread (`BAAFFM`), 10-year − fed funds spread (`T10YFFM`), VXO
  (`VXOCLSx`);
* additional audit state variables: unemployment, CPI, industrial production, S&P dividend yield,
  stock variance (`svar`).

Training: nine seeds per ladder rung, CPZ epoch schedule.

All Sharpe ratios are monthly mean / std, the same units as the CPZ ensemble. The run used one CPU
container: about 40 minutes for the glass-box ladder plus audit, 8.5 hours for the nine-seed neural
rung, and about 40 minutes per Paper 2 model.

---

## Paper 1: glass-box SDF, price of interpretability, audit

### Complexity ladder

| SDF | parameters | train SR | valid SR | test SR |
|---|---|---|---|---|
| linear | 47 | 1.19 | 0.84 | 0.46 |
| linear + state | 188 | 1.26 | 0.83 | 0.31 |
| **additive** | 415 | 1.24 | 1.26 | **0.64** |
| additive + state | 1,660 | 1.54 | 1.21 | 0.51 |
| additive + state + screened pairs | 1,785 | 1.62 | 1.04 | 0.54 |
| neural rung (CPZ-style FFN, no LSTM) | 18,625 | 1.50 | 0.44 | 0.11 |
| **CPZ GAN ensemble (published)** | — | 3.02 | 1.39 | **0.77** |

* **Shape functions help; whether state variation helps depends on the state variables.** Moving from
  linear to additive raises the test SR from 0.46 to 0.64. With the baseline state variables (credit
  spread, term spread, VXO), letting shapes move with the state raises the train SR but lowers the test
  SR, even with the screened interactions (PM × ROE, D2P × Rel2High, Rel2High × Variance,
  Rel2High × Resid_Var, CTO × PM). The robustness check below shows this is **specific to the baseline
  set**: with Welch-Goyal, real-activity or valuation state variables, the state-varying additive SDF
  matches the static one.
* **My neural rung is not a usable benchmark.** It omits CPZ's LSTM state network, their dropout and
  their hyperparameter search, and it overfits. Measured against it, the "price of interpretability"
  is negative for every glass-box rung, which says nothing. The reference is the published CPZ GAN.
* **Weight drivers** of the richest glass-box SDF (share of weight variance): short-term reversal
  0.15, then D2P, PROF, AT and NOA at about 0.04–0.05 each, and a long flat tail.

### Spanning tests against the CPZ GAN (test period 1992–2016)

| glass-box SDF | SR² gap to GAN | GAN alpha on rung (t) | rung alpha on GAN (t) | correlation |
|---|---|---|---|---|
| linear | 0.38 | 7.7 | −2.2 | 0.75 |
| additive | **0.18** | 6.6 | 2.8 | 0.68 |
| additive + state | 0.32 | 9.2 | 5.3 | 0.22 |
| additive + state + pairs | 0.30 | 9.3 | 5.7 | 0.21 |

* The GAN earns significant alpha over every glass-box SDF. Its complexity buys pricing information
  that additive shapes with three state variables do not span.
* The reverse also holds: the glass-box SDFs, especially the state-varying ones, earn significant
  alpha over the GAN. Neither nests the other, so a combination would dominate both.
* So the price of interpretability is not a single number. The best glass-box SDF gives up a
  squared-Sharpe gap of about 0.18 per month, yet carries priced information the GAN misses.

### Audit (discovery: 1967–1986; confirmation: 1987–2016)

| SDF | largest hold-out \|t\| (fixed rule class) | global p | rules discovered |
|---|---|---|---|
| CAPM | 11.9 | < 0.001 | 2 |
| neural rung | 10.6 | < 0.001 | 1 |
| FF3 | 10.1 | < 0.001 | 2 |
| FF5 + UMD | 8.7 | < 0.001 | 2 |
| linear + state | 8.4 | < 0.001 | 1 |
| additive + state | 7.5 | < 0.001 | 2 |
| linear | 7.2 | < 0.001 | 4 |
| additive + state + pairs | 6.9 | 0.001 | 1 |
| CPZ GAN | 6.0 | 0.001 | 1 |
| **additive** | **5.2** | 0.002 | 1 |

* **Every SDF is rejected out of sample, but the ordering is informative.** The factor models fail
  worst. The plain additive glass-box SDF has the smallest worst-case pricing error, below the CPZ
  GAN's, even though its Sharpe ratio is lower.
* **The dominant failure is short-term reversal.** It is confirmed out of sample:

  | rule | SDFs | hold-out t | classification |
  |---|---|---|---|
  | recent winners (ST_REV > 0.1) far below their 52-week high (Rel2High < −0.1) | CAPM / FF3 / FF5 + UMD | −9.8 / −10.3 / −8.5 | — |
  | recent winners (ST_REV > 0.3) with weak 12-month momentum (r12_2 < −0.3) | additive + state, additive + state + pairs | −7.1, −6.6 | state-dependent (alpha loads on VXO, t ≈ −2.8) — **not robust**, see the state-variable robustness section |
  | — | linear + state | −6.7 | — |

  None of the factor sets contains a reversal factor. The same characteristic dominates the glass-box
  weights (above) and the Paper 2 identification results (below).
* **In-sample rules that did not replicate:**
  * the CPZ GAN's and the neural rung's discovered rules (RNA × AC: t = 6.3 in discovery, −1.8 in
    the hold-out);
  * the linear SDF's four rules.

  Their global rejections come from portfolios that the stepwise search did not keep. This is worth
  investigating.
* **No limits-to-arbitrage failures** appear under the default friction list.
* **Classification caveat:** the default friction list includes `Rel2High`. That puts the
  reversal × 52-week-high failure in "mixed" rather than "characteristic" or "state-dependent". Whether
  a 52-week-high variable counts as a limits-to-arbitrage measure must be decided explicitly in the
  paper.

---

## Paper 2: the Rashomon set of SDFs

The closed-form glass-box SDF carries an L2 shrinkage penalty chosen by validation Sharpe ratio.
Without it, the in-sample optimum of the 1,660-parameter model overfits: train SR 4.2, test SR about 0.
That first, invalid run is not reported.

The penalty selected is 0.01 for both models. For the additive SDF, validation SR rises from 0.58
unpenalized to 1.35. With the penalty in place, the closed-form SDFs match the GAN-trained glass-box
rungs out of sample.

| SDF | parameters | test SR (reference) | test SR range over the ε = 1% Rashomon set |
|---|---|---|---|
| additive | 415 | 0.65 | 0.58 – 0.66 |
| additive + state | 1,660 | 0.50 | 0.38 – 0.41 (raw SDF portfolio) |

### What the data identifies

An entry is "identified" when the lower bound of its importance (its share of SDF-weight variance)
exceeds 1% of the total, over the whole ε-Rashomon set or by bootstrap (100–500 block-bootstrap refits,
Bonferroni across features). For "direction", the sign of the shape-function slope is the same for
every member.

| feature | importance (additive / + state) | identified (additive) | identified (+ state) |
|---|---|---|---|
| **short-term reversal** | 0.16 / 0.17 | every ε and bootstrap; direction identified (negative weight on recent winners) | every ε and bootstrap; direction identified |
| **value (group)** | 0.12 / 0.22 | ε ≤ 1% and bootstrap | every ε and bootstrap |
| size (LME) | 0.06 / 0.04 | ε ≤ 1%; direction negative (ε ≤ 1%, bootstrap) | ε = 0.2% and bootstrap; direction negative (ε ≤ 1%, bootstrap) |
| D2P | — / 0.07 | — | ε ≤ 1% and bootstrap; direction not identified |
| other ~40 characteristics | < 0.05 each | mostly not identified at ε = 1% | mostly not identified at ε = 1% |

* **One characteristic is robustly identified: short-term reversal.** It holds under both models, at
  every tolerance and under the bootstrap.
* **Value is identified as a theme, never as a single variable.** The value group's lower bound is
  well above zero, but no individual value measure is identified. This is the collinear-proxy pattern
  the method was built to detect.
* **The rest of the factor zoo is not separately identified.** At ε = 1%, only 4 of 46 characteristics
  (plus the value group) pass for the additive SDF, and only 2 for the state-varying SDF.
  At ε = 5% only reversal survives.

### SDF ambiguity

* **Most stock-level weights are not identified.** At ε = 1%, the sign of a stock's SDF weight is
  not identified for 85% of test stock-months under the additive SDF, and 97% under the state-varying
  SDF.
* **Ambiguity is not priced in the hold-out.** Quintile sorts on ambiguity show H − L alphas with
  t = 0.48 (additive) and −0.72 (+ state). In Fama-MacBeth regressions with characteristic,
  squared-characteristic and SDF-weight controls, the ambiguity coefficient has t = 0.02 and 0.34.
  The ambiguity-premium hypothesis is not supported in this sample. The ownership and
  analyst-dispersion tests (13F, IBES) remain to be run.

---

## Combined glass-box + CPZ GAN SDF

`scripts/run_combined.py` builds M = 1 − b₁·F_glass − b₂·F_GAN. F_glass is the normalized SDF portfolio of
the 9-seed additive glass-box ensemble, F_GAN is the published CPZ ensemble, and b holds their tangency
weights.

The GAN was fit on the training period, where its Sharpe ratio is an in-sample 3.0, so weights
estimated there are not comparable to weights estimated out of sample. Two variants are reported:

| variant | weights estimated on | weight on glass-box / GAN | train SR | valid SR | test SR |
|---|---|---|---|---|---|
| combo_train | 1967–1986 | 0.23 / 0.77 | 3.15 | 1.46 | **0.78** |
| combo_valid | 1987–1991 (both models out of sample) | 0.41 / 0.59 | 3.00 | 1.47 | **0.78** |
| CPZ GAN alone | — | — | 3.02 | 1.39 | 0.77 |
| glass-box additive alone | — | — | 1.24 | 1.26 | 0.64 |

* **The combination spans both components but adds little Sharpe out of sample.** Neither component
  has significant test-period alpha on the combination (|t| ≤ 1.4). But the test Sharpe ratio rises
  only from 0.77 to 0.78. The two-way alphas found earlier show the components carry different priced
  information. With weights fixed before the test period, most of that gain does not carry over.
* **The audit improves slightly on the GAN, not on the glass-box SDF.** Discovery is on 1967–1986.
  Each SDF is compared with its components on the same confirmation sample:

  | SDF | confirmation 1987–2016: max \|t\| (p) | confirmation 1992–2016: max \|t\| (p) |
  |---|---|---|
  | combo_valid | — (weights use 1987–91) | **4.42 (0.008)** |
  | combo_train | 5.14 (0.003) | 4.78 (0.005) |
  | glass-box additive | 5.02 (0.003) | 4.82 (0.005) |
  | CPZ GAN | 5.97 (< 0.001) | 5.50 (< 0.001) |

  The combination is still rejected, but its worst-case pricing error is the smallest of any SDF
  audited so far.
* **The reversal failure disappears.** The glass-box SDF's confirmed failure (recent winners far
  below their 52-week high; hold-out t −2.1 to −2.4) is not among the combination's discovered rules.
  The GAN prices it.
* **None of the combination's discovered rules replicate** (hold-out |t| ≤ 2.5, max-t p ≥ 0.92). Its
  remaining rejection is diffuse: the stepwise search finds no specific readable portfolio to explain it.
* **Scaling note:** the components here enter as normalized SDF portfolios (Σ|w| = 1 each month). The
  glass-box audit numbers differ slightly from the ladder audit, which used the raw-weight SDF:
  5.02 vs 5.15.

## Robustness to the state variables

`scripts/run_state_robustness.py` repeats Paper 1 under five state-variable sets. The state-varying
rungs are refit with nine seeds, and every SDF is re-audited with the set as its audit state variables.
Supporting files: `cpz_results/state_robustness_*.csv`.

| set | variables |
|---|---|
| baseline | BAAFFM, T10YFFM, VXOCLSx |
| Welch-Goyal | dfy, tms, svar |
| real activity | UNRATE, INDPRO, CPIAUCSL |
| valuation | dp, tbl, ntis |
| VXO only | VXOCLSx |

The baseline set reproduces the original run exactly, so the refits are deterministic.

### State-varying SDFs (test Sharpe ratio, monthly; static additive = 0.64)

| state set | linear + state | additive + state |
|---|---|---|
| baseline | 0.31 | 0.51 |
| Welch-Goyal | 0.56 | 0.63 |
| real activity | 0.41 | 0.61 |
| valuation | 0.56 | **0.66** |
| VXO only | 0.31 | 0.47 |

* **Correction:** the earlier finding that state variation hurts out of sample holds only for the
  baseline set and for VXO alone. With Welch-Goyal, real-activity or valuation variables, the
  state-varying additive SDF matches the static one (0.61–0.66 against 0.64). The valuation set is
  slightly better, but it does not beat the static SDF by a meaningful margin.
* Implied volatility seems to be the problem: both sets containing VXO do worst.

### Audit across state sets

* **The reversal failure is robust.** In every set, the top confirmed failure of every SDF except the
  CPZ GAN is a short-term-reversal rule:
  * CAPM and FF5 + UMD: recent winners below their 52-week high, hold-out t = −9.8 and −8.5 in every
    set. The one exception is FF5 + UMD under the valuation set, where the top rule is recent winners
    with weak momentum, t = −7.0;
  * the static additive glass-box SDF: hold-out t = −2.9;
  * the state-varying SDFs: recent winners with weak momentum or below the 52-week high, hold-out
    |t| from 3.2 to 7.1. In one case (linear + state, real activity) the top rule is the mirror image,
    small recent losers with positive alpha (t = 5.1).
* **Its classification is not robust.** The "state-dependent, loads on VXO" label from the baseline run
  does not survive. The rule's alpha loads significantly on a different state variable in different
  sets (VXO, term spread, T-bill rate, credit spread), or on none, so it is classified state-dependent,
  characteristic or mixed depending on the set. The defensible statement is that the reversal failure
  is a **characteristic** failure whose time variation is not tied to any particular macro state.
* **No confirmed failure for the CPZ GAN in any set.** Its global rejection (max |t| = 5.97) is
  unchanged, but none of its discovered rules replicates out of sample (max-t p ≥ 0.99).
* **Global tests are unaffected for the state-free SDFs.** Their largest hold-out |t| comes from
  characteristic-only rules, so it is identical in every set.

## Caveats and next steps

* **Remaining robustness checks.** The state-variable set is done (above), and so are the knots and
  rule thresholds, the discovery window and the friction list ([robustness.md](robustness.md)). The factor
  models' reversal failure survives every grid, knot count and discovery window, including discovery and
  confirmation both inside 1987–2016. The Paper 2 identification of reversal and value survives every
  knot count. The automatic failure class does not survive changes to the friction list.
* **Neural benchmark.** Replace my neural rung with a faithful CPZ replication (LSTM state network,
  CPZ hyperparameters), or use the published GAN as the only complex benchmark.
* **Combination (done, see above).** Next: combination weights that vary with the macro state, and
  auditing the combination with rules beyond the beam.
* **Unreplicated global rejections.** Investigate why the CPZ GAN's global rejection does not survive
  as a confirmed discovered rule. Candidates: a smaller stepwise stopping threshold, or rules beyond
  the beam.
* **Data extensions.**
  * extend the panel past 2016 for a true out-of-sample period;
  * add 13F / IBES / short interest as `Panel.extra` for the Paper 2 ownership tests.

## Reproducing

```bash
python scripts/download_ff_factors.py --config config_glassbox/cpz.json
python scripts/run_paper1.py --config config_glassbox/cpz.json --out results/paper1_cpz --skip_neural
python scripts/run_paper1.py --config config_glassbox/cpz.json --out results/paper1_cpz_factors --reuse results/paper1_cpz   # adds the neural rung if not saved there
python scripts/run_paper2.py --config config_glassbox/cpz.json --out results/paper2_cpz_additive --rung additive
python scripts/run_paper2.py --config config_glassbox/cpz.json --out results/paper2_cpz_additive_state --rung additive_state
python scripts/run_combined.py --config config_glassbox/cpz.json --models results/paper1_cpz_neural --out results/combined_cpz
python scripts/run_state_robustness.py --config config_glassbox/cpz.json --models results/paper1_cpz_neural --out results/state_robustness_cpz
```

The additive + state Paper 2 run used `n_boot = 100` instead of the config's 500, to fit in a CPU
session. The spanning tests against the CPZ GAN came from a short script on the saved models. Their
neural-rung test SR (0.103) was computed before the eval-mode fix (`be7cafd`); the ladder value is 0.105.
