# Paper 1 — A Glass-Box SDF and a Glass-Box Auditor: Where Do Asset Pricing Models Fail, and Is It Risk or Mispricing?

Working title alternatives: *"Auditing the Stochastic Discount Factor"*, *"The Price of Interpretability in Asset Pricing"*.

## 1. Question and contribution

Deep SDFs (Chen, Pelger & Zhu, MS 2024; Kelly, Kuznetsov, Malamud & Xu, NBER 2025) price the
cross-section well but are black boxes. Post-hoc explanations (gradients, SHAP) only approximate them.
We keep the economically meaningful part of CPZ, the **conditional moment game**
`min_SDF max_h E[M R h]^2`, and make the two players readable.

1. **Glass-box SDF (part A).** Portfolio weights are a sum of exact shape functions of each characteristic.
   Their scale moves linearly with a few *named* macro state variables, plus a short, screened list of
   pairwise interactions:

   `w_ti = c(s_t) + Σ_k f_k(z_tik; s_t) + Σ_{(j,k)∈S} f_jk(z_tij, z_tik)`,  `f_k(z; s) = Σ_m (θ_km + Γ_km' s_t) B_m(z)`.

   The SDF is linear in its parameters, so `M_t = 1 − Ψ_t'θ` with a small design matrix `Ψ_t = Σ_i R_ti φ(z_ti, s_t)`.
   Every figure is the model itself, not an approximation of it.
2. **Price of interpretability.** A nested ladder (linear → linear×state → additive → additive×state →
   + screened pairs → CPZ-style neural net) estimated on the same data against the same adversary.
   At each step we measure out-of-sample Sharpe ratio, EV and XS-R², and run a spanning test of the
   most complex SDF on each simpler one. The headline number is `SR²(neural) − SR²(rung)`.
   *Hypothesis A1:* the additive, state-varying SDF recovers most of the deep SDF's out-of-sample
   Sharpe ratio. If so, the nonlinearities in the factor zoo are mostly **separable**, and macro
   conditions act as a few **regime-dependent prices of characteristics**. If not, the ladder shows
   exactly which interactions are needed and when.
3. **Glass-box auditor (part B).** A specification test that works for *any* SDF M: CAPM, FF5, q5, IPCA,
   the CPZ GAN, or our own. Test assets are readable rules: at most two characteristic conditions
   (which stocks) and at most one macro-state condition (when the portfolio is held), for example
   *"LME < −0.3 AND Spread > 0.3"* or *"r12_2 > 0.3 AND VIX > 0.8"*. The steps are:
   * *discovery* on the training period: beam search, parsimony penalty, and forward-stepwise selection,
     so each new rule is a failure not explained by earlier ones;
   * *confirmation* on the hold-out period: HAC t-statistics, Holm, and stationary-bootstrap max-t p-values;
   * a *global test* over a fixed rule class: is the SDF rejected at all?
   * *classification*:

     | class | definition | interpretation |
     |---|---|---|
     | limits-to-arbitrage | only trading-friction characteristics, no state dependence | mispricing-type |
     | state-dependent | a state condition, or a hold-out alpha that loads on a macro state, and no friction characteristic | missing risk / time-varying price of risk |
     | characteristic | an unconditional premium on non-friction characteristics | missing characteristic factor |
     | mixed | everything else | — |

   *Hypothesis B1 (mispricing view):* the out-of-sample failures of the leading models are concentrated
   in the limits-to-arbitrage region (small, illiquid, high-idiosyncratic-volatility stocks).
   *Hypothesis B2 (risk view):* they are concentrated in macro states (credit spread, VIX, funding
   conditions) and cut across stock types.
   The share of squared hold-out t-statistics in each class is the "failure map" of each model.
   It is a direct, quantitative way to compare these two classic explanations, model by model.
4. **Constructive step.** `augment_sdf` adds a discovered portfolio to the SDF and re-audits.
   If a state-dependent failure disappears out of sample after the addition, it is a candidate missing
   priced state variable. If a limits-to-arbitrage failure survives every augmentation, that is
   consistent with mispricing.

Positioning:
* Bryzgalova, Pelger & Zhu (JF 2025, *Forest Through the Trees*) and Cong, Feng, He & He (JFE 2025,
  *P-Trees*) build tree test assets to *span the efficient frontier*. We build rule test assets to
  *locate the failures of a given model* and classify them.
* Hansen & Jagannathan (1997): our statistic is a maximal pricing error over an interpretable function class.
* Barillas & Shanken: the out-of-sample alpha logic.
* Romano–Wolf and White's reality check: inference over the searched class.

## 2. Methods: implementation notes (`glassbox/`)

| piece | file | key facts |
|---|---|---|
| data | `data.py` | reads the repo's npz files unchanged; `concat_time` for valid+test hold-out |
| SDF models | `sdf.py` | `LinearSDF`, `AdditiveSDF` (linear in parameters: `sdf()` uses the cached design; `forward()` gives exact stock weights), `NeuralSDF` (CPZ-style FFN), `FixedWeightsSDF` |
| adversaries | `adversary.py` | `NeuralAdversary` (CPZ moment net), `SoftRuleAdversary` (differentiable conjunction rules, `hard_rules()` read-out) |
| training | `train.py` | CPZ phases (unconditional → adversary → conditional); selection on validation Sharpe of 1 − M |
| ladder | `ladder.py` | `run_ladder`, `screen_pairs` (interactions the additive SDF leaves unpriced), spanning tests |
| auditor | `audit.py` | `discover`, `confirm`, `global_test`, `state_dependence`, `classify`, `failure_map`, `augment_sdf` |
| read-outs | `interpret.py` | exact shape tables and plots, variance shares, pair surfaces |

Econometric choices, each motivated by a failure found in synthetic validation:
* **Pricing error = alpha on the SDF portfolio** `F = 1 − M`, with the equal-weighted market as a
  control. If M prices F, then `E[Mx] = 0 ⇔ α = 0`. The raw moment `E[Mx]` is far noisier; the hold-out
  t-statistic of the planted mispricing went from 4.5 to 10–16. The market control removes factor
  exposure that a time-varying SDF portfolio leaves unspanned under a constant-beta regression.
* **Influence-function standard errors** `α̂ + T a_t e_t` (full HAC sandwich, including slope
  estimation error). Without them, stepwise search with a high-Sharpe added portfolio produced
  spurious |t| ≈ 7–15.
* **Stopping rule** `stop_t='auto'`: the Bonferroni critical value for the size of the searched class
  (about 10⁵ rules gives about 4.7). This is the level the largest |t| reaches under pure noise.
* **State-dependence test:** a state-dependent failure often shows up first as an unconditional
  characteristic rule, because the conditional premium averages into an unconditional one. Every
  confirmed rule's hold-out alpha series is therefore regressed on each named state variable, and
  the classification uses that test.

## 3. Synthetic validation (ground truth known; `glassbox/synthetic.py`, `tests/`)

The DGP has:
* a priced market factor with a time-varying price of risk (credit spread);
* a priced value factor;
* additive, nonlinear and state-dependent exposures, plus one interaction;
* a near-duplicate value proxy (CF2P);
* **planted mispricing** in the LME-low × Spread-high region;
* a **planted missing risk factor**: momentum is priced only in high-VIX months.

Auditing the oracle SDF (which prices both factors) gives:
* the mispricing region is found and confirmed out of sample (hold-out t ≈ 9–16, max-t p < 0.001) and
  classified *limits-to-arbitrage* on every seed;
* the momentum failure is found; its hold-out alpha loads on **VIX** (t ≈ 4–12) and not on the other
  five macro series, so it is classified *state-dependent*;
* under the null DGP the global test has size: p = 0.46 and 0.84 on two seeds, and nothing confirms
  out of sample;
* **power caveat for the paper:** a failure that occurs in about 20% of months, with about 30 such
  months in the hold-out, sits near the multiple-testing bar. Report power curves.

Full pipeline on the synthetic panel (snapshot in `docs/synthetic_validation/paper1_REPORT.md`; train 300 /
valid 100 / test 200 months, N = 1000):
* **Ladder.** Test Sharpe ratios (monthly) are 0.76 linear, 0.77 additive, 0.80 additive + state + pairs,
  and only 0.43 for the neural SDF. The neural net overfits: train 0.80, valid 0.27. Its portfolio has no
  significant alpha on any glass-box rung (|t| < 0.8), so in this economy the price of interpretability
  is *negative*. Pair screening picked the LME × Spread interaction, which is the planted mispricing
  region. On real data the size of this gap is the headline number.
* **Audit.** The oracle SDF and the neural SDF both fail only on the limits-to-arbitrage region (hold-out
  t = 9–11). The trained glass-box SDFs load on that region, so they price it and no rule is
  confirmed against them. The linear SDF, which cannot express the region, fails on it and on a
  state-dependent momentum rule.
* **Caveats to address in the paper:**
  1. the momentum × VIX failure is missed with a 300-month discovery window. That is a power issue,
     so report power curves and a pooled-sample discovery as robustness;
  2. the state-dependence test is run for each rule at a Bonferroni bound over the state variables,
     and one of 4 non-state rules crossed it (ip_growth, t = 2.8). Use a stricter bound, or a
     multiple-testing correction across rules × states, when classifying.

## 4. Empirical design on real data

* **Data:** CPZ panel (46 characteristics, 178 macro series, 1967–2016) from the repo's Google Drive.
  The paper should extend it to 2024 with CRSP/Compustat (WRDS) using the same construction.
* **Samples:** discovery = CPZ training period. Confirmation = validation + test. Robustness: a rolling
  discovery window, and post-2016 data as a true out-of-sample test.
* **SDFs audited:** CAPM, FF3, FF5, FF5+momentum, q5 (Hou-Xue-Zhang), IPCA, Kozak-Nagel-Santosh
  shrinkage SDF, CPZ GAN (published checkpoints: `sdf_normalized_ensemble.npy`), and every rung of
  our ladder. Factor SDFs are `M = 1 − b'F` with b estimated in the discovery period (`pricing.factor_sdf`).
* **State variables** (named, few, theory-motivated): credit spread (BAA − FF), term spread, VIX/VXO,
  NFCI or intermediary capital (He-Kelly-Manela), unemployment, inflation.
* **Tables:**
  1. the ladder;
  2. glass-box variance shares and state loadings;
  3. audit summary by model (global test, number of confirmed failures, failure-map shares);
  4. the confirmed rules for 3–4 key models;
  5. augmentation results;
  6. robustness (equal vs value weights inside rules, thresholds, min_stocks, NYSE breakpoints,
     excluding microcaps).
* **Figures:**
  1. shape functions at low/mid/high credit spread;
  2. pair surfaces;
  3. failure maps (stacked bars by model);
  4. the time series of confirmed state-dependent rule alphas.
* **Referee-proofing:**
  * microcaps: show failure maps with and without them;
  * transaction costs: rules in the limits-to-arbitrage class should not be exploitable after costs
    if they reflect mispricing that limits to arbitrage protect; report net alphas using the
    `Spread` characteristic as the cost proxy;
  * multiple testing: max-t and Holm are already in place, and Harvey-Liu-Zhu-style thresholds can be
    reported as well;
  * data mining: every reported rule is out of sample, and the rule class is fixed in advance in the
    pre-analysis config.

## 5. How to run

```bash
pip install -r requirements_glassbox.txt
python scripts/make_synthetic.py                                   # validation data
python scripts/run_paper1.py --config config_glassbox/synthetic.json --out results/paper1_synthetic
# real data: put the CPZ npz files under datasets/ (see README), then
python scripts/run_paper1.py --config config_glassbox/cpz.json --list_macro   # choose state_vars
python scripts/run_paper1.py --config config_glassbox/cpz.json --out results/paper1_cpz
python -m pytest tests
```

Compute: the glass-box rungs train in minutes on the full CRSP panel, because training only touches the
T × D design. The neural rung and the 9-seed ensembles want a GPU. Discovery is about one minute per
SDF on the full panel.

## 6. Next steps checklist

- [ ] Download CPZ data; run `--list_macro`; set `state_vars` / `audit_state_vars` in `config_glassbox/cpz.json`
- [ ] Export factor returns (FF5 + UMD, q5) to `datasets/factors/*.csv` with a `date` column matching the npz dates (loader: `factor_files` in `config_glassbox/cpz.json`)
- [ ] Run the ladder with `n_seeds: 9` (CPZ ensemble convention)
- [ ] Run the audit for all SDFs; write the failure-map figure
- [ ] Extend the data to 2024 for a true out-of-sample test
- [ ] Power curves on synthetic data with realistic effect sizes calibrated to the real panel
