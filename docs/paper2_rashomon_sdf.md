# Paper 2 — The Rashomon Set of Stochastic Discount Factors: What Does the Data Identify, and Is SDF Ambiguity Priced?

## 1. Question and contribution

Many SDFs price the data almost equally well. They can still disagree about which characteristics
matter, in which direction weights move, and how much weight each stock gets. Empirical asset pricing
usually reports one point estimate per paper and hyperparameter choice. We characterize the whole set
of near-optimal SDFs and ask two questions.

1. **Identification:** which conclusions hold across *every* near-optimal SDF, and which depend on which
   model you pick? Examples: "value matters", "the momentum premium shrinks when credit spreads widen",
   "BEME rather than CF2P". This is a disciplined answer to the factor-zoo question: report what the
   data identifies rather than a single fit.
2. **SDF ambiguity as an economic variable:** stock-level disagreement across the set measures how much
   *model uncertainty* a sophisticated investor faces when pricing that stock. Ambiguity-aversion
   theory (Garlappi, Uppal & Wang RFS 2007; Epstein & Schneider) predicts that high-ambiguity stocks
   have less sophisticated ownership, slower correction of mispricing, and different expected returns.
   The monthly average is an aggregate **model-uncertainty index**.

Positioning:
* Rashomon sets and predictive multiplicity are established in ML (Breiman 2001; Fisher, Rudin &
  Dominici 2019; Marx, Calmon & Ustun 2020). They have been applied in credit risk but, as far as our
  search found, not to SDF estimation.
* Related finance work:
  * uncertainty of ML *return forecasts* (Liao, Ma, Neuhierl & Schilling 2025);
  * weak identification in factor models (Kan & Zhang 1999; Kleibergen 2009; Bryzgalova's spurious
    factors);
  * misspecification-robust inference (Gospodinov, Kan & Robotti).
  We work with SDF *weights* and their economic read-outs. *Verify these citations before submission.*

## 2. The key technical result: the Rashomon set is an exact ellipsoid

For the glass-box SDF of Paper 1, `M_t = 1 − Ψ_t'θ` is linear in θ. With the instruments h fixed, the CPZ
objective (unconditional plus conditional moments plus a quadratic smoothness penalty) is **exactly
quadratic**:

`J(θ) = J* + (θ − θ*)' A (θ − θ*)`, with `A = B'ΩB + P` and closed-form `θ* = A⁻¹ b`.

So the ε-Rashomon set `{θ : J(θ) ≤ (1 + ε) J*}` is an ellipsoid, and:
* any **linear** read-out has an exact range `g'θ* ± sqrt(ε J* · g'A⁻¹g)`. This covers a stock's SDF
  weight, a shape-function value, and the slope `f_k(0.5) − f_k(−0.5)`;
* any **quadratic** read-out, such as the variance a characteristic or a *group* of characteristics
  contributes to SDF weights, has exact bounds from a trust-region subproblem (`rashomon._trs_max`);
* **stock-month ambiguity** is the half-width of the weight range divided by that month's
  cross-sectional std of weights. **Sign ambiguity** means the range contains zero.

No sampling or retraining is needed: a full analysis on the CRSP panel takes minutes.
`tests/test_rashomon.py` checks the closed form against the torch objective, boundary samples at exactly
`(1 + ε)J*`, the trust-region solver against brute force, and sampled linear read-outs against the
analytic bounds.

Two complementary notions are reported:
* **Rashomon multiplicity:** a relative tolerance ε, reported on a grid (0.2%, 1%, 5%). How different can
  SDFs be that give up at most ε of pricing fit?
* **Statistical identification:** marginal stationary-block-bootstrap intervals from closed-form refits on
  resampled months, Bonferroni-adjusted across features. A bootstrap-calibrated *simultaneous* region
  (`eps=None`) is also available, but with hundreds of parameters it is very conservative
  (Scheffé-type), which is itself worth reporting.

## 3. Lessons from synthetic validation (worth stating in the paper)

* **Weights are a different object from risk premia.** In a one-factor (CAPM) world the equal-weighted
  market prices everything. Characteristics then move betas and expected returns, but *SDF weights* on
  characteristics are not identified. The tool reports exactly this. Identification needs priced
  characteristic risk that the market does not span. On real data, compare identification of weights
  with identification of the beta / risk-premium side (CPZ's second network,
  `src/model/model_RtnFcst.py`).
* **Collinear proxies.** With a priced value factor loading on BEME and a near-duplicate CF2P
  (correlation 0.95–0.99), the joint value contribution has a lower bound of 0.17, so it is identified.
  BEME and CF2P individually each have lower bound 0: either can carry all of it. This is the
  prototype "the data identifies the theme, not the variable" result to look for among the CPZ value,
  profitability and trading-friction clusters.
* **Mechanical components of ambiguity:**
  1. ambiguity is high where weights are extreme or nonlinear, so it can proxy for the model's own
     pricing signal. All ambiguity regressions therefore control for the reference SDF weight and its
     absolute value. Without these controls, synthetic data with no planted effect showed a spurious
     ambiguity "premium" (t = 3.9);
  2. state-varying shapes extrapolate in extreme states, so the monthly index co-moves with |s_t|.
     Report the index both raw and residualized on the state variables.

## 4. Empirical design on real data

* **Model:** the additive, state-varying glass-box SDF from Paper 1 (`lam_group = 0`, so the penalty is
  quadratic). Instruments come from the trained neural adversary. Robustness: alternative adversaries,
  knots, and state variables. Model-class multiplicity is the union over these.
* **Identification tables:**
  * importance ranges per characteristic and per economic group (CPZ categories: past returns,
    investment, profitability, intangibles, value, trading frictions);
  * slope-sign identification at low/mid/high states;
  * the Sharpe-ratio range across the set.
* **Ambiguity tests** (hold-out period):
  1. monthly quintile sorts on ambiguity: mean returns and alphas relative to the SDF;
  2. Fama-MacBeth regressions of returns on ambiguity, with controls for characteristics, the SDF
     weight and |weight|;
  3. regressions of *institutional ownership* (13F), *analyst dispersion* (IBES) and *short interest* on
     ambiguity. Add these through `Panel.extra` and the pipeline runs them automatically;
  4. persistence: do pricing errors (Paper 1 rule alphas) decay more slowly in high-ambiguity stocks?
  5. the aggregate index against VIX, economic policy uncertainty, macro uncertainty (Jurado-Ludvigson-Ng)
     and dispersion in professional forecasts, and as a predictor of SDF-portfolio volatility and
     drawdowns.
* **Hypotheses:**
  * H1: ownership by sophisticated investors falls with ambiguity;
  * H2: |alpha| and the persistence of mispricing rise with ambiguity;
  * H3: the aggregate index rises in crises and predicts lower SDF-portfolio Sharpe ratios.
* **Threats:**
  * ambiguity correlated with microcap status: run with and without microcaps, and with size and
    liquidity controls;
  * look-ahead: the ellipsoid is estimated on the training period only, and all tests use the hold-out.

## 5. How to run

```bash
python scripts/make_synthetic.py
python scripts/run_paper2.py --config config_glassbox/synthetic.json --out results/paper2_synthetic
python scripts/run_paper2.py --config config_glassbox/cpz.json --out results/paper2_cpz   # real data
```

Main API (`glassbox/rashomon.py`): `RashomonEllipsoid(model, batch, h, eps, n_boot)`, `.set_eps`,
`.importance_table(names, groups, method)`, `.slope_table`, `.shape_band`, `.weight_ambiguity`,
`.sharpe_range`, `.linear_range`, `.quadratic_range`; tests `ambiguity_sorts`, `ambiguity_regressions`,
`ambiguity_index_regressions`.

## 6. Next steps checklist

- [ ] Run on the CPZ panel; report the identification table for the six CPZ categories
- [ ] Merge 13F institutional ownership, IBES dispersion and short interest as `Panel.extra` arrays (T × N, same permno order)
- [ ] Residualize the ambiguity index on |state| and compare with JLN macro uncertainty
- [ ] Model-class multiplicity: repeat across knots / state sets / adversary seeds; report union ranges
- [ ] Compare identification of SDF weights with identification of betas (CPZ beta network)
