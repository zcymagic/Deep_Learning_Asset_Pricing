# Replication package: glass-box SDF, rule auditor and Rashomon identification

This package reruns every result in:
* [`docs/cpz_results.md`](../docs/cpz_results.md): the CPZ panel;
* [`docs/osap_results.md`](../docs/osap_results.md): CRSP + Open Source Asset Pricing, 1967–2024;
* [`docs/jkp_results.md`](../docs/jkp_results.md): the 153 Jensen–Kelly–Pedersen characteristics and their 13 themes;
* [`docs/robustness.md`](../docs/robustness.md): knots, rule thresholds, discovery window and friction list;
* [`docs/synthetic_validation/`](../docs/synthetic_validation/): planted ground truth.

It then compares the new tables with the committed ones. Code, configs and the small public factor files
are in the repository. **The licensed and third-party data are not included;** see [Data](#data).

```bash
pip install -r replication/requirements.txt
python replication/check_inputs.py          # which stages have their inputs
replication/run_all.sh                      # every stage, in order (about 1.5–2 CPU days)
replication/run_all.sh tests synthetic      # or selected stages
```

## Contents

| file | purpose |
|---|---|
| `run_all.sh` | stage driver; one log per step in `replication_logs/` |
| `check_inputs.py` | lists each stage's input files and whether they are present |
| `collect_and_compare.py` | copies results into the `docs/` layout under `replication_output/` and compares every CSV with the committed copy |
| `factors/` | the Fama–French 5-factor + momentum files used for the committed results (Kenneth French data library, downloaded 27–28 October 2026) |
| `requirements.txt` | package versions the committed results were produced with |

## Data

Put each input at the path shown, relative to the repository root. `check_inputs.py` checks them.

| input | path | source | needed by |
|---|---|---|---|
| CPZ characteristics and macro data | `datasets/char/Char_{train,valid,test}.npz`, `datasets/macro/macro_{train,valid,test}.npz` | `datasets.zip` from the Chen–Pelger–Zhu Google Drive linked in the repository [README](../README.md) | `cpz`, `cpz_robustness`, `consistency` |
| CPZ GAN SDF (published ensemble) | `sample_checkpoints/sdf_normalized_ensemble.npy` | `sample_checkpoints.zip` from the same Google Drive | `cpz`, `cpz_robustness` |
| CRSP monthly stock file, 1961–2024 | `crsp_download/crsp_msf_1961_2024.csv.gz` | WRDS; run [`scripts/wrds_export_msf.sas`](../scripts/wrds_export_msf.sas) | `osap_build` |
| Open Source Asset Pricing signals (October 2025 release) | `osap_download/signals/<Signal>.csv`, one file per signal with columns `permno, yyyymm, <Signal>` | [openassetpricing.com](https://www.openassetpricing.com/data/), firm-level signals | `osap_build` |
| Welch–Goyal predictors (2025 update) | `goyal_classic_download/Data2025.xlsx` | [Amit Goyal's website](https://sites.google.com/view/agoyal145) | `osap_build` |
| JKP US stock-month characteristics | `data_raw/kelly.tar.bz2` holding `kelly.csv` (or set `JKP_ARCHIVE`) | WRDS (Jensen, Kelly & Pedersen global factor data, US) | `jkp_build` |
| Fama–French factors | `datasets/factors/*.csv` | copied from `replication/factors/`; `FF_DOWNLOAD=1` downloads them afresh | `cpz`, `osap`, `jkp` |

**OSAP signals.** The build needs these 34 signals:

> AM, Accruals, AssetGrowth, AssetTurnover, BM, Beta, BetaFP, BidAskSpread, CF, CapTurnover, Cash, DivYieldAnn,
> EP, GP, High52, IdioVol3F, IdioVolAHT, InvestPPEInv, Leverage, NOA, OPLeverage, OperProf, PM, PctAcc,
> RealizedVol, RetNOA, RoE, SP, ShareIss1Y, VolMkt, VolumeTrend, cfp, depr, roaq

`check_inputs.py osap_build` lists any that are missing.

**JKP file.** `kelly.csv` has one row per US stock-month with these columns:
* `date` (yyyymmdd) and `id` (the CRSP permno);
* `ret_exc`, the month's excess return;
* the characteristics lagged one month, with suffix `_l`, ranked cross-sectionally to [−1, 1], with
  missing values set to 0.

The build uses JKP's 153 characteristics, listed in [`glassbox/jkp_themes.py`](../glassbox/jkp_themes.py). It
keeps CRSP securities only (`id` ≤ 99999). The file ends in December 2022, so the JKP post-sample test is
2017–2022.

**Licences.**
* CRSP, Compustat and the JKP data are licensed through WRDS. The OSAP, Goyal and CPZ files carry their
  providers' terms.
* `datasets/`, `results/` and the download folders are git-ignored. The panels and saved models under
  them contain or are derived from these data, so do not commit or share them.

## Stages

Run in this order; later stages read earlier outputs. Runtimes are wall-clock times on a 4-vCPU, 15 GB,
CPU-only container. Times marked † were measured while another run shared the CPUs, so expect about half
on an idle machine.

| stage | what it runs | main outputs (`results/` unless stated) | runtime |
|---|---|---|---|
| `tests` | unit and synthetic validation tests | — | 1 min |
| `synthetic` | Papers 1 and 2 on the planted synthetic panel | `paper1_synthetic`, `paper2_synthetic` | 10–20 min |
| `cpz` | Paper 1 ladder (with the neural rung unless `SKIP_NEURAL=1`), factor models and CPZ GAN audit; spanning tests against the GAN; Paper 2 (additive; additive + state with 100 bootstrap refits); GAN + glass-box combination; state-variable robustness | `paper1_cpz_neural`, `spanning_cpz_gan`, `paper2_cpz_additive`, `paper2_cpz_additive_state`, `combined_cpz`, `state_robustness_cpz` | 50 min for Paper 1 without the neural rung, plus 8.5 h for it; 40 + 40 min Paper 2; 2.7 h state robustness |
| `cpz_robustness` | knots and rule thresholds; discovery window and friction list | `knot_threshold_cpz`, `window_friction_cpz` | 30 + 12 min |
| `osap_build` | CRSP + OSAP + Goyal panel, median-imputed and complete-case | `datasets/osap`, `datasets/osap_cc` | 9 + 4 min |
| `osap` | Paper 1 (test 1992–2016); audit on 2017–2024; Paper 2; reversal-rule alpha over time | `paper1_osap`, `paper1_osap_post_only`, `paper2_osap_additive`, `rule_alpha` | 2.2 h + 4 min + 6 min + 1 min |
| `osap_robustness` | knots and thresholds; discovery window and friction list | `knot_threshold_osap`, `window_friction_osap` | 45 + 12 min |
| `jkp_build` | one pass over `kelly.csv`: the 153-characteristic panel and the June/December sample used for cross-validation; then the 13-theme panel | `datasets/jkp_panel`, `datasets/jkp/jkp_sample.npz`, `datasets/jkp_themes` | 25 min |
| `jkp` | Paper 1, 2017–2022 audit and Paper 2 on 153 characteristics and on the 13 themes; audit of the theme SDFs against the 153-characteristic rules | `paper1_jkp`, `paper1_jkp_post`, `paper2_jkp_additive`, `paper1_jkp_themes`, `paper1_jkp_themes_post`, `paper2_jkp_themes_additive`, `cross_audit_themes_on_jkp153` | about 4–8 h† |
| `consistency` | stock-level comparison of CPZ and OSAP characteristics; three-way cross-validation with JKP | `char_consistency`, `char_crossvalidation` | 3 min |
| `compare` | collects everything into `replication_output/` and compares it with `docs/` | `replication_output/comparison.csv` | seconds |
| `jkp_noclip` (optional) | the 153-characteristic ladder without state clipping, which shows the training failure reported in `jkp_results.md` | `paper1_jkp_noclip` | about 2–4 h† |

Dependencies between stages:
* `jkp_build` reads the Welch–Goyal macro files from `datasets/osap`, so run `osap_build` first.
* `consistency` needs the CPZ, OSAP and JKP data.

## Comparing with the committed results

`replication/run_all.sh compare` reports each committed table as one of:

| status | meaning |
|---|---|
| match | equal to 1e-6 relative |
| close | within 5% relative or 1e-4 absolute |
| differs | anything else, with the first difference shown |
| missing | the stage was not run |

What to expect:
* **These should match exactly:**
  * data builds (`build_meta.json`, the characteristic comparisons);
  * factor-model audits and the closed-form Paper 2 steps, given the same fitted models.
* **Network training is seeded,** but floating-point results depend on the CPU, the BLAS library, the
  thread count and the PyTorch version. On other hardware the ladder Sharpe ratios and everything
  downstream of the fitted SDFs (audits, Rashomon sets) can move in the last digits. Rarely, a rule
  near the confirmation threshold can switch.
* **This run reproduces the committed tables:** the mapping was checked against the original result
  folders, with 90 of 102 tables matching exactly. The other 12 are explained in the next section.

## Differences from the original run history

The committed results were produced interactively, so a few steps differ in form but not in substance:
* **CPZ Paper 1.** It was originally run in three steps with `--reuse`:
  1. the glass-box rungs;
  2. the neural rung added;
  3. the factor-model audits added once the factor files were downloaded.

  Here a single run produces all of them, with the same seeds.
* **CPZ spanning tests against the GAN** came from an ad hoc script on the saved models. They are now
  [`scripts/spanning_vs_external.py`](../scripts/spanning_vs_external.py), which reproduces the committed
  table. The neural row differs (0.105 instead of 0.103), because the committed number predates the
  evaluation-mode fix noted in `cpz_results.md`.
* **OSAP labels.** Two OSAP mappings were corrected after the first build: `Beta`/`MktBeta` were swapped,
  and `CF2P` now maps to OSAP's `CF` (`osap_results.md`, consistency section). The fix only renamed
  columns, and the committed tables carry the corrected names. A rebuild with the current
  `build_osap_panel.py` gives those names directly. The fitted numbers are unchanged.
* **JKP state clipping.** On the 153-characteristic panel, the static rungs of the committed clipped run
  were reused from the unclipped run, because they don't use the state variables. Here everything is
  fitted in one run.
* **JKP post-sample.** `config_glassbox/jkp_post.json` runs to 2022, because the JKP file ends there.

## Hardware and disk

* **Memory.** 16 GB is enough. The JKP stages need the most memory, and they ran in 15 GB.
* **GPU.** None is needed. The neural rung is the slow part and can be skipped with `SKIP_NEURAL=1`;
  only its ladder and spanning rows depend on it.
* **Disk.**
  * Panels: about 1.2 GB (CPZ), 1.5 GB (OSAP, both versions) and 1.4 GB (JKP, with themes).
  * The JKP archive itself: 4.8 GB.
  * Optional cache of the raw JKP extract (`JKP_CACHE=path`): 2 GB, which lets `jkp_build` rerun in
    about 3 minutes.
  * Results: a few hundred MB.
