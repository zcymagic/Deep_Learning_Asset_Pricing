# Paper 1 results: synthetic.json

## Complexity ladder (price of interpretability)

| model | n_params | SR_train | SR_valid | SR_test | EV_test | XS_R2_test | pairs | SR2_gap_vs_neural | alpha_of_neural | t_alpha_of_neural | R2_span_neural |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear | 11 | 0.5926 | 0.8240 | 0.7602 | 0.0042 | 0.0206 |  | -0.3972 | 0.0017 | 0.7162 | 0.2385 |
| linear_state | 33 | 0.6414 | 0.7087 | 0.6721 | 0.0367 | 0.2668 |  | -0.2710 | 0.0012 | 0.6028 | 0.3321 |
| additive | 91 | 0.7034 | 0.8397 | 0.7728 | 0.0054 | -0.0035 |  | -0.4164 | 0.0013 | 0.5556 | 0.2484 |
| additive_state | 273 | 0.7998 | 0.7676 | 0.7169 | 0.0109 | 0.0787 |  | -0.3333 | 0.0012 | 0.5419 | 0.2931 |
| additive_state_pairs | 348 | 0.9214 | 0.8112 | 0.7964 | 0.0059 | 0.0376 | LME x Spread, LME x IdioVol, Spread x IdioVol | -0.4536 | 0.0015 | 0.5753 | 0.2281 |
| neural | 1633 | 0.7950 | 0.2723 | 0.4251 | 0.0660 | 0.4112 |  | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

## Glass-box SDF (additive_state_pairs): weight-variance shares

| feature | variance_share |
|---|---|
| Spread x IdioVol | 0.2675 |
| LME x Spread | 0.2244 |
| LME x IdioVol | 0.1408 |
| Spread | 0.0917 |
| IdioVol | 0.0713 |
| CF2P | 0.0710 |
| BEME | 0.0386 |
| ST_REV | 0.0251 |
| OP | 0.0168 |
| LTurnover | 0.0147 |
| r12_2 | 0.0136 |
| Investment | 0.0125 |
| LME | 0.0119 |

## Audit summary (discovery: train; confirmation: valid + test)

| model | holdout_max_abs_t | holdout_global_p | holdout_rms_alpha | n_rules_discovered | n_sig_limits-to-arbitrage | share_limits-to-arbitrage | n_sig_state-dependent | share_state-dependent | n_sig_characteristic | share_characteristic | n_sig_mixed | share_mixed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear | 4.9476 | 0.0000 | 0.0015 | 2 | 0.0000 | 0.0000 | 1.0000 | 0.0892 | 0.0000 | 0.0000 | 1.0000 | 0.9108 |
| linear_state | 6.7628 | 0.0000 | 0.0017 | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| additive | 4.9924 | 0.0000 | 0.0014 | 0 |  |  |  |  |  |  |  |  |
| additive_state | 5.9051 | 0.0000 | 0.0017 | 0 |  |  |  |  |  |  |  |  |
| additive_state_pairs | 4.5211 | 0.0020 | 0.0015 | 0 |  |  |  |  |  |  |  |  |
| neural | 9.2064 | 0.0000 | 0.0026 | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| oracle | 9.1151 | 0.0000 | 0.0031 | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Failures of `linear`

| description | t_discovery | t_stepwise | t_holdout | t_holdout_stepwise | p_maxt | state_dep_var | t_state_dep | class |
|---|---|---|---|---|---|---|---|---|
| Spread > 0.30 AND LME < -0.30 | 5.977 | 5.977 | 7.585 | 7.585 | 0.000 | ip_growth | 2.837 | mixed |
| r12_2 > 0.30 AND ST_REV > 0.10 AND credit_spread > 0.06 | 4.500 | 5.665 | 2.373 | 3.133 | 0.020 | unemployment | -1.059 | state-dependent |

### Failures of `linear_state`

| description | t_discovery | t_stepwise | t_holdout | t_holdout_stepwise | p_maxt | state_dep_var | t_state_dep | class |
|---|---|---|---|---|---|---|---|---|
| Spread > 0.30 | 5.905 | 5.905 | 6.763 | 6.763 | 0.000 | vix | 1.739 | limits-to-arbitrage |

### Failures of `additive`

_none found_

### Failures of `additive_state`

_none found_

### Failures of `additive_state_pairs`

_none found_

### Failures of `neural`

| description | t_discovery | t_stepwise | t_holdout | t_holdout_stepwise | p_maxt | state_dep_var | t_state_dep | class |
|---|---|---|---|---|---|---|---|---|
| Spread > 0.30 AND LME < -0.10 | 7.739 | 7.739 | 10.832 | 10.832 | 0.000 | ip_growth | 1.342 | limits-to-arbitrage |

### Failures of `oracle`

| description | t_discovery | t_stepwise | t_holdout | t_holdout_stepwise | p_maxt | state_dep_var | t_state_dep | class |
|---|---|---|---|---|---|---|---|---|
| Spread > 0.30 | 8.652 | 8.652 | 9.115 | 9.115 | 0.000 | unemployment | 1.366 | limits-to-arbitrage |

