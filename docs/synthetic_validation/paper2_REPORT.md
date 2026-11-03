# Paper 2 results: synthetic.json

Glass-box SDF: `additive_state`, 273 parameters, L2 shrinkage 0.001 (chosen on validation Sharpe), J* = 3.127e-05, bootstrap-implied simultaneous eps (95%) = 0.917

## Shrinkage selection (closed-form SDF)

| shrink | sharpe_train | sharpe_valid |
|---|---|---|
| 0.0000 | 0.9323 | 0.7538 |
| 0.0001 | 0.9322 | 0.7677 |
| 0.0010 | 0.9111 | 0.7868 |
| 0.0100 | 0.7707 | 0.7587 |
| 0.1000 | 0.6038 | 0.7118 |
| 1.0000 | 0.5302 | 0.6678 |

## Reference performance

| split | sharpe_raw | sharpe | ev | xs_r2 |
|---|---|---|---|---|
| train | 0.9111 | 0.8015 | 0.0081 | 0.0442 |
| valid | 0.7868 | 0.7695 | 0.0113 | 0.0392 |
| test | 0.7722 | 0.7199 | 0.0137 | 0.0987 |

## Identification: eps=0.002

| feature | importance | min | max | identified | slope | lo | hi | sign_identified |
|---|---|---|---|---|---|---|---|---|
| LME | 0.146 | 0.073 | 0.247 | True | -0.098 | -0.137 | -0.060 | True |
| Spread | 0.332 | 0.222 | 0.468 | True | 0.163 | 0.125 | 0.201 | True |
| IdioVol | 0.051 | 0.014 | 0.113 | True | -0.049 | -0.085 | -0.012 | True |
| LTurnover | 0.026 | 0.005 | 0.068 | False | -0.025 | -0.062 | 0.013 | False |
| r12_2 | 0.018 | 0.014 | 0.027 | True | 0.015 | -0.002 | 0.032 | False |
| ST_REV | 0.073 | 0.028 | 0.140 | True | 0.072 | 0.037 | 0.106 | True |
| BEME | 0.025 | 0.000 | 0.095 | False | 0.003 | -0.040 | 0.046 | False |
| CF2P | 0.048 | 0.011 | 0.116 | True | 0.019 | -0.025 | 0.063 | False |
| OP | 0.049 | 0.013 | 0.110 | True | -0.039 | -0.075 | -0.002 | True |
| Investment | 0.021 | 0.005 | 0.050 | False | -0.015 | -0.049 | 0.020 | False |
| group:value | 0.096 | 0.058 | 0.143 | True |  |  |  |  |

## Identification: eps=0.01

| feature | importance | min | max | identified | slope | lo | hi | sign_identified |
|---|---|---|---|---|---|---|---|---|
| LME | 0.146 | 0.019 | 0.413 | True | -0.098 | -0.184 | -0.012 | True |
| Spread | 0.332 | 0.119 | 0.671 | True | 0.163 | 0.077 | 0.249 | True |
| IdioVol | 0.051 | 0.000 | 0.225 | False | -0.049 | -0.130 | 0.033 | False |
| LTurnover | 0.026 | 0.000 | 0.157 | False | -0.025 | -0.108 | 0.059 | False |
| r12_2 | 0.018 | 0.011 | 0.051 | True | 0.015 | -0.022 | 0.052 | False |
| ST_REV | 0.073 | 0.002 | 0.254 | False | 0.072 | -0.005 | 0.149 | False |
| BEME | 0.025 | 0.000 | 0.247 | False | 0.003 | -0.093 | 0.099 | False |
| CF2P | 0.048 | 0.000 | 0.245 | False | 0.019 | -0.079 | 0.118 | False |
| OP | 0.049 | 0.000 | 0.220 | False | -0.039 | -0.120 | 0.043 | False |
| Investment | 0.021 | 0.000 | 0.108 | False | -0.015 | -0.091 | 0.062 | False |
| group:value | 0.096 | 0.025 | 0.217 | True |  |  |  |  |

## Identification: eps=0.05

| feature | importance | min | max | identified | slope | lo | hi | sign_identified |
|---|---|---|---|---|---|---|---|---|
| LME | 0.146 | 0.000 | 0.951 | False | -0.098 | -0.291 | 0.095 | False |
| Spread | 0.332 | 0.007 | 1.272 | False | 0.163 | -0.029 | 0.355 | False |
| IdioVol | 0.051 | 0.000 | 0.630 | False | -0.049 | -0.230 | 0.133 | False |
| LTurnover | 0.026 | 0.000 | 0.525 | False | -0.025 | -0.211 | 0.162 | False |
| r12_2 | 0.018 | 0.007 | 0.170 | False | 0.015 | -0.068 | 0.098 | False |
| ST_REV | 0.073 | 0.000 | 0.643 | False | 0.072 | -0.100 | 0.244 | False |
| BEME | 0.025 | 0.000 | 0.843 | False | 0.003 | -0.211 | 0.217 | False |
| CF2P | 0.048 | 0.000 | 0.753 | False | 0.019 | -0.201 | 0.239 | False |
| OP | 0.049 | 0.000 | 0.605 | False | -0.039 | -0.221 | 0.143 | False |
| Investment | 0.021 | 0.000 | 0.353 | False | -0.015 | -0.186 | 0.157 | False |
| group:value | 0.096 | 0.000 | 0.443 | False |  |  |  |  |

## Identification: bootstrap (marginal, Bonferroni)

| feature | importance | min | max | identified | slope | lo | hi | sign_identified |
|---|---|---|---|---|---|---|---|---|
| LME | 0.146 | 0.028 | 1.009 | True | -0.098 | -0.253 | 0.059 | False |
| Spread | 0.332 | 0.115 | 1.403 | True | 0.163 | -0.016 | 0.282 | False |
| IdioVol | 0.051 | 0.020 | 0.506 | True | -0.049 | -0.170 | 0.109 | False |
| LTurnover | 0.026 | 0.029 | 0.636 | True | -0.025 | -0.194 | 0.111 | False |
| r12_2 | 0.018 | 0.016 | 0.200 | True | 0.015 | -0.046 | 0.099 | False |
| ST_REV | 0.073 | 0.010 | 0.733 | True | 0.072 | -0.094 | 0.203 | False |
| BEME | 0.025 | 0.005 | 0.270 | False | 0.003 | -0.085 | 0.100 | False |
| CF2P | 0.048 | 0.019 | 0.263 | True | 0.019 | -0.076 | 0.130 | False |
| OP | 0.049 | 0.018 | 0.684 | True | -0.039 | -0.183 | 0.141 | False |
| Investment | 0.021 | 0.016 | 0.519 | True | -0.015 | -0.184 | 0.147 | False |
| group:value | 0.096 | 0.038 | 0.452 | True |  |  |  |  |

## SDF ambiguity

Share of stock-months whose SDF weight sign is not identified (eps=0.01): {'train': 0.8331309158152072, 'valid': 0.8139293384626277, 'test': 0.8431817914347248}

### Hold-out sorts on ambiguity

| portfolio | mean | t_mean | alpha | t_alpha |
|---|---|---|---|---|
| Q1 | 0.0121 | 4.7415 | -0.0006 | -1.2209 |
| Q2 | 0.0126 | 5.2676 | -0.0005 | -1.0867 |
| Q3 | 0.0127 | 5.2416 | -0.0000 | -0.0536 |
| Q4 | 0.0131 | 5.3765 | 0.0005 | 1.3013 |
| Q5 | 0.0133 | 5.6873 | 0.0007 | 1.1419 |
| H-L | 0.0012 | 1.8068 | 0.0013 | 1.3832 |

### Hold-out Fama-MacBeth: returns on standardized ambiguity, reference SDF weight, |weight|, characteristics and their squares

| variable | coef | t |
|---|---|---|
| const | 0.0098 | 3.5582 |
| ambiguity | 0.0002 | 0.5703 |
| sdf_weight | 0.0018 | 0.5912 |
| abs_sdf_weight | 0.0005 | 1.6830 |
| LME | -0.0015 | -0.3670 |
| Spread | -0.0004 | -0.0602 |
| IdioVol | -0.0022 | -1.1231 |
| LTurnover | 0.0013 | 0.9119 |
| r12_2 | 0.0588 | 5.1709 |
| ST_REV | -0.0010 | -0.3216 |
| BEME | 0.0160 | 4.6027 |
| CF2P | 0.0019 | 0.6497 |
| OP | 0.0072 | 2.3785 |
| Investment | -0.0083 | -3.9281 |
| LME^2 | 0.0047 | 0.6709 |
| Spread^2 | 0.0086 | 0.7240 |
| IdioVol^2 | -0.0021 | -0.5630 |
| LTurnover^2 | 0.0004 | 0.0856 |
| r12_2^2 | 0.0020 | 0.6849 |
| ST_REV^2 | 0.0260 | 4.1536 |
| BEME^2 | 0.0067 | 1.1608 |
| CF2P^2 | -0.0029 | -0.5332 |
| OP^2 | -0.0053 | -1.3433 |
| Investment^2 | -0.0027 | -0.5685 |

### Monthly ambiguity index vs macro series (top 15 by |corr|; all in ambiguity_index_vs_macro.csv)

| macro | corr |
|---|---|
| vix | 0.7180 |
| credit_spread | 0.5668 |
| ip_growth | -0.1549 |
| inflation | -0.1132 |
| term_spread | -0.0277 |
| unemployment | 0.0210 |

## Sharpe ratio across the Rashomon set (test, raw SDF portfolio)

| eps | sharpe_raw_ref | min | max |
|---|---|---|---|
| 0.0100 | 0.7722 | 0.7557 | 0.7891 |
