# Paper 2 results: synthetic.json

Glass-box SDF: `additive_state`, 273 parameters, J* = 2.926e-05, bootstrap-implied simultaneous eps (95%) = 1.224

## Reference performance

| split | sharpe_raw | sharpe | ev | xs_r2 |
|---|---|---|---|---|
| train | 0.9323 | 0.8947 | 0.0039 | 0.0249 |
| valid | 0.7538 | 0.7797 | 0.0050 | 0.0187 |
| test | 0.7642 | 0.7486 | 0.0069 | 0.0461 |

## Identification: eps=0.002

| feature | importance | min | max | identified | slope | lo | hi | sign_identified |
|---|---|---|---|---|---|---|---|---|
| LME | 0.101 | 0.042 | 0.192 | True | -0.104 | -0.156 | -0.051 | True |
| Spread | 0.370 | 0.246 | 0.522 | True | 0.235 | 0.184 | 0.286 | True |
| IdioVol | 0.072 | 0.028 | 0.139 | True | -0.084 | -0.128 | -0.040 | True |
| LTurnover | 0.023 | 0.004 | 0.064 | False | -0.033 | -0.079 | 0.012 | False |
| r12_2 | 0.014 | 0.009 | 0.024 | False | 0.006 | -0.011 | 0.023 | False |
| ST_REV | 0.066 | 0.027 | 0.124 | True | 0.093 | 0.052 | 0.133 | True |
| BEME | 0.608 | 0.225 | 1.180 | True | -0.076 | -0.195 | 0.043 | False |
| CF2P | 0.846 | 0.366 | 1.533 | True | 0.102 | -0.021 | 0.225 | False |
| OP | 0.060 | 0.020 | 0.124 | True | -0.052 | -0.095 | -0.008 | True |
| Investment | 0.025 | 0.009 | 0.054 | False | -0.029 | -0.069 | 0.011 | False |
| group:value | 0.154 | 0.094 | 0.232 | True |  |  |  |  |

## Identification: eps=0.01

| feature | importance | min | max | identified | slope | lo | hi | sign_identified |
|---|---|---|---|---|---|---|---|---|
| LME | 0.101 | 0.008 | 0.352 | False | -0.104 | -0.221 | 0.013 | False |
| Spread | 0.370 | 0.132 | 0.751 | True | 0.235 | 0.121 | 0.349 | True |
| IdioVol | 0.072 | 0.002 | 0.254 | False | -0.084 | -0.183 | 0.015 | False |
| LTurnover | 0.023 | 0.000 | 0.148 | False | -0.033 | -0.135 | 0.069 | False |
| r12_2 | 0.014 | 0.007 | 0.043 | False | 0.006 | -0.032 | 0.045 | False |
| ST_REV | 0.066 | 0.003 | 0.225 | False | 0.093 | 0.003 | 0.183 | True |
| BEME | 0.608 | 0.010 | 2.147 | False | -0.076 | -0.343 | 0.190 | False |
| CF2P | 0.846 | 0.049 | 2.679 | True | 0.102 | -0.173 | 0.377 | False |
| OP | 0.060 | 0.000 | 0.235 | False | -0.052 | -0.149 | 0.046 | False |
| Investment | 0.025 | 0.000 | 0.110 | False | -0.029 | -0.118 | 0.061 | False |
| group:value | 0.154 | 0.045 | 0.359 | True |  |  |  |  |

## Identification: eps=0.05

| feature | importance | min | max | identified | slope | lo | hi | sign_identified |
|---|---|---|---|---|---|---|---|---|
| LME | 0.101 | 0.000 | 0.905 | False | -0.104 | -0.366 | 0.158 | False |
| Spread | 0.370 | 0.011 | 1.438 | True | 0.235 | -0.020 | 0.490 | False |
| IdioVol | 0.072 | 0.000 | 0.659 | False | -0.084 | -0.306 | 0.137 | False |
| LTurnover | 0.023 | 0.000 | 0.488 | False | -0.033 | -0.261 | 0.195 | False |
| r12_2 | 0.014 | 0.004 | 0.125 | False | 0.006 | -0.080 | 0.092 | False |
| ST_REV | 0.066 | 0.000 | 0.579 | False | 0.093 | -0.109 | 0.294 | False |
| BEME | 0.608 | 0.000 | 5.364 | False | -0.076 | -0.672 | 0.519 | False |
| CF2P | 0.846 | 0.000 | 6.506 | False | 0.102 | -0.514 | 0.718 | False |
| OP | 0.060 | 0.000 | 0.624 | False | -0.052 | -0.269 | 0.166 | False |
| Investment | 0.025 | 0.000 | 0.330 | False | -0.029 | -0.229 | 0.171 | False |
| group:value | 0.154 | 0.001 | 0.767 | False |  |  |  |  |

## Identification: bootstrap (marginal, Bonferroni)

| feature | importance | min | max | identified | slope | lo | hi | sign_identified |
|---|---|---|---|---|---|---|---|---|
| LME | 0.101 | 0.022 | 1.524 | True | -0.104 | -0.431 | 0.166 | False |
| Spread | 0.370 | 0.105 | 2.417 | True | 0.235 | -0.071 | 0.469 | False |
| IdioVol | 0.072 | 0.021 | 0.723 | True | -0.084 | -0.283 | 0.149 | False |
| LTurnover | 0.023 | 0.022 | 1.057 | True | -0.033 | -0.276 | 0.194 | False |
| r12_2 | 0.014 | 0.009 | 0.173 | False | 0.006 | -0.071 | 0.109 | False |
| ST_REV | 0.066 | 0.014 | 0.706 | True | 0.093 | -0.115 | 0.285 | False |
| BEME | 0.608 | 0.126 | 6.508 | True | -0.076 | -0.683 | 0.509 | False |
| CF2P | 0.846 | 0.155 | 8.487 | True | 0.102 | -0.528 | 0.750 | False |
| OP | 0.060 | 0.015 | 1.389 | True | -0.052 | -0.242 | 0.250 | False |
| Investment | 0.025 | 0.014 | 0.740 | True | -0.029 | -0.309 | 0.200 | False |
| group:value | 0.154 | 0.093 | 1.020 | True |  |  |  |  |

## SDF ambiguity

Share of stock-months whose SDF weight sign is not identified (eps=0.01): {'train': 0.7865370149815082, 'valid': 0.7664440301149371, 'test': 0.8121064833884502}

### Hold-out sorts on ambiguity

| portfolio | mean | t_mean | alpha | t_alpha |
|---|---|---|---|---|
| Q1 | 0.0125 | 4.9906 | -0.0001 | -0.1635 |
| Q2 | 0.0129 | 5.2119 | -0.0003 | -0.4845 |
| Q3 | 0.0121 | 5.0214 | -0.0005 | -0.8358 |
| Q4 | 0.0132 | 5.5465 | 0.0008 | 1.5134 |
| Q5 | 0.0132 | 5.3917 | 0.0000 | 0.0283 |
| H-L | 0.0007 | 0.9903 | 0.0001 | 0.1276 |

### Hold-out Fama-MacBeth: returns on standardized ambiguity, reference SDF weight, |weight|, characteristics and their squares

| variable | coef | t |
|---|---|---|
| const | 0.0091 | 3.3447 |
| ambiguity | -0.0000 | -0.1734 |
| sdf_weight | 0.0026 | 0.6858 |
| abs_sdf_weight | 0.0003 | 0.9601 |
| LME | -0.0012 | -0.2825 |
| Spread | -0.0018 | -0.2217 |
| IdioVol | -0.0022 | -0.6992 |
| LTurnover | 0.0009 | 0.5193 |
| r12_2 | 0.0587 | 5.1662 |
| ST_REV | -0.0015 | -0.4028 |
| BEME | 0.0194 | 2.1439 |
| CF2P | -0.0026 | -0.2555 |
| OP | 0.0071 | 1.8485 |
| Investment | -0.0085 | -3.4439 |
| LME^2 | 0.0038 | 0.4692 |
| Spread^2 | 0.0083 | 0.7051 |
| IdioVol^2 | -0.0002 | -0.0590 |
| LTurnover^2 | 0.0026 | 0.5409 |
| r12_2^2 | 0.0026 | 0.7694 |
| ST_REV^2 | 0.0273 | 4.4399 |
| BEME^2 | 0.0082 | 1.4049 |
| CF2P^2 | -0.0022 | -0.4042 |
| OP^2 | -0.0050 | -1.3404 |
| Investment^2 | -0.0018 | -0.4197 |

### Monthly ambiguity index vs macro series

| macro | corr |
|---|---|
| vix | 0.5729 |
| credit_spread | 0.5627 |
| ip_growth | -0.1638 |
| inflation | -0.0625 |
| unemployment | -0.0098 |
| term_spread | -0.0061 |

## Sharpe ratio across the Rashomon set (test, raw SDF portfolio)

| eps | sharpe_raw_ref | min | max |
|---|---|---|---|
| 0.0100 | 0.7642 | 0.7470 | 0.7815 |
