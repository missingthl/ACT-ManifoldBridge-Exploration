# Latent Residual Flow Pilot Report

`latent_residual_direct` is a debug probe for train-only same-class RBF residual targets, not a competing paper baseline.

## Leaderboard
| Method | Probe? | Mean F1 | 95% CI | Δ vs U5 | Δ vs Random | Δ vs Bank | Δ vs CS-Flow | Δ vs wDBA | Seed W/T/L vs U5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wdba_sameclass | no | 0.667922 | [0.547478, 0.781965] | 0.00268042 | 0.0197921 | 0.0143764 | 0.0131077 | 0 | 9/0/12 |
| csta_topk_uniform_top5 | no | 0.665242 | [0.53335, 0.787462] | 0 | 0.0171117 | 0.011696 | 0.0104272 | -0.00268042 | 0/21/0 |
| dba_sameclass | no | 0.663309 | [0.534296, 0.787966] | -0.00193285 | 0.0151788 | 0.00976312 | 0.00849439 | -0.00461327 | 10/0/11 |
| cs_flow_single_step | no | 0.654815 | [0.519349, 0.778123] | -0.0104272 | 0.00668444 | 0.00126872 | 0 | -0.0131077 | 11/0/10 |
| csta_template_random_within_bank | no | 0.653546 | [0.520056, 0.774976] | -0.011696 | 0.00541572 | 0 | -0.00126872 | -0.0143764 | 4/1/16 |
| random_cov_state | no | 0.64813 | [0.505436, 0.774537] | -0.0171117 | 0 | -0.00541572 | -0.00668444 | -0.0197921 | 10/1/10 |
| latent_residual_flow | no | 0.6475 | [0.508605, 0.777321] | -0.0177419 | -0.000630176 | -0.0060459 | -0.00731462 | -0.0204223 | 6/0/15 |

## Latent Residual Diagnostics
| Method | Train MSE | Train Cos | Pred Vel Norm | Residual Rank | Residual Cos | Generated Cos | Effective Mult | Fallback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latent_residual_flow | 0.0986669 | 0.859699 | 0.397337 | 23.1766 | -0.0155675 | 0.0916253 | 10 | 0 |

## Decision Notes
- Direction collapse is reduced vs CS-Flow reference (0.091625 < 0.976901).
- Pilot7 gate should be considered only if latent_residual_flow beats CS-Flow, reduces collapse, and is competitive with random_cov or close to U5.
- latent_residual_flow Δ vs CS-Flow: -0.00731462; Δ vs U5: -0.0177419; Δ vs random_cov: -0.000630176.