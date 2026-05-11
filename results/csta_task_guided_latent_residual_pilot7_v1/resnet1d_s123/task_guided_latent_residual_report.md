# Task-Guided Latent Residual Flow Report

`task_guided_residual_direct` is a debug-only probe for task-reweighted train-only residual targets; it is not a competing method or paper baseline.

## Leaderboard
| Method | Probe? | Mean F1 | 95% CI | Delta U5 | Delta Random | Delta Bank | Delta Latent Flow | Delta wDBA | Seed W/T/L vs U5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wdba_sameclass | no | 0.667922 | [0.550561, 0.783066] | 0.00268042 | 0.0197921 | 0.0143764 | 0.0204223 | 0 | 9/0/12 |
| csta_topk_uniform_top5 | no | 0.665242 | [0.536484, 0.789836] | 0 | 0.0171117 | 0.011696 | 0.0177419 | -0.00268042 | 0/21/0 |
| dba_sameclass | no | 0.663309 | [0.537124, 0.786556] | -0.00193285 | 0.0151788 | 0.00976312 | 0.015809 | -0.00461327 | 10/0/11 |
| csta_template_random_within_bank | no | 0.653546 | [0.523265, 0.777377] | -0.011696 | 0.00541572 | 0 | 0.0060459 | -0.0143764 | 4/1/16 |
| random_cov_state | no | 0.64813 | [0.511432, 0.779631] | -0.0171117 | 0 | -0.00541572 | 0.000630176 | -0.0197921 | 10/1/10 |
| latent_residual_flow | no | 0.6475 | [0.511126, 0.778664] | -0.0177419 | -0.000630176 | -0.0060459 | 0 | -0.0204223 | 6/0/15 |
| task_guided_latent_residual_flow | no | 0.646855 | [0.510916, 0.777197] | -0.0183867 | -0.001275 | -0.00669073 | -0.000644827 | -0.0210671 | 9/0/12 |
| task_guided_residual_direct | yes | 0.641684 | [0.498271, 0.77926] | -0.0235573 | -0.00644562 | -0.0118613 | -0.00581545 | -0.0262377 | 6/0/15 |
| latent_residual_direct | no | 0.633353 | [0.490194, 0.772934] | -0.0318891 | -0.0147774 | -0.0201931 | -0.0141472 | -0.0345695 | 9/0/12 |

## Task Guidance Diagnostics
| Method | Task Entropy | Eff Support | KL Task/Geo | Bad Margin Mass | Wrong Pred Mass | Fallback | Invalid | Gen Cos | Eff Mult |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_guided_latent_residual_flow | 1.94009 | 10.4422 | 0.433222 | 0.327907 | 0.327907 | 0 | 0 | 0.114134 | 10 |
| task_guided_residual_direct | 1.94009 | 10.4422 | 0.433222 | 0.327907 | 0.327907 | 0 | 0 | 0.634864 | 5.15512 |

## Fitting And Safety
| Method | Train MSE | Train Cos | Utility Mean | Margin Mean | Warmup Loss | Bridge Success | Gamma Ratio | Safe Clip | Transport Err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_guided_latent_residual_flow | 0.0914275 | 0.855245 | 0.866168 | 1.61722 | 0.725474 | 1 | 0.994325 | 0.025 | 0.772323 |
| task_guided_residual_direct | 0 | 1 | 0.866168 | 1.61722 | 0.725474 | 1 | 0.994325 | 0.025 | 0.775615 |

## Interpretation Notes
- Direction collapse is below the CS-Flow reference; inspect whether F1 also improves.
- task_guided_latent_residual_flow deltas: vs latent_flow=-0.000644827, vs U5=-0.0183867, vs random_cov=-0.001275.
- Direct probe gap (flow - direct): 0.0135024. If direct improves but flow does not, inspect fitting before changing the task utility.