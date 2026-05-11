# Task-Guided Latent Residual Flow Report

`task_guided_residual_direct` is a debug-only probe for task-reweighted train-only residual targets; it is not a competing method or paper baseline.

## Leaderboard
| Method | Probe? | Mean F1 | 95% CI | Delta U5 | Delta Random | Delta Bank | Delta Latent Flow | Delta wDBA | Seed W/T/L vs U5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_guided_latent_residual_flow | no | 0.945059 | [0.916917, 0.969717] | 0.00266372 | 0.00434763 | 0.0164388 | 0.00641358 | 0.00543951 | 5/0/4 |
| csta_topk_uniform_top5 | no | 0.942395 | [0.91468, 0.966594] | 0 | 0.0016839 | 0.0137751 | 0.00374986 | 0.00277578 | 0/9/0 |
| latent_residual_direct | no | 0.940724 | [0.910267, 0.966153] | -0.00167122 | 1.26776e-05 | 0.0121038 | 0.00207863 | 0.00110456 | 6/0/3 |
| random_cov_state | no | 0.940711 | [0.912295, 0.967844] | -0.0016839 | 0 | 0.0120912 | 0.00206595 | 0.00109188 | 5/1/3 |
| wdba_sameclass | no | 0.939619 | [0.917374, 0.959946] | -0.00277578 | -0.00109188 | 0.0109993 | 0.000974073 | 0 | 3/0/6 |
| task_guided_residual_direct | yes | 0.939256 | [0.907859, 0.963819] | -0.00313871 | -0.00145481 | 0.0106363 | 0.000611144 | -0.000362929 | 2/0/7 |
| latent_residual_flow | no | 0.938645 | [0.910103, 0.964684] | -0.00374986 | -0.00206595 | 0.0100252 | 0 | -0.000974073 | 4/0/5 |
| dba_sameclass | no | 0.936623 | [0.9046, 0.964113] | -0.00577167 | -0.00408777 | 0.00800339 | -0.00202181 | -0.00299589 | 3/0/6 |
| csta_template_random_within_bank | no | 0.92862 | [0.903317, 0.953585] | -0.0137751 | -0.0120912 | 0 | -0.0100252 | -0.0109993 | 0/0/9 |

## Task Guidance Diagnostics
| Method | Task Entropy | Eff Support | KL Task/Geo | Bad Margin Mass | Wrong Pred Mass | Fallback | Invalid | Gen Cos | Eff Mult |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_guided_latent_residual_flow | 2.70487 | 15.3143 | 0.505749 | 0.0187729 | 0.0187729 | 0 | 0 | 0.119367 | 10 |
| task_guided_residual_direct | 2.70487 | 15.3143 | 0.505749 | 0.0187729 | 0.0187729 | 0 | 0 | 0.543188 | 6.93868 |

## Fitting And Safety
| Method | Train MSE | Train Cos | Utility Mean | Margin Mean | Warmup Loss | Bridge Success | Safe Clip | Transport Err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_guided_latent_residual_flow | 0.0300493 | 0.869274 | 0.1179 | 3.6626 | 0.318036 | 1 | 0 | 1.79117 |
| task_guided_residual_direct | 0 | 1 | 0.1179 | 3.6626 | 0.318036 | 1 | 0 | 1.79882 |

## Interpretation Notes
- Direction collapse is below the CS-Flow reference; inspect whether F1 also improves.
- task_guided_latent_residual_flow deltas: vs latent_flow=0.00641358, vs U5=0.00266372, vs random_cov=0.00434763.
- Direct probe gap (flow - direct): 0.00433495. If direct improves but flow does not, inspect fitting before changing the task utility.