# Latent Residual Flow Pilot Report

`latent_residual_direct` is a debug probe for train-only same-class RBF residual targets, not a competing paper baseline.

## Leaderboard
| Method | Probe? | Mean F1 | 95% CI | Δ vs U5 | Δ vs Random | Δ vs Bank | Δ vs CS-Flow | Δ vs wDBA | Seed W/T/L vs U5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| csta_topk_uniform_top5 | no | 0.942395 | [0.915205, 0.966729] | 0 | 0.0016839 | 0.0137751 | 0.00343299 | 0.00277578 | 0/9/0 |
| latent_residual_direct | yes | 0.940724 | [0.910634, 0.968127] | -0.00167122 | 1.26776e-05 | 0.0121038 | 0.00176177 | 0.00110456 | 6/0/3 |
| random_cov_state | no | 0.940711 | [0.912093, 0.967288] | -0.0016839 | 0 | 0.0120912 | 0.00174909 | 0.00109188 | 5/1/3 |
| wdba_sameclass | no | 0.939619 | [0.91759, 0.961247] | -0.00277578 | -0.00109188 | 0.0109993 | 0.000657206 | 0 | 3/0/6 |
| cs_flow_single_step | no | 0.938962 | [0.910839, 0.96483] | -0.00343299 | -0.00174909 | 0.0103421 | 0 | -0.000657206 | 3/0/6 |
| latent_residual_flow | no | 0.938645 | [0.910419, 0.965248] | -0.00374986 | -0.00206595 | 0.0100252 | -0.000316867 | -0.000974073 | 4/0/5 |
| dba_sameclass | no | 0.936623 | [0.905041, 0.965524] | -0.00577167 | -0.00408777 | 0.00800339 | -0.00233868 | -0.00299589 | 3/0/6 |
| csta_template_random_within_bank | no | 0.92862 | [0.903123, 0.953896] | -0.0137751 | -0.0120912 | 0 | -0.0103421 | -0.0109993 | 0/0/9 |

## Latent Residual Diagnostics
| Method | Train MSE | Train Cos | Pred Vel Norm | Residual Rank | Residual Cos | Generated Cos | Effective Mult | Fallback |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| latent_residual_direct | 0 | 1 | 1.41051 | 39.6782 | -0.00221453 | 0.49943 | 8.21523 | 0 |
| latent_residual_flow | 0.030228 | 0.870818 | 0.392133 | 39.6782 | -0.00221453 | 0.11189 | 10 | 0 |

## Decision Notes
- Direction collapse is reduced vs CS-Flow reference (0.111890 < 0.976901).
- Pilot7 gate should be considered only if latent_residual_flow beats CS-Flow, reduces collapse, and is competitive with random_cov or close to U5.
- latent_residual_flow Δ vs CS-Flow: -0.000316867; Δ vs U5: -0.00374986; Δ vs random_cov: -0.00206595.
- Direct probe gap (flow - direct): -0.00207863. If direct is strong but flow is weak, inspect fitting diagnostics before changing target sampling.