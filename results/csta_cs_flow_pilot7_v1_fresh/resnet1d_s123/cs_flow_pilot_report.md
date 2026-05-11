# CS-Flow Pilot 7 Final Report

## Leaderboard
| Method | Mean F1 | 95% CI | Δ vs U5 | Δ vs Random | Δ vs wDBA | Δ vs Bank | Seed W/T/L vs U5 | Dataset W/T/L vs U5 | Seed W/T/L vs wDBA | Wilcoxon p vs U5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wdba_sameclass | 0.667922 | [0.547478, 0.781965] | 0.00268042 | 0.0197921 | 0 | 0.0143764 | 9/0/12 | 3/0/4 | 0/21/0 | 1 |
| csta_topk_uniform_top5 | 0.665242 | [0.53335, 0.787462] | 0 | 0.0171117 | -0.00268042 | 0.011696 | 0/21/0 | 0/7/0 | 12/0/9 | n/a |
| dba_sameclass | 0.663309 | [0.534296, 0.787966] | -0.00193285 | 0.0151788 | -0.00461327 | 0.00976312 | 10/0/11 | 3/0/4 | 10/0/11 | 0.759288 |
| csta_template_random_within_bank | 0.653546 | [0.520056, 0.774976] | -0.011696 | 0.00541572 | -0.0143764 | 0 | 4/1/16 | 1/0/6 | 8/0/13 | 0.0250935 |
| random_cov_state | 0.64813 | [0.505436, 0.774537] | -0.0171117 | 0 | -0.0197921 | -0.00541572 | 10/1/10 | 2/0/5 | 10/0/11 | 0.455273 |
| pca_cov_state | 0.64703 | [0.501593, 0.779218] | -0.0182116 | -0.00109991 | -0.020892 | -0.00651564 | 8/0/13 | 2/0/5 | 8/0/13 | 0.320457 |
| cs_flow_single_step | 0.636248 | [0.492459, 0.763936] | -0.0289938 | -0.0118821 | -0.0316742 | -0.0172978 | 9/1/11 | 3/0/4 | 13/0/8 | 0.601213 |

## CS-Flow Diagnostics
| Method | Train MSE | Pred-Target Cos | Effective Rank | Pairwise Cosine | Uniq Ratio | Bridge Succ |
| --- | --- | --- | --- | --- | --- | --- |
| cs_flow_single_step | 0.0530832 | 0.353308 | 25.9975 | 0.976901 | 0.999883 | 1 |

## Paired CS-Flow Comparisons
| reference | mean_delta | bootstrap_95ci | seed_W/T/L | dataset_W/T/L | wilcoxon_p |
| --- | --- | --- | --- | --- | --- |
| u5 | -0.0289938 | [-0.0698235, 0.00229216] | 9/1/11 | 3/0/4 | 0.601213 |
| random_cov | -0.0118821 | [-0.0331127, 0.00742023] | 7/1/13 | 3/0/4 | 0.390533 |
| wdba | -0.0316742 | [-0.0771958, 0.00719376] | 13/0/8 | 4/0/3 | 0.609149 |
| bank_random | -0.0172978 | [-0.0592743, 0.0179843] | 13/0/8 | 3/0/4 | 0.657827 |

## Critical Mechanism Analysis
> [!WARNING]
> **Concentrated Dominant Flow Detected**
> CS-Flow v1 learns concentrated dominant vicinal flow directions (pairwise_cosine=0.976901) rather than high-diversity generation.

## Execution Guardrails
- Pilot7 stability gate status: **PENDING/FAILED**