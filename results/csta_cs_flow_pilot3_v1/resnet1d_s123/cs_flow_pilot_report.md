# CS-Flow Pilot Report

Source: `standalone_projects/ACT_ManifoldBridge/results/csta_cs_flow_pilot3_v1/resnet1d_s123/per_seed_external.csv`

`cs_flow_target_direct` is a debug probe, not a competing paper baseline.

## Leaderboard

| method | role | n_rows | mean_f1 | delta_vs_u5 | delta_vs_random_cov | delta_vs_bank_random |
| --- | --- | --- | --- | --- | --- | --- |
| cs_flow_single_step | method | 9 | 0.950717 | 0.00832223 | 0.0100061 | 0.0220973 |
| csta_topk_uniform_top5 | method | 9 | 0.942395 | 0 | 0.0016839 | 0.0137751 |
| random_cov_state | method | 9 | 0.940711 | -0.0016839 | 0 | 0.0120912 |
| wdba_sameclass | method | 9 | 0.939619 | -0.00277578 | -0.00109188 | 0.0109993 |
| dba_sameclass | method | 9 | 0.936623 | -0.00577167 | -0.00408777 | 0.00800339 |
| cs_flow_target_direct | debug_probe_not_competing_method | 9 | 0.935589 | -0.00680626 | -0.00512236 | 0.0069688 |
| pca_cov_state | method | 9 | 0.93378 | -0.00861521 | -0.00693131 | 0.00515985 |
| ag_pia_multihead5 | method | 9 | 0.932318 | -0.010077 | -0.00839311 | 0.00369805 |
| csta_template_random_within_bank | method | 9 | 0.92862 | -0.0137751 | -0.0120912 | 0 |

## CS-Flow Diagnostics

| method | train_mse | train_cosine | pred_target_cosine | fallback_rate | effective_aug_multiplier | generated_pairwise_cosine | bridge_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cs_flow_single_step | 0.0153463 | 0.577135 | 0.41399 | 0 | 9.99907 | 0.995888 | 1 |
| cs_flow_target_direct | 0 | 1 | 1 | 0 | 1 | 1 | 1 |

## Diversity Collapse Check

- `cs_flow_single_step` shows possible generated-direction collapse before F1 interpretation (effective_aug_multiplier=9.99907, pairwise_cosine=0.995888).
- `cs_flow_target_direct` shows possible generated-direction collapse before F1 interpretation (effective_aug_multiplier=1, pairwise_cosine=1).

## Interpretation Guardrails

- Do not tune CS-Flow hyperparameters inside Phase 1 after this Pilot3.
- If target-direct is strong but learned flow is weak, inspect fitting diagnostics before Pilot7.
- Stop v1 if both target-direct and learned flow are below random covariance control.
