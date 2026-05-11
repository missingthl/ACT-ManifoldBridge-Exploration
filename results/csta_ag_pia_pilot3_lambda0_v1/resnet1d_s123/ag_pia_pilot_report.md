# AG-PIA Pilot Report

Source: `standalone_projects/ACT_ManifoldBridge/results/csta_ag_pia_pilot3_lambda0_v1/resnet1d_s123/per_seed_external.csv`

## Leaderboard

| method | n_rows | mean_f1 | std_f1 |
| --- | --- | --- | --- |
| csta_topk_uniform_top5 | 9 | 0.942395 | 0.042076 |
| random_cov_state | 9 | 0.940711 | 0.045109 |
| wdba_sameclass | 9 | 0.939619 | 0.0355065 |
| ag_target_direct | 9 | 0.936676 | 0.0530128 |
| dba_sameclass | 9 | 0.936623 | 0.0492153 |
| ag_pia_single | 9 | 0.935289 | 0.0587124 |
| ag_pia_multihead5 | 9 | 0.934602 | 0.0543551 |
| pca_cov_state | 9 | 0.93378 | 0.0441845 |
| csta_template_random_within_bank | 9 | 0.92862 | 0.0415706 |

## AG Diagnostics

| method | ag_target_effective_rank | ag_target_pairwise_cosine_mean | ag_target_norm_mean | ag_target_norm_std | ag_head_pairwise_cosine_mean | ag_head_effective_rank | ag_head_usage_entropy | ag_operator_train_mse_mean | ag_operator_train_cosine_mean | ag_pred_target_cosine_mean | ag_tangent_available_rate | ag_fallback_rate | ag_pos_dist_mean | ag_neg_centroid_dist_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ag_target_direct | 42.9687 | 0.000805217 | 2.72127 | 0.544173 |  | 0 | -1.00009e-12 | 0 | 1 | 1 | 1 | 0 | 3.80954 | 3.6987 |
| ag_pia_single | 42.9687 | 0.000805217 | 2.72127 | 0.544173 |  | 0 | -1.00009e-12 | 0.00897771 | 0.875084 | 0.875084 | 1 | 0 | 3.80954 | 3.6987 |
| ag_pia_multihead5 | 49.1023 | 0.00265203 | 3.28034 | 0.682204 | 0.224441 | 3.72466 | 1.60845 | 0.00759573 | 0.891496 | 0.891893 | 1 | 0 | 4.24281 | 3.6987 |

## Interpretation Guardrails

- `ag_target_direct` is debug-only and must not be presented as a paper baseline.
- If direct target is strong but AG operator is weak, inspect operator train cosine/MSE before abandoning AG-PIA.
- AG-PIA rows are CSTA internal methods, not external baselines.
