# Label-Consistent Latent Residual Flow Report

`lc_residual_direct` is a debug-only probe, not a competing paper baseline.

## Leaderboard
| Method | Probe? | Mean F1 | 95% CI | Delta U5 | Delta Random | Delta Bank | Delta Task | Seed W/T/L vs U5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task_guided_latent_residual_flow | no | 0.945059 | [0.917177, 0.970235] | 0.00266372 | 0.00434763 | 0.0164388 | 0 | 5/0/4 |
| csta_topk_uniform_top5 | no | 0.942395 | [0.915002, 0.966735] | 0 | 0.0016839 | 0.0137751 | -0.00266372 | 0/9/0 |
| random_cov_state | no | 0.940711 | [0.912462, 0.967167] | -0.0016839 | 0 | 0.0120912 | -0.00434763 | 5/1/3 |
| wdba_sameclass | no | 0.939619 | [0.917566, 0.959745] | -0.00277578 | -0.00109188 | 0.0109993 | -0.00543951 | 3/0/6 |
| latent_residual_flow | no | 0.938645 | [0.91086, 0.964502] | -0.00374986 | -0.00206595 | 0.0100252 | -0.00641358 | 4/0/5 |
| lc_residual_direct | yes | 0.936935 | [0.905984, 0.965665] | -0.0054596 | -0.00377569 | 0.00831547 | -0.00812332 | 2/0/7 |
| dba_sameclass | no | 0.936623 | [0.904601, 0.96476] | -0.00577167 | -0.00408777 | 0.00800339 | -0.00843539 | 3/0/6 |
| lc_latent_residual_flow | no | 0.934725 | [0.907471, 0.961521] | -0.00767046 | -0.00598656 | 0.0061046 | -0.0103342 | 3/0/6 |
| csta_template_random_within_bank | no | 0.92862 | [0.904124, 0.954133] | -0.0137751 | -0.0120912 | 0 | -0.0164388 | 0/0/9 |

## LC Diagnostics
| Method | Valid Rate | No-valid Fallback | Wrong Mass | Bad Margin | Eff Support | Top1 Mass | Gen Cos | Eff Mult |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lc_residual_direct | 0.981459 | 0.0174897 | 0.0174897 | 0.0174897 | 19.903 | 0.0935308 | 0.522373 | 7.85195 |
| lc_latent_residual_flow | 0.981459 | 0.0174897 | 0.0174897 | 0.0174897 | 19.903 | 0.0935308 | 0.114214 | 10 |

## Gate Notes
- Wrong-pred mass: 0.0174897; bad-margin mass: 0.0174897.
- Deltas: vs task-guided=-0.0103342, vs latent=-0.0039206, vs U5=-0.00767046, vs random=-0.00598656.