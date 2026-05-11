# RN-ECL-SPG-PIA Report

`rn_ecl_spg_pia_zhead_deterministic` is diagnostic only. SPG reference cosine is 0.991163; ECL reference cosine is 0.578470.
## Leaderboard
| Method | Probe? | Mean F1 | 95% CI | Delta U5 | Delta Random | Delta Bank | Seed W/T/L vs U5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| task_guided_latent_residual_flow | no | 0.945059 | [0.919481, 0.969917] | 0.00266372 | 0.00434763 | 0.0164388 | 5/0/4 |
| csta_topk_uniform_top5 | no | 0.942395 | [0.916886, 0.967144] | 0 | 0.0016839 | 0.0137751 | 0/9/0 |
| spg_pia_zhead | no | 0.942051 | [0.915224, 0.966829] | -0.000344426 | 0.00133948 | 0.0134306 | 4/0/5 |
| ecl_spg_pia_zhead | no | 0.941411 | [0.908478, 0.971167] | -0.000983622 | 0.00070028 | 0.0127914 | 5/0/4 |
| random_cov_state | no | 0.940711 | [0.913487, 0.96771] | -0.0016839 | 0 | 0.0120912 | 5/1/3 |
| wdba_sameclass | no | 0.939619 | [0.91898, 0.960911] | -0.00277578 | -0.00109188 | 0.0109993 | 3/0/6 |
| rn_ecl_spg_pia_zhead | no | 0.938618 | [0.912167, 0.965181] | -0.00377684 | -0.00209294 | 0.00999822 | 3/0/6 |
| dba_sameclass | no | 0.936623 | [0.906299, 0.965814] | -0.00577167 | -0.00408777 | 0.00800339 | 3/0/6 |
| csta_template_random_within_bank | no | 0.92862 | [0.904078, 0.955035] | -0.0137751 | -0.0120912 | 0 | 0/0/9 |

## SPG/ECL Diagnostics
| Method | z-head acc | Proj Energy | ECL alpha | ECL align | Dir Cos | Eff Mult | Support Rank | Bridge | Safe Clip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spg_pia_zhead | 0.923148 | 0.730119 | n/a | n/a | 0.991163 | 10 | 10 | 1 | 0 |
| ecl_spg_pia_zhead | 0.923148 | 0.579015 | 1.12249 | 0.72574 | 0.57847 | 10 | 10 | 1 | 0 |
| rn_ecl_spg_pia_zhead | 0.923148 | 0.58841 | 0.364978 | 0.924931 | 0.866567 | 10 | 10 | 1 | 0 |