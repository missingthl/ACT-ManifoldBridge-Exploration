# SPG-CFM One-Step Pilot7 Report

**Target**: `spg_cfm_one_step` | **Datasets**: 7 | **Seeds**: 3 | **Total rows expected**: 168

**Actual target rows**: 21

---

## 1. Leaderboard (Mean Macro-F1)

```
                                      mean       std  count
method                                                     
wdba_sameclass                    0.667922  0.281589     21
csta_topk_uniform_top5            0.665242  0.304897     21
dba_sameclass                     0.663309  0.300305     21
spg_cfm_one_step                  0.654910  0.316051     21
csta_template_random_within_bank  0.653546  0.305522     21
random_cov_state                  0.648130  0.320643     21
latent_residual_flow              0.647500  0.320989     21
spg_pia_zhead                     0.639050  0.341855     21
```

## 2. Seed-Level W/T/L vs Baselines (with 95% Bootstrap CI on Δ)

```
                        baseline  W  T  L  mean_delta    CI_lo   CI_hi
          csta_topk_uniform_top5  9  0 12    -0.01033 -0.03580 0.01206
                random_cov_state 12  0  9     0.00678 -0.01297 0.02848
csta_template_random_within_bank 11  1  9     0.00136 -0.01746 0.01728
                   spg_pia_zhead 12  1  8     0.01586 -0.01074 0.04674
            latent_residual_flow 13  0  8     0.00741 -0.00857 0.02836
                  wdba_sameclass 10  0 11    -0.01301 -0.04275 0.01583
                   dba_sameclass 11  0 10    -0.00840 -0.03645 0.01940
```

## 3. Dataset-Level W/T/L

### vs `csta_topk_uniform_top5`

```
              dataset W/T/L
   atrialfibrillation 0/0/3
                ering 1/0/2
handmovementdirection 1/0/2
          handwriting 2/0/1
       japanesevowels 3/0/0
               natops 2/0/1
         racketsports 0/0/3
```

### vs `random_cov_state`

```
              dataset W/T/L
   atrialfibrillation 2/0/1
                ering 1/0/2
handmovementdirection 3/0/0
          handwriting 2/0/1
       japanesevowels 2/0/1
               natops 2/0/1
         racketsports 0/0/3
```

### vs `csta_template_random_within_bank`

```
              dataset W/T/L
   atrialfibrillation 1/0/2
                ering 0/1/2
handmovementdirection 2/0/1
          handwriting 2/0/1
       japanesevowels 3/0/0
               natops 3/0/0
         racketsports 0/0/3
```

### vs `spg_pia_zhead`

```
              dataset W/T/L
   atrialfibrillation 3/0/0
                ering 1/0/2
handmovementdirection 1/0/2
          handwriting 2/0/1
       japanesevowels 3/0/0
               natops 1/1/1
         racketsports 1/0/2
```

### vs `latent_residual_flow`

```
              dataset W/T/L
   atrialfibrillation 2/0/1
                ering 2/0/1
handmovementdirection 2/0/1
          handwriting 1/0/2
       japanesevowels 2/0/1
               natops 3/0/0
         racketsports 1/0/2
```

### vs `wdba_sameclass`

```
              dataset W/T/L
   atrialfibrillation 0/0/3
                ering 2/0/1
handmovementdirection 0/0/3
          handwriting 3/0/0
       japanesevowels 2/0/1
               natops 3/0/0
         racketsports 0/0/3
```

### vs `dba_sameclass`

```
              dataset W/T/L
   atrialfibrillation 1/0/2
                ering 0/0/3
handmovementdirection 1/0/2
          handwriting 3/0/0
       japanesevowels 2/0/1
               natops 2/0/1
         racketsports 2/0/1
```

## 4. Dataset Breakdown (Mean F1)

```
method                 csta_template_random_within_bank  csta_topk_uniform_top5  dba_sameclass  latent_residual_flow  random_cov_state  spg_cfm_one_step  spg_pia_zhead  wdba_sameclass
dataset                                                                                                                                                                                
atrialfibrillation                               0.2287                  0.2685         0.2590                0.1849            0.1722            0.1762         0.1193          0.3113
ering                                            0.8167                  0.8205         0.8338                0.8024            0.7878            0.8056         0.8253          0.7726
handmovementdirection                            0.2866                  0.2724         0.3526                0.2610            0.2903            0.3075         0.2681          0.3694
handwriting                                      0.4570                  0.4681         0.3879                0.4682            0.4644            0.4685         0.4345          0.4033
japanesevowels                                   0.9727                  0.9785         0.9801                0.9809            0.9840            0.9837         0.9767          0.9749
natops                                           0.9313                  0.9609         0.9573                0.9525            0.9535            0.9681         0.9646          0.9471
racketsports                                     0.8819                  0.8878         0.8725                0.8825            0.8846            0.8748         0.8848          0.8969
```

## 5. SPG-CFM Diagnostics

```
                                                  spg_cfm_one_step
spg_cfm_generated_direction_pairwise_cosine_mean          0.096724
spg_cfm_alignment_to_spg_mean                            -0.018750
spg_cfm_projection_energy_mean                            0.729878
spg_cfm_projection_energy_std                             0.079970
spg_cfm_effective_aug_multiplier                         10.000000
bridge_success_rate                                       1.000000
safe_clip_rate                                            0.028214
gamma_used_ratio_mean                                     0.993657
spg_zhead_train_acc                                       0.707068
transport_error_logeuc_mean                                    NaN
```

## 6. Speed Audit

```
                                  augmentation_build_time_sec  spg_cfm_generation_time_sec  generation_time_per_aug_sample_ms
method                                                                                                                       
csta_template_random_within_bank                          NaN                          NaN                                NaN
csta_topk_uniform_top5                                    NaN                          NaN                                NaN
dba_sameclass                                             NaN                          NaN                                NaN
latent_residual_flow                                      NaN                          NaN                                NaN
random_cov_state                                          NaN                          NaN                                NaN
spg_cfm_one_step                                     0.917403                     0.367289                            0.33594
spg_pia_zhead                                             NaN                          NaN                                NaN
wdba_sameclass                                            NaN                          NaN                                NaN
```

```
relative_speed_vs_wdba    N/A (wdba no time)
relative_speed_vs_u5        N/A (u5 no time)
```

