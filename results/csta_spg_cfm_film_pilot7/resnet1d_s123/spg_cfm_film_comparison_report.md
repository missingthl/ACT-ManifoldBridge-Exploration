# SPG-CFM Align & FiLM Architecture Validation Report (Pilot 7)

## Executive Summary
Comparing Align-based Loss vs FiLM-based injection vs Concat-based injection.

### Leaderboard (Mean F1)
                            mean       std  count
method                                           
wdba_sameclass          0.667922  0.281589     21
csta_topk_uniform_top5  0.665242  0.304897     21
dba_sameclass           0.663309  0.300305     21
spg_cfm_align_one_step  0.637667  0.335035     21

### Comparison vs U5 (W/T/L)
                   method   mean_f1  win  tie  loss  score
1          wdba_sameclass  0.667922    8    3    10     -2
2           dba_sameclass  0.663309    5    8     8     -3
0  spg_cfm_align_one_step  0.637667    5    5    11     -6

## Dataset Level Analysis
method                 csta_topk_uniform_top5  dba_sameclass  spg_cfm_align_one_step  wdba_sameclass
dataset                                                                                             
atrialfibrillation                   0.268543       0.258951                0.162911        0.311276
ering                                0.820515       0.833818                0.821233        0.772643
handmovementdirection                0.272393       0.352625                0.246976        0.369405
handwriting                          0.468056       0.387899                0.420686        0.403273
japanesevowels                       0.978534       0.980085                0.985571        0.974855
natops                               0.960870       0.957282                0.956706        0.947145
racketsports                         0.887781       0.872504                0.869585        0.896858
