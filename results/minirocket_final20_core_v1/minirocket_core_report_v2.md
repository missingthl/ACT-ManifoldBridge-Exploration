# MiniRocket Core Performance (Strict Subset Audit)

- **Subset Scope**: 18 / 20 datasets
- **Missing Datasets**: japanesevowels, pendigits
- **Actual N (Aligned Pairs)**: 54
- **Expected N (for completed subset)**: 54

## Performance (CSTA-U5 vs No-Aug)
- **Mean Delta F1**: +0.0100
- **W/T/L (Seed-level)**: 25 Win / 14 Tie / 15 Loss
  *(Audit check: 25 + 14 + 15 = 54)

### Dataset-level Summary
| Dataset | Mean Delta F1 |
| :--- | :--- |
| handwriting | +0.0840 |
| fingermovements | +0.0367 |
| motorimagery | +0.0310 |
| selfregulationscp2 | +0.0190 |
| natops | +0.0163 |
| racketsports | +0.0120 |
| uwavegesturelibrary | +0.0108 |
| handmovementdirection | +0.0091 |
| heartbeat | +0.0042 |
| libras | +0.0029 |
| epilepsy | +0.0000 |
| cricket | +0.0000 |
| basicmotions | +0.0000 |
| articularywordrecognition | -0.0000 |
| ering | -0.0013 |
| ethanolconcentration | -0.0041 |
| har | -0.0111 |
| atrialfibrillation | -0.0302 |
