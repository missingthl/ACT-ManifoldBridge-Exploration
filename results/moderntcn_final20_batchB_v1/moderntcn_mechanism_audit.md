# ModernTCN Mechanism Audit (Strict Shared-Pair)

- **Shared Pair N**: 60
- **Datasets**: 20

## 1. Overall Performance Comparison
| Arm | Mean Delta (F1) | W/T/L (vs no_aug) |
| :--- | :--- | :--- |
| CSTA-U5 | +0.0696 | 46 / 2 / 12 |
| Random Cov | +0.0628 | 45 / 1 / 14 |
| PCA Cov | +0.0551 | 41 / 6 / 13 |

## 2. Direct Head-to-Head (W/T/L vs CSTA)
- CSTA-U5 vs Random Cov: 31 / 10 / 19
- CSTA-U5 vs PCA Cov: 35 / 7 / 18

## 3. Dataset-level Detail
| Dataset | Delta CSTA | Delta Random | Delta PCA |
| :--- | :--- | :--- | :--- |
| cricket | +0.3872 | +0.3919 | +0.3919 |
| ering | +0.2570 | +0.2409 | +0.2409 |
| libras | +0.1565 | +0.1555 | +0.1442 |
| motorimagery | +0.1481 | +0.1519 | +0.0119 |
| racketsports | +0.1038 | +0.0746 | +0.0766 |
| fingermovements | +0.0724 | +0.0476 | +0.0617 |
| basicmotions | +0.0672 | +0.0672 | +0.0672 |
| heartbeat | +0.0650 | +0.0074 | +0.0614 |
| atrialfibrillation | +0.0602 | +0.0221 | +0.0034 |
| handwriting | +0.0454 | +0.0228 | +0.0225 |
| selfregulationscp2 | +0.0249 | +0.0140 | +0.0128 |
| har | +0.0122 | +0.0154 | +0.0126 |
| epilepsy | +0.0121 | +0.0004 | +0.0003 |
| uwavegesturelibrary | +0.0100 | -0.0044 | -0.0015 |
| ethanolconcentration | +0.0084 | +0.0402 | +0.0025 |
| japanesevowels | +0.0038 | +0.0019 | +0.0054 |
| handmovementdirection | +0.0005 | +0.0232 | +0.0097 |
| pendigits | +0.0003 | +0.0019 | +0.0012 |
| articularywordrecognition | -0.0002 | -0.0035 | -0.0035 |
| natops | -0.0417 | -0.0150 | -0.0181 |


## 4. Negative Gain Datasets (CSTA)
['natops']
