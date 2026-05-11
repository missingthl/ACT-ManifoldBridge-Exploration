# CSTA-U5 vs Covariance-State Controls: Statistical Report

**Status**: OFFICIAL — based on locked Final20 experiment outputs.
**Date**: 2026-05-05
**Config**: csta_topk_uniform_top5, gamma=0.1, eta_safe=0.75, resnet1d, seeds 1/2/3

---

## 1. CSTA-U5 vs Random Covariance-State

| Metric | Value |
| :--- | :--- |
| Mean F1 (CSTA) | 0.7279 |
| Mean F1 (Random Cov) | 0.7229 |
| Mean Delta | 0.0050 |
| Median Delta | 0.0004 |
| Dataset W / T / L | 11 / 2 / 7 |
| Seed W / T / L | 30 / 5 / 25 |
| Bootstrap CI (95%) | [-0.0062, 0.0178] |
| CI crosses zero | **True** |
| Wilcoxon (dataset) p | 0.5461 |
| Wilcoxon (seed) p | 0.6814 |

**CI crosses zero. The improvement over random_cov_state is not statistically significant.**

Datasets where CSTA < random_cov: ['handmovementdirection', 'japanesevowels', 'libras', 'motorimagery', 'pendigits', 'selfregulationscp2', 'uwavegesturelibrary']

## 2. CSTA-U5 vs PCA Covariance-State

| Metric | Value |
| :--- | :--- |
| Mean F1 (CSTA) | 0.7279 |
| Mean F1 (PCA Cov) | 0.7221 |
| Mean Delta | 0.0058 |
| Median Delta | 0.0005 |
| Dataset W / T / L | 11 / 1 / 8 |
| Seed W / T / L | 32 / 3 / 25 |
| Bootstrap CI (95%) | [-0.0077, 0.0215] |
| CI crosses zero | **True** |
| Wilcoxon (dataset) p | 0.8405 |
| Wilcoxon (seed) p | 0.5592 |

**CI crosses zero. The improvement over pca_cov_state is not statistically significant.**

Datasets where CSTA < pca_cov: ['cricket', 'ering', 'handmovementdirection', 'libras', 'motorimagery', 'pendigits', 'selfregulationscp2', 'uwavegesturelibrary']

## 3. Paper Wording Guardrails

### For CSTA vs random_cov:
- CSTA-U5 achieves a consistent mean improvement over random covariance-state directions, but the improvement is not statistically significant across all datasets.
- Covariance-state perturbation itself is a strong augmentation baseline; PIA adds structure and auditability rather than a strictly dominant performance advantage.

### For CSTA vs pca_cov:
- CSTA-U5 achieves a consistent mean improvement over PCA covariance-state directions, but the improvement is not statistically significant across all datasets.

## 4. Per-Dataset Summary

| Dataset | CSTA F1 | Rand F1 | PCA F1 | Δ vs Rand | Δ vs PCA |
| :--- | :--- | :--- | :--- | :--- | :--- |
| articularywordrecognition | 0.9798 | 0.9654 | 0.9687 | +0.0144 | +0.0110 |
| atrialfibrillation | 0.2685 | 0.1722 | 0.1126 | +0.0963 | +0.1559 |
| basicmotions | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| cricket | 0.9814 | 0.9814 | 0.9860 | -0.0000 | -0.0046 |
| epilepsy | 0.9715 | 0.9706 | 0.9668 | +0.0009 | +0.0047 |
| ering | 0.8205 | 0.7878 | 0.8502 | +0.0327 | -0.0297 |
| ethanolconcentration | 0.2742 | 0.2707 | 0.2457 | +0.0035 | +0.0285 |
| fingermovements | 0.5289 | 0.5090 | 0.5173 | +0.0199 | +0.0116 |
| handmovementdirection | 0.2724 | 0.2903 | 0.2977 | -0.0180 | -0.0253 |
| handwriting | 0.4681 | 0.4644 | 0.4674 | +0.0037 | +0.0006 |
| har | 0.9554 | 0.9504 | 0.9527 | +0.0049 | +0.0027 |
| heartbeat | 0.6554 | 0.6279 | 0.5920 | +0.0276 | +0.0634 |
| japanesevowels | 0.9785 | 0.9840 | 0.9702 | -0.0054 | +0.0084 |
| libras | 0.8675 | 0.8864 | 0.8941 | -0.0189 | -0.0267 |
| motorimagery | 0.4487 | 0.4633 | 0.4844 | -0.0146 | -0.0357 |
| natops | 0.9609 | 0.9535 | 0.9513 | +0.0073 | +0.0096 |
| pendigits | 0.9852 | 0.9878 | 0.9879 | -0.0026 | -0.0026 |
| racketsports | 0.8878 | 0.8846 | 0.8799 | +0.0031 | +0.0079 |
| selfregulationscp2 | 0.4638 | 0.4899 | 0.5129 | -0.0261 | -0.0491 |
| uwavegesturelibrary | 0.7900 | 0.8188 | 0.8047 | -0.0288 | -0.0147 |

## 5. Key Conclusion
Both comparisons have bootstrap CIs crossing zero. PIA/UniformTop5 provides a mean advantage over generic covariance-state perturbations but the dataset-level behavior is mixed. Covariance-state augmentation is already a strong augmentation space; PIA contributes structure and auditability within this space.
