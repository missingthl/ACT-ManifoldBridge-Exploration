# ModernTCN Final20 Robustness Audit & Report

## 1. Integrity Audit Results
- **Total Rows**: 120 (Expected: 120 + header)
- **Uniqueness**: PASS (Duplicates: 0)
- **Missing Rows**: 0 items missing
- **Backbone Check**: PASS (Found: [])
- **Arms Check**: PASS (Found: [])
- **Status Check**: FAIL (Failed: 3)
  - Failed Rows: [{'dataset': 'motorimagery', 'seed': 1, 'method': 'csta_topk_uniform_top5', 'fail_reason': 'csta_topk_uniform_top5 did not produce a success row in /home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/results/moderntcn_final20_robustness_v1/_csta_runs/csta_topk_uniform_top5/motorimagery/s1/motorimagery_results.csv'}, {'dataset': 'motorimagery', 'seed': 2, 'method': 'csta_topk_uniform_top5', 'fail_reason': 'csta_topk_uniform_top5 did not produce a success row in /home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/results/moderntcn_final20_robustness_v1/_csta_runs/csta_topk_uniform_top5/motorimagery/s2/motorimagery_results.csv'}, {'dataset': 'motorimagery', 'seed': 3, 'method': 'csta_topk_uniform_top5', 'fail_reason': 'csta_topk_uniform_top5 did not produce a success row in /home/THL/project/MTS-PIA/standalone_projects/ACT_ManifoldBridge/results/moderntcn_final20_robustness_v1/_csta_runs/csta_topk_uniform_top5/motorimagery/s3/motorimagery_results.csv'}]
- **NaN Check**: FAIL (NaN F1: 3)
- **Hparam Audit**: PASS

---

## 2. Core Performance Analysis (CSTA-U5 vs No-Aug)
Comparison Pairs: 60 (20 datasets x 3 seeds)

- **Mean Delta (F1)**: +0.0830
- **W/T/L (Threshold 0.001)**: 47 / 1 / 9

### Dataset-level Performance Breakdown (Mean over 3 seeds)

| Dataset | Mean Delta F1 |
| :--- | :--- |
| ering | +0.4294 |
| cricket | +0.3872 |
| libras | +0.1565 |
| racketsports | +0.1038 |
| fingermovements | +0.0724 |
| basicmotions | +0.0672 |
| heartbeat | +0.0650 |
| atrialfibrillation | +0.0602 |
| handwriting | +0.0513 |
| epilepsy | +0.0475 |
| uwavegesturelibrary | +0.0334 |
| articularywordrecognition | +0.0282 |
| selfregulationscp2 | +0.0249 |
| natops | +0.0240 |
| har | +0.0123 |
| ethanolconcentration | +0.0084 |
| japanesevowels | +0.0038 |
| handmovementdirection | +0.0005 |
| pendigits | +0.0003 |
| motorimagery | +nan |
