# Local Tangent Audit Report

This is a post-hoc diagnostic for CSTA/PIA template directions. It does not change training or augmentation outputs.

## Coverage

- Per-seed rows: 4
- Dataset-summary rows: 4
- Overall rows: 8

## Overall Alignment

```csv
method,direction_source,mean_tangent_alignment,mean_normal_leakage,n_datasets,n_seeds,available_rate,mean_selected_minus_random,mean_selected_minus_pca
csta_top1_current,pca_cov,0.09082662148973483,0.909173378510265,1,1,1.0,0.036112031433371,-0.0412812526262826
csta_top1_current,pia_selected,0.04954536886345217,0.9504546311365478,1,1,1.0,0.036112031433371,-0.0412812526262826
csta_top1_current,random_cov,0.013433337430081132,0.9865666625699188,1,1,1.0,0.036112031433371,-0.0412812526262826
csta_topk_uniform_top5,pca_cov,0.09082662148973483,0.909173378510265,1,1,1.0,0.0431663186255958,-0.0342269654340578
csta_topk_uniform_top5,pia_selected,0.05659965605567704,0.9434003439443229,1,1,1.0,0.0431663186255958,-0.0342269654340578
csta_topk_uniform_top5,random_cov,0.013433337430081132,0.9865666625699188,1,1,1.0,0.0431663186255958,-0.0342269654340578
pca_cov_state,pca_cov,0.09082662148973483,0.909173378510265,1,1,1.0,,
random_cov_state,random_cov,0.013433337430081132,0.9865666625699188,1,1,1.0,,
```
