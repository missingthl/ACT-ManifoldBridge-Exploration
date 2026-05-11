# CSTA Mechanism Evidence Pack

This pack combines post-hoc local tangent alignment with pilot7 performance summaries.

## Key Interpretation

- Local tangent alignment is a mechanism diagnostic, not a selector or training component.
- PIA directions should be interpreted as tangent-relevant proposal directions when they exceed random covariance directions.
- Higher alignment alone is not sufficient utility evidence; UniformTop5 may trade peak alignment for high-response neighborhood diversity.

- CSTA UniformTop5 mean F1: 0.665242; PIA selected alignment: 0.264496.
- CSTA Top1 selected alignment: 0.303621; compare this with performance to avoid overclaiming alignment as utility.

## Files

- `mechanism_main_table.csv`
- `tangent_alignment_table.csv`
- `diversity_alignment_tradeoff.csv`
- `alignment_performance_correlation.csv`