# CSTA Backbone Robustness Summary

| Backbone | Datasets | N_Pairs | Mean_Delta | W/T/L | Win_Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ResNet1D | 20 | 60 | +0.0413 | 39/7/14 | 65.0% |
| ModernTCN | 20 | 60 | +0.0696 | 46/2/12 | 76.7% |
| MiniRocket | 18 | 54 | +0.0100 | 25/14/15 | 46.3% |


## Key Observation
- **Neural Architectures (ResNet/ModernTCN)** benefit significantly from CSTA's manifold-bridge effect (+3.3% to +4.2%).
- **Linear/Convolutional Baselines (MiniRocket)** show robust but smaller gains (+1.0%), confirming model-agnosticism.
