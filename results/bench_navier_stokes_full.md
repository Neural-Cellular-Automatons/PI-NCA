### navier_stokes  (grid=16, train_steps=6, eval_steps=12, epochs=40, seeds=2, clip=None)

| metric | plain_nca | pi_nca | resnet_iso | identity |
|---|---|---|---|---|
| rel-L2 ↓ | 1.983e-01±3.7e-02 | 1.918e-01±3.9e-02 | **1.902e-01±4.0e-02** | 1.970e-01±3.7e-02 |
| MSE ↓ | 1.266e-02±6.0e-03 | 1.189e-02±5.9e-03 | **1.172e-02±6.0e-03** | 1.249e-02±5.9e-03 |
| RMSE ↓ | 1.109e-01±2.7e-02 | 1.073e-01±2.8e-02 | **1.064e-01±2.8e-02** | 1.101e-01±2.7e-02 |
| MAE ↓ | 5.373e-02±7.3e-03 | 5.340e-02±7.6e-03 | **5.298e-02±8.8e-03** | 5.300e-02±7.2e-03 |
| L∞ ↓ | 7.915e-01±2.8e-01 | 8.298e-01±3.2e-01 | 8.310e-01±3.0e-01 | **7.768e-01±2.8e-01** |
| PSNR(dB) ↑ | 34.1±0.15 | 34.4±0.023 | **34.5±0.022** | 34.2±0.17 |
| SSIM ↑ | 0.784±0.22 | **0.981±0.0075** | 0.919±0.081 | 0.981±0.0071 |
| hi-freq err frac ↓ | 9.795e-03±2.7e-03 | 2.063e-02±1.1e-02 | 1.668e-02±1.3e-03 | **7.281e-03±6.6e-04** |
| rel-L2 @T/4 ↓ | 5.097e-02±9.6e-03 | 4.982e-02±9.8e-03 | **4.897e-02±1.0e-02** | 5.028e-02±9.5e-03 |
| rel-L2 @T/2 ↓ | 1.012e-01±1.9e-02 | 9.859e-02±2.0e-02 | **9.717e-02±2.0e-02** | 1.001e-01±1.9e-02 |
| rel-L2 @3T/4 ↓ | 1.505e-01±2.8e-02 | 1.460e-01±2.9e-02 | **1.443e-01±3.0e-02** | 1.491e-01±2.8e-02 |
| rel-L2 @T ↓ | 1.983e-01±3.7e-02 | 1.918e-01±3.9e-02 | **1.902e-01±4.0e-02** | 1.970e-01±3.7e-02 |
| err-growth T/(T/4) ↓ | 3.89±0.0025 | **3.85±0.026** | 3.88±0.012 | 3.92±0.0079 |
| mass-cons err ↓ | 1.235e+00±1.0e+00 | 6.020e-06±1.7e-06 | 7.337e-01±2.8e-01 | **0.000e+00±0.0e+00** |
| periodic-BC res ↓ | 1.724e-01±2.6e-02 | **1.697e-01±2.4e-02** | 1.755e-01±2.7e-02 | 1.772e-01±2.7e-02 |
| grad-energy | 1.131e-01±1.8e-02 | 1.086e-01±1.7e-02 | 1.105e-01±1.3e-02 | 1.180e-01±1.7e-02 |
| params ↓ | 6784 | 4576 | 5364 | **1** |
| train wall(s) ↓ | 1.57±0.11 | 1.53±0.037 | 1.85±0.12 | **0.773±0.11** |
| infer s/step ↓ | 2.521e-04±7.6e-05 | 1.841e-04±1.3e-05 | 1.029e-03±1.6e-04 | **5.775e-06±1.8e-07** |
| throughput cells/s ↑ | 8.51e+06 | 1.12e+07 | 2.02e+06 | **3.55e+08** |

### navier_stokes - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **resnet_iso** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 3 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

| architecture | rel-L2 mean | 95% CI | median | vs resnet_iso (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| resnet_iso | 1.5115e-01 | [1.211e-01, 1.812e-01] | 1.5629e-01 | - | - | **reference** |
| pi_nca | 1.5411e-01 | [1.242e-01, 1.835e-01] | 1.5351e-01 | +2.959e-03 | 0.14 | tie |
| identity | 1.5562e-01 | [1.227e-01, 1.875e-01] | 1.6346e-01 | +4.470e-03 | 0.051 | tie |
| plain_nca | 1.5778e-01 | [1.252e-01, 1.897e-01] | 1.6321e-01 | +6.637e-03 | 0.013 | worse |

Statistically indistinguishable from resnet_iso at this sample size: `pi_nca`, `identity`.
