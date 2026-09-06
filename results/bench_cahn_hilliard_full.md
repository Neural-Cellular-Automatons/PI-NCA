### cahn_hilliard  (grid=16, train_steps=6, eval_steps=12, epochs=40, seeds=2, clip=None)

| metric | plain_nca | pi_nca | resnet_iso | identity |
|---|---|---|---|---|
| rel-L2 ↓ | 1.330e+00±5.1e-02 | 1.241e+00±1.5e-02 | 1.373e+00±4.2e-02 | **9.277e-01±4.4e-04** |
| MSE ↓ | 1.241e+00±9.7e-02 | 1.080e+00±2.9e-02 | 1.322e+00±8.4e-02 | **6.031e-01±1.7e-03** |
| RMSE ↓ | 1.114e+00±4.3e-02 | 1.039e+00±1.4e-02 | 1.149e+00±3.7e-02 | **7.766e-01±1.1e-03** |
| MAE ↓ | 8.850e-01±3.6e-02 | 8.265e-01±9.2e-03 | 9.167e-01±2.8e-02 | **7.305e-01±3.8e-03** |
| L∞ ↓ | 2.375e+00±0.0e+00 | 2.375e+00±0.0e+00 | 2.375e+00±0.0e+00 | **1.233e+00±4.4e-02** |
| PSNR(dB) ↑ | 5.09±0.34 | 5.69±0.11 | 4.81±0.28 | **8.22±0.012** |
| SSIM ↑ | -0.0148±0.21 | **0.29±0.28** | -0.147±0.087 | 0.142±0.008 |
| hi-freq err frac ↓ | 8.830e-01±5.8e-03 | 8.587e-01±2.9e-03 | 8.516e-01±1.7e-02 | **8.462e-01±2.2e-02** |
| rel-L2 @T/4 ↓ | 7.979e-01±8.1e-03 | **7.870e-01±1.3e-02** | 8.140e-01±9.4e-04 | 9.027e-01±1.4e-03 |
| rel-L2 @T/2 ↓ | 7.730e-01±9.9e-03 | **7.392e-01±2.1e-02** | 7.808e-01±1.1e-02 | 9.271e-01±1.4e-03 |
| rel-L2 @3T/4 ↓ | 1.039e+00±3.0e-02 | 9.869e-01±5.7e-03 | 1.080e+00±6.6e-02 | **9.291e-01±8.5e-04** |
| rel-L2 @T ↓ | 1.330e+00±5.1e-02 | 1.241e+00±1.5e-02 | 1.373e+00±4.2e-02 | **9.277e-01±4.4e-04** |
| err-growth T/(T/4) ↓ | 1.67±0.046 | 1.58±0.0064 | 1.69±0.05 | **1.03±0.0012** |
| mass-cons err ↓ | 1.210e+01±1.4e+00 | 1.440e+01±1.4e+01 | 1.947e+01±1.3e+01 | **0.000e+00±0.0e+00** |
| periodic-BC res ↓ | 1.354e+00±1.8e-01 | 1.346e+00±2.1e-01 | 1.318e+00±9.8e-02 | **8.568e-02±4.1e-03** |
| grad-energy | 7.071e-01±5.0e-02 | 9.898e-01±4.6e-02 | 9.004e-01±1.5e-01 | 1.222e-02±6.2e-05 |
| params ↓ | 6784 | 4576 | 5364 | **1** |
| train wall(s) ↓ | 1.87±0.12 | 1.8±0.056 | 2.1±0.16 | **1.1±0.27** |
| infer s/step ↓ | 2.169e-04±4.6e-05 | 1.963e-04±2.7e-05 | 9.488e-04±1.4e-04 | **6.300e-06±1.1e-06** |
| throughput cells/s ↑ | 9.65e+06 | 1.05e+07 | 2.18e+06 | **3.30e+08** |

### cahn_hilliard - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **identity** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 3 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

| architecture | rel-L2 mean | 95% CI | median | vs identity (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| identity | 9.2758e-01 | [9.233e-01, 9.320e-01] | 9.2668e-01 | - | - | **reference** |
| pi_nca | 1.2404e+00 | [1.215e+00, 1.265e+00] | 1.2451e+00 | +3.128e-01 | 3.1e-05 | worse |
| plain_nca | 1.3292e+00 | [1.300e+00, 1.360e+00] | 1.3119e+00 | +4.016e-01 | 3.1e-05 | worse |
| resnet_iso | 1.3719e+00 | [1.339e+00, 1.403e+00] | 1.3816e+00 | +4.444e-01 | 3.1e-05 | worse |
