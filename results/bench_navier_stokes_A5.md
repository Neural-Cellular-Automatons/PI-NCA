### navier_stokes  (grid=16, train_steps=6, eval_steps=12, epochs=150, seeds=2, clip=None)

| metric | abl_k3 | abl_k5 | abl_multiscale |
|---|---|---|---|
| rel-L2 ↓ | 1.716e-01±3.7e-02 | 1.394e-01±2.8e-02 | **1.102e-01±2.1e-02** |
| MSE ↓ | 9.565e-03±5.0e-03 | 6.290e-03±3.2e-03 | **3.908e-03±1.8e-03** |
| RMSE ↓ | 9.602e-02±2.6e-02 | 7.801e-02±2.0e-02 | **6.162e-02±1.5e-02** |
| MAE ↓ | 5.012e-02±7.8e-03 | 4.506e-02±5.6e-03 | **3.716e-02±3.4e-03** |
| L∞ ↓ | 7.243e-01±2.5e-01 | 5.322e-01±2.3e-01 | **4.861e-01±1.6e-01** |
| PSNR(dB) ↑ | 35.4±0.12 | 37.2±0.0093 | **39.2±0.17** |
| SSIM ↑ | 0.985±0.0065 | 0.99±0.004 | **0.994±0.0023** |
| hi-freq err frac ↓ | **9.909e-02±1.2e-02** | 1.761e-01±8.4e-02 | 2.325e-01±8.0e-02 |
| rel-L2 @T/4 ↓ | 4.379e-02±8.9e-03 | 3.523e-02±7.0e-03 | **2.847e-02±5.7e-03** |
| rel-L2 @T/2 ↓ | 8.669e-02±1.8e-02 | 6.999e-02±1.4e-02 | **5.619e-02±1.1e-02** |
| rel-L2 @3T/4 ↓ | 1.291e-01±2.7e-02 | 1.046e-01±2.1e-02 | **8.335e-02±1.6e-02** |
| rel-L2 @T ↓ | 1.716e-01±3.7e-02 | 1.394e-01±2.8e-02 | **1.102e-01±2.1e-02** |
| err-growth T/(T/4) ↓ | 3.91±0.064 | 3.96±0.019 | **3.88±0.05** |
| mass-cons err ↓ | 8.205e-06±1.6e-06 | 7.365e-06±7.1e-07 | **6.628e-06±3.7e-06** |
| periodic-BC res ↓ | **1.769e-01±1.8e-02** | 1.808e-01±1.5e-02 | 1.806e-01±1.8e-02 |
| grad-energy | 1.115e-01±1.6e-02 | 1.131e-01±1.6e-02 | 1.138e-01±1.6e-02 |
| params ↓ | **4576** | 5088 | 9312 |
| train wall(s) ↓ | **10.5±2.2** | 14.4±2.2 | 20.3±1.3 |
| infer s/step ↓ | **4.737e-04±6.6e-05** | 5.995e-04±3.5e-05 | 1.842e-03±2.1e-04 |
| throughput cells/s ↑ | **4.37e+06** | 3.42e+06 | 1.12e+06 |

### navier_stokes - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_multiscale** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 2 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_multiscale (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_multiscale | 9.6104e-02 | [8.395e-02, 1.089e-01] | 9.1158e-02 | - | - | **reference** |
| abl_k5 | 1.2222e-01 | [1.063e-01, 1.387e-01] | 1.1455e-01 | +2.611e-02 | 3.1e-05 | worse |
| abl_k3 | 1.4430e-01 | [1.212e-01, 1.674e-01] | 1.3816e-01 | +4.820e-02 | 6.1e-05 | worse |
