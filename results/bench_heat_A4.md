### heat  (grid=16, train_steps=6, eval_steps=12, epochs=150, seeds=2, clip=None)

| metric | abl_flux | abl_residual |
|---|---|---|
| rel-L2 ↓ | **2.414e-02±8.3e-03** | 4.598e-02±3.5e-04 |
| MSE ↓ | **2.365e-03±1.9e-03** | 7.645e-03±1.6e-03 |
| RMSE ↓ | **4.652e-02±2.0e-02** | 8.720e-02±9.0e-03 |
| MAE ↓ | **2.944e-02±1.1e-02** | 5.602e-02±4.9e-03 |
| L∞ ↓ | **2.977e-01±2.2e-01** | 5.980e-01±2.7e-01 |
| PSNR(dB) ↑ | **45.6±3.7** | 39.7±0.78 |
| SSIM ↑ | **1±0.00032** | 0.998±1.6e-05 |
| hi-freq err frac ↓ | 4.197e-01±1.5e-01 | **1.650e-01±3.8e-02** |
| rel-L2 @T/4 ↓ | **1.196e-02±2.1e-03** | 2.132e-02±2.1e-04 |
| rel-L2 @T/2 ↓ | **1.862e-02±3.9e-03** | 3.433e-02±4.2e-04 |
| rel-L2 @3T/4 ↓ | **2.214e-02±6.0e-03** | 4.193e-02±3.9e-04 |
| rel-L2 @T ↓ | **2.414e-02±8.3e-03** | 4.598e-02±3.5e-04 |
| err-growth T/(T/4) ↓ | **1.99±0.35** | 2.16±0.0046 |
| mass-cons err ↓ | **9.632e-05±2.6e-05** | 3.989e+00±1.5e+00 |
| periodic-BC res ↓ | **4.433e-01±1.3e-02** | 4.470e-01±1.3e-02 |
| grad-energy | 8.024e-01±1.6e-01 | 8.489e-01±1.6e-01 |
| params ↓ | 4576 | **4544** |
| train wall(s) ↓ | 9.56±2.6 | **8.25±0.93** |
| infer s/step ↓ | **4.677e-04±2.6e-05** | 4.775e-04±1.3e-04 |
| throughput cells/s ↑ | 4.39e+06 | **4.46e+06** |

### heat - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_flux** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_flux (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_flux | 2.4346e-02 | [2.141e-02, 2.735e-02] | 2.2031e-02 | - | - | **reference** |
| abl_residual | 4.6331e-02 | [4.362e-02, 4.897e-02] | 4.6147e-02 | +2.198e-02 | 3.1e-05 | worse |
