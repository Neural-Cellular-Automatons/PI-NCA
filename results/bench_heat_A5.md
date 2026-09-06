### heat  (grid=16, train_steps=6, eval_steps=12, epochs=150, seeds=2, clip=None)

| metric | abl_k3 | abl_k5 | abl_multiscale |
|---|---|---|---|
| rel-L2 ↓ | **2.414e-02±8.3e-03** | 4.976e-02±2.4e-03 | 3.693e-02±2.6e-03 |
| MSE ↓ | **2.365e-03±1.9e-03** | 8.868e-03±8.3e-04 | 4.871e-03±2.4e-04 |
| RMSE ↓ | **4.652e-02±2.0e-02** | 9.412e-02±4.4e-03 | 6.978e-02±1.7e-03 |
| MAE ↓ | **2.944e-02±1.1e-02** | 5.712e-02±6.3e-03 | 4.771e-02±1.9e-03 |
| L∞ ↓ | **2.977e-01±2.2e-01** | 6.976e-01±2.7e-01 | 3.771e-01±2.4e-02 |
| PSNR(dB) ↑ | **45.6±3.7** | 39±0.29 | 41.6±0.089 |
| SSIM ↑ | **1±0.00032** | 0.998±0.00019 | 0.999±0.00015 |
| hi-freq err frac ↓ | 4.197e-01±1.5e-01 | 3.601e-01±6.7e-02 | **3.489e-01±1.8e-02** |
| rel-L2 @T/4 ↓ | **1.196e-02±2.1e-03** | 2.420e-02±1.2e-03 | 1.932e-02±7.4e-04 |
| rel-L2 @T/2 ↓ | **1.862e-02±3.9e-03** | 3.840e-02±1.9e-03 | 2.996e-02±1.5e-03 |
| rel-L2 @3T/4 ↓ | **2.214e-02±6.0e-03** | 4.604e-02±2.2e-03 | 3.508e-02±2.1e-03 |
| rel-L2 @T ↓ | **2.414e-02±8.3e-03** | 4.976e-02±2.4e-03 | 3.693e-02±2.6e-03 |
| err-growth T/(T/4) ↓ | 1.99±0.35 | 2.06±0.0032 | **1.91±0.064** |
| mass-cons err ↓ | 9.632e-05±2.6e-05 | **7.343e-05±6.7e-06** | 8.106e-05±1.5e-05 |
| periodic-BC res ↓ | **4.433e-01±1.3e-02** | 4.498e-01±1.2e-02 | 4.478e-01±9.6e-03 |
| grad-energy | 8.024e-01±1.6e-01 | 8.378e-01±1.6e-01 | 8.240e-01±1.5e-01 |
| params ↓ | **4576** | 5088 | 9312 |
| train wall(s) ↓ | 7.92±1.1 | **7.55±0.95** | 11.1±2.2 |
| infer s/step ↓ | 5.575e-04±1.8e-05 | **4.914e-04±1.6e-04** | 1.223e-03±4.9e-04 |
| throughput cells/s ↑ | 3.68e+06 | **4.41e+06** | 1.82e+06 |

### heat - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_k3** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 2 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_k3 (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_k3 | 2.4346e-02 | [2.141e-02, 2.735e-02] | 2.2031e-02 | - | - | **reference** |
| abl_multiscale | 3.7805e-02 | [3.582e-02, 3.985e-02] | 3.7038e-02 | +1.346e-02 | 9.2e-05 | worse |
| abl_k5 | 5.0081e-02 | [4.654e-02, 5.385e-02] | 4.9397e-02 | +2.573e-02 | 3.1e-05 | worse |
