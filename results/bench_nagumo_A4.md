### nagumo  (grid=48, train_steps=12, eval_steps=48, epochs=1200, seeds=5, clip=None)

| metric | abl_flux | abl_residual |
|---|---|---|
| rel-L2 ↓ | 3.792e-01±7.4e-04 | **1.605e-02±4.7e-03** |
| MSE ↓ | 9.428e-02±6.9e-04 | **1.803e-04±1.0e-04** |
| RMSE ↓ | 3.071e-01±1.1e-03 | **1.299e-02±3.8e-03** |
| MAE ↓ | 2.998e-01±1.1e-03 | **8.279e-03±2.1e-03** |
| L∞ ↓ | 5.327e-01±1.3e-02 | **8.856e-02±2.1e-02** |
| PSNR(dB) ↑ | 8.43±0.31 | **36.2±2.6** |
| SSIM ↑ | 0.803±0.0029 | **0.997±0.0018** |
| hi-freq err frac ↓ | **4.105e-03±2.6e-04** | 5.883e-02±1.7e-02 |
| rel-L2 @T/4 ↓ | 1.120e-01±3.4e-04 | **3.353e-03±4.0e-04** |
| rel-L2 @T/2 ↓ | 2.164e-01±4.2e-04 | **4.808e-03±5.0e-04** |
| rel-L2 @3T/4 ↓ | 3.066e-01±4.8e-04 | **7.497e-03±1.5e-03** |
| rel-L2 @T ↓ | 3.792e-01±7.4e-04 | **1.605e-02±4.7e-03** |
| err-growth T/(T/4) ↓ | **3.39±0.012** | 4.76±1.2 |
| mass-cons err ↓ | **1.831e-05±2.5e-05** | 7.075e+02±6.4e+00 |
| periodic-BC res ↓ | 8.164e-02±2.0e-03 | **5.703e-02±3.1e-03** |
| grad-energy | 1.568e-02±6.0e-04 | 9.596e-03±4.7e-04 |
| params ↓ | 4576 | **4544** |
| train wall(s) ↓ | **14.3±0.17** | 14.4±0.23 |
| infer s/step ↓ | 1.264e-03±4.4e-05 | **1.248e-03±7.1e-05** |
| throughput cells/s ↑ | 1.46e+07 | **1.48e+07** |

### nagumo - rel-L2 with uncertainty and paired tests

n = 40 paired evaluations (5 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_residual** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_residual (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_residual | 1.5957e-02 | [1.464e-02, 1.733e-02] | 1.5707e-02 | - | - | **reference** |
| abl_flux | 3.7922e-01 | [3.784e-01, 3.801e-01] | 3.7936e-01 | +3.633e-01 | 1.8e-12 | worse |
