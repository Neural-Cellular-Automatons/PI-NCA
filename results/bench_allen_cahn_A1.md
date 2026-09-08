### allen_cahn  (grid=48, train_steps=12, eval_steps=48, epochs=1200, seeds=5, clip=None)

| metric | multiscale_flux_nca | bounded_multiscale_nca |
|---|---|---|
| rel-L2 ↓ | **4.941e-02±1.6e-04** | 4.947e-02±2.0e-04 |
| MSE ↓ | **2.345e-03±1.6e-05** | 2.351e-03±1.9e-05 |
| RMSE ↓ | **4.843e-02±1.6e-04** | 4.849e-02±2.0e-04 |
| MAE ↓ | **3.863e-02±2.0e-04** | 3.868e-02±1.9e-04 |
| L∞ ↓ | **1.911e-01±1.1e-02** | 1.913e-01±1.2e-02 |
| PSNR(dB) ↑ | **32.3±0.028** | 32.3±0.035 |
| SSIM ↑ | **0.996±0.0023** | 0.996±0.0023 |
| hi-freq err frac ↓ | **7.244e-01±5.9e-03** | 7.250e-01±5.6e-03 |
| rel-L2 @T/4 ↓ | 3.312e-02±1.5e-04 | **3.308e-02±2.8e-04** |
| rel-L2 @T/2 ↓ | 4.422e-02±1.9e-04 | **4.417e-02±3.9e-04** |
| rel-L2 @3T/4 ↓ | 4.797e-02±1.9e-04 | **4.796e-02±3.5e-04** |
| rel-L2 @T ↓ | **4.941e-02±1.6e-04** | 4.947e-02±2.0e-04 |
| err-growth T/(T/4) ↓ | **1.49±0.003** | 1.5±0.0093 |
| mass-cons err ↓ | 9.584e-06±2.9e-06 | **8.607e-06±1.6e-06** |
| periodic-BC res ↓ | 1.009e+00±2.0e-02 | **1.009e+00±1.8e-02** |
| grad-energy | 9.595e-01±6.1e-03 | 9.585e-01±8.3e-03 |
| params ↓ | **5520** | 5520 |
| train wall(s) ↓ | **64±0.41** | 64.6±0.23 |
| infer s/step ↓ | **1.395e-03±1.0e-04** | 1.421e-03±4.8e-05 |
| throughput cells/s ↑ | **1.33e+07** | 1.30e+07 |

### allen_cahn - rel-L2 with uncertainty and paired tests

n = 40 paired evaluations (5 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **multiscale_flux_nca** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs multiscale_flux_nca (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| multiscale_flux_nca | 4.9401e-02 | [4.920e-02, 4.960e-02] | 4.9317e-02 | - | - | **reference** |
| bounded_multiscale_nca | 4.9465e-02 | [4.926e-02, 4.967e-02] | 4.9368e-02 | +6.326e-05 | 0.0021 | worse |
