### heat  (grid=48, train_steps=12, eval_steps=48, epochs=1200, seeds=5, clip=None)

| metric | abl_flux | abl_residual |
|---|---|---|
| rel-L2 ↓ | **7.430e-03±6.0e-03** | 3.394e-02±2.9e-02 |
| MSE ↓ | **3.792e-04±4.7e-04** | 8.245e-03±1.0e-02 |
| RMSE ↓ | **1.572e-02±1.3e-02** | 7.192e-02±6.2e-02 |
| MAE ↓ | **8.116e-03±5.8e-03** | 3.302e-02±2.6e-02 |
| L∞ ↓ | **1.488e-01±1.5e-01** | 1.025e+00±1.1e+00 |
| PSNR(dB) ↑ | **59.9±8.6** | 46.7±8.6 |
| SSIM ↑ | **1±7.7e-05** | 0.999±0.0017 |
| hi-freq err frac ↓ | **4.447e-01±4.9e-01** | 7.644e-01±2.4e-01 |
| rel-L2 @T/4 ↓ | **9.429e-04±4.6e-04** | 2.613e-03±6.7e-04 |
| rel-L2 @T/2 ↓ | **1.724e-03±9.2e-04** | 4.853e-03±1.3e-03 |
| rel-L2 @3T/4 ↓ | **3.038e-03±1.5e-03** | 1.031e-02±5.2e-03 |
| rel-L2 @T ↓ | **7.430e-03±6.0e-03** | 3.394e-02±2.9e-02 |
| err-growth T/(T/4) ↓ | **8.21±6.8** | 12.7±11 |
| mass-cons err ↓ | **7.019e-05±2.0e-05** | 1.429e+01±1.0e+01 |
| periodic-BC res ↓ | **1.948e-01±2.6e-02** | 2.076e-01±3.4e-02 |
| grad-energy | 1.687e-01±1.7e-02 | 1.709e-01±1.8e-02 |
| params ↓ | 4576 | **4544** |
| train wall(s) ↓ | 14.9±0.27 | **14.7±0.29** |
| infer s/step ↓ | **1.242e-03±1.1e-04** | 1.670e-03±1.1e-03 |
| throughput cells/s ↑ | **1.49e+07** | 1.36e+07 |

### heat - rel-L2 with uncertainty and paired tests

n = 40 paired evaluations (5 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_flux** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_flux (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_flux | 7.3803e-03 | [5.685e-03, 9.201e-03] | 6.1365e-03 | - | - | **reference** |
| abl_residual | 3.2580e-02 | [2.419e-02, 4.194e-02] | 2.0582e-02 | +2.520e-02 | 1.8e-12 | worse |
