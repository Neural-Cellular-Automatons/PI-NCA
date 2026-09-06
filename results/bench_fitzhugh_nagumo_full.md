### fitzhugh_nagumo  (grid=12, train_steps=3, eval_steps=8, epochs=25, seeds=1, clip=None)

| metric | plain_nca | mc_flux_nca | fno | pi_nca | resnet_iso | identity |
|---|---|---|---|---|---|---|
| rel-L2 ↓ | **1.247e-01±0.0e+00** | 9.986e-01±0.0e+00 | 1.986e-01±0.0e+00 | 9.121e-01±0.0e+00 | 9.709e-01±0.0e+00 | 1.045e+00±0.0e+00 |
| MSE ↓ | **4.225e-04±0.0e+00** | 2.708e-02±0.0e+00 | 1.071e-03±0.0e+00 | 3.342e-03±0.0e+00 | 3.786e-03±0.0e+00 | 4.384e-03±0.0e+00 |
| RMSE ↓ | **2.056e-02±0.0e+00** | 1.646e-01±0.0e+00 | 3.273e-02±0.0e+00 | 5.781e-02±0.0e+00 | 6.153e-02±0.0e+00 | 6.621e-02±0.0e+00 |
| MAE ↓ | **1.482e-02±0.0e+00** | 1.389e-01±0.0e+00 | 2.956e-02±0.0e+00 | 4.792e-02±0.0e+00 | 4.863e-02±0.0e+00 | 5.355e-02±0.0e+00 |
| L∞ ↓ | 9.666e-02±0.0e+00 | 2.685e-01±0.0e+00 | **5.672e-02±0.0e+00** | 2.154e-01±0.0e+00 | 2.280e-01±0.0e+00 | 2.372e-01±0.0e+00 |
| PSNR(dB) ↑ | **26.5±0** | 8.41±0 | 22.4±0 | 17.4±0 | 16.9±0 | 16.2±0 |
| SSIM ↑ | **0.987±0** | 4.82e-05±0 | 0.96±0 | 0.128±0 | 0.369±0 | 0.124±0 |
| hi-freq err frac ↓ | 6.415e-01±0.0e+00 | **6.651e-03±0.0e+00** | 4.822e-02±0.0e+00 | 8.898e-01±0.0e+00 | 8.481e-01±0.0e+00 | 8.377e-01±0.0e+00 |
| rel-L2 @T/4 ↓ | 5.208e-01±0.0e+00 | 7.626e-01±0.0e+00 | 4.200e-01±0.0e+00 | **3.271e-01±0.0e+00** | 3.432e-01±0.0e+00 | 3.549e-01±0.0e+00 |
| rel-L2 @T/2 ↓ | 2.444e-01±0.0e+00 | 9.552e-01±0.0e+00 | **1.676e-01±0.0e+00** | 5.820e-01±0.0e+00 | 6.171e-01±0.0e+00 | 6.459e-01±0.0e+00 |
| rel-L2 @3T/4 ↓ | 1.496e-01±0.0e+00 | 9.903e-01±0.0e+00 | **9.664e-02±0.0e+00** | 7.733e-01±0.0e+00 | 8.241e-01±0.0e+00 | 8.743e-01±0.0e+00 |
| rel-L2 @T ↓ | **1.247e-01±0.0e+00** | 9.986e-01±0.0e+00 | 1.986e-01±0.0e+00 | 9.121e-01±0.0e+00 | 9.709e-01±0.0e+00 | 1.045e+00±0.0e+00 |
| err-growth T/(T/4) ↓ | **0.239±0** | 1.31±0 | 0.473±0 | 2.79±0 | 2.83±0 | 2.94±0 |
| mass-cons err ↓ | 9.217e+01±0.0e+00 | 1.668e-06±0.0e+00 | 8.621e+01±0.0e+00 | 3.874e-07±0.0e+00 | 1.260e+00±0.0e+00 | **0.000e+00±0.0e+00** |
| periodic-BC res ↓ | 2.452e-02±0.0e+00 | 1.124e-02±0.0e+00 | **1.027e-02±0.0e+00** | 9.739e-02±0.0e+00 | 1.099e-01±0.0e+00 | 1.128e-01±0.0e+00 |
| grad-energy | 3.313e-04±0.0e+00 | 2.470e-04±0.0e+00 | 2.086e-04±0.0e+00 | 6.663e-03±0.0e+00 | 9.069e-03±0.0e+00 | 9.578e-03±0.0e+00 |
| params ↓ | 7264 | 10464 | 592946 | 4928 | 5484 | **1** |
| train wall(s) ↓ | 85.7±0 | 120±0 | 85.8±0 | 1.88±0 | 2.09±0 | **1.65±0** |
| infer s/step ↓ | 2.270e-03±0.0e+00 | 1.097e-03±0.0e+00 | 1.051e-02±0.0e+00 | 2.644e-04±0.0e+00 | 1.419e-03±0.0e+00 | **7.500e-06±0.0e+00** |
| throughput cells/s ↑ | 2.03e+06 | 4.20e+06 | 4.39e+05 | 4.36e+06 | 8.12e+05 | **1.54e+08** |

### fitzhugh_nagumo - rel-L2 with uncertainty and paired tests

n = 8 paired evaluations (1 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **pi_nca** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 2 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs pi_nca (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| pi_nca | 9.1507e-01 | [8.588e-01, 9.776e-01] | 8.7030e-01 | - | - | **reference** |
| resnet_iso | 9.7483e-01 | [9.136e-01, 1.040e+00] | 9.4963e-01 | +5.976e-02 | 0.0078 | worse |
| identity | 1.0485e+00 | [9.951e-01, 1.107e+00] | 1.0190e+00 | +1.335e-01 | 0.0078 | worse |
