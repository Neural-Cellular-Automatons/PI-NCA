### gray_scott  (grid=12, train_steps=3, eval_steps=8, epochs=25, seeds=1, clip=None)

| metric | plain_nca | mc_flux_nca | fno | pi_nca | resnet_iso | identity |
|---|---|---|---|---|---|---|
| rel-L2 ↓ | 1.762e+00±0.0e+00 | 6.916e-01±0.0e+00 | 6.741e-01±0.0e+00 | 4.103e-01±0.0e+00 | **2.127e-01±0.0e+00** | 3.889e-01±0.0e+00 |
| MSE ↓ | 6.384e-01±0.0e+00 | 9.837e-02±0.0e+00 | 9.345e-02±0.0e+00 | 3.252e-02±0.0e+00 | **8.738e-03±0.0e+00** | 2.922e-02±0.0e+00 |
| RMSE ↓ | 7.990e-01±0.0e+00 | 3.136e-01±0.0e+00 | 3.057e-01±0.0e+00 | 1.803e-01±0.0e+00 | **9.348e-02±0.0e+00** | 1.709e-01±0.0e+00 |
| MAE ↓ | 4.727e-01±0.0e+00 | 2.365e-01±0.0e+00 | 2.408e-01±0.0e+00 | 1.498e-01±0.0e+00 | **7.360e-02±0.0e+00** | 1.464e-01±0.0e+00 |
| L∞ ↓ | 4.318e+00±0.0e+00 | 1.265e+00±0.0e+00 | 1.501e+00±0.0e+00 | 4.581e-01±0.0e+00 | **3.605e-01±0.0e+00** | 3.752e-01±0.0e+00 |
| PSNR(dB) ↑ | 1.77±0 | 9.89±0 | 10.1±0 | 14.2±0 | **19.9±0** | 14.7±0 |
| SSIM ↑ | 0.21±0 | 0.653±0 | 0.558±0 | 0.781±0 | **0.909±0** | 0.787±0 |
| hi-freq err frac ↓ | **1.404e-02±0.0e+00** | 8.083e-02±0.0e+00 | 2.423e-01±0.0e+00 | 1.153e-01±0.0e+00 | 2.997e-01±0.0e+00 | 1.025e-01±0.0e+00 |
| rel-L2 @T/4 ↓ | 1.461e-01±0.0e+00 | 2.774e-01±0.0e+00 | 1.854e-01±0.0e+00 | 1.498e-01±0.0e+00 | **1.317e-01±0.0e+00** | 1.434e-01±0.0e+00 |
| rel-L2 @T/2 ↓ | **1.524e-01±0.0e+00** | 4.455e-01±0.0e+00 | 2.993e-01±0.0e+00 | 2.280e-01±0.0e+00 | 1.694e-01±0.0e+00 | 2.149e-01±0.0e+00 |
| rel-L2 @3T/4 ↓ | 4.819e-01±0.0e+00 | 5.852e-01±0.0e+00 | 4.588e-01±0.0e+00 | 3.123e-01±0.0e+00 | **1.917e-01±0.0e+00** | 2.943e-01±0.0e+00 |
| rel-L2 @T ↓ | 1.762e+00±0.0e+00 | 6.916e-01±0.0e+00 | 6.741e-01±0.0e+00 | 4.103e-01±0.0e+00 | **2.127e-01±0.0e+00** | 3.889e-01±0.0e+00 |
| err-growth T/(T/4) ↓ | 12.1±0 | 2.49±0 | 3.64±0 | 2.74±0 | **1.61±0** | 2.71±0 |
| mass-cons err ↓ | 1.113e+02±0.0e+00 | 1.450e-04±0.0e+00 | 1.729e+02±0.0e+00 | 1.831e-04±0.0e+00 | 1.749e+01±0.0e+00 | **0.000e+00±0.0e+00** |
| periodic-BC res ↓ | **1.316e-01±0.0e+00** | 1.752e-01±0.0e+00 | 2.682e-01±0.0e+00 | 3.605e-01±0.0e+00 | 3.321e-01±0.0e+00 | 3.250e-01±0.0e+00 |
| grad-energy | 2.164e-01±0.0e+00 | 3.129e-02±0.0e+00 | 9.328e-02±0.0e+00 | 3.202e-02±0.0e+00 | 2.412e-02±0.0e+00 | 2.302e-02±0.0e+00 |
| params ↓ | 7264 | 10464 | 592946 | 4928 | 5484 | **1** |
| train wall(s) ↓ | 75.7±0 | 118±0 | 128±0 | 1.55±0 | 1.86±0 | **1.15±0** |
| infer s/step ↓ | 1.009e-03±0.0e+00 | 1.317e-03±0.0e+00 | 8.026e-03±0.0e+00 | 3.290e-04±0.0e+00 | 9.527e-04±0.0e+00 | **7.350e-06±0.0e+00** |
| throughput cells/s ↑ | 4.57e+06 | 3.50e+06 | 5.74e+05 | 3.50e+06 | 1.21e+06 | **1.57e+08** |

### gray_scott - rel-L2 with uncertainty and paired tests

n = 8 paired evaluations (1 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **resnet_iso** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 2 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs resnet_iso (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| resnet_iso | 2.1375e-01 | [1.985e-01, 2.239e-01] | 2.2389e-01 | - | - | **reference** |
| identity | 3.9089e-01 | [3.542e-01, 4.153e-01] | 4.1532e-01 | +1.771e-01 | 0.0078 | worse |
| pi_nca | 4.1225e-01 | [3.703e-01, 4.402e-01] | 4.4025e-01 | +1.985e-01 | 0.0078 | worse |
