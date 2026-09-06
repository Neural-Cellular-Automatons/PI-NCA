### wave  (grid=12, train_steps=3, eval_steps=8, epochs=25, seeds=1, clip=None)

| metric | plain_nca | mc_flux_nca | fno | pi_nca | resnet_iso | identity |
|---|---|---|---|---|---|---|
| rel-L2 ↓ | 5.205e-02±0.0e+00 | 5.571e-02±0.0e+00 | 5.650e-02±0.0e+00 | **4.243e-02±0.0e+00** | 4.840e-02±0.0e+00 | 7.550e-02±0.0e+00 |
| MSE ↓ | 2.115e-03±0.0e+00 | 2.423e-03±0.0e+00 | 2.492e-03±0.0e+00 | **2.056e-03±0.0e+00** | 2.676e-03±0.0e+00 | 6.510e-03±0.0e+00 |
| RMSE ↓ | 4.599e-02±0.0e+00 | 4.923e-02±0.0e+00 | 4.992e-02±0.0e+00 | **4.535e-02±0.0e+00** | 5.173e-02±0.0e+00 | 8.069e-02±0.0e+00 |
| MAE ↓ | 2.757e-02±0.0e+00 | 2.634e-02±0.0e+00 | 2.410e-02±0.0e+00 | **2.094e-02±0.0e+00** | 2.794e-02±0.0e+00 | 4.075e-02±0.0e+00 |
| L∞ ↓ | **2.986e-01±0.0e+00** | 3.788e-01±0.0e+00 | 3.946e-01±0.0e+00 | 1.033e+00±0.0e+00 | 9.855e-01±0.0e+00 | 7.676e-01±0.0e+00 |
| PSNR(dB) ↑ | 41.5±0 | 40.9±0 | 40.8±0 | **45.3±0** | 44.2±0 | 40.3±0 |
| SSIM ↑ | 0.998±0 | 0.998±0 | 0.998±0 | **0.999±0** | 0.998±0 | 0.997±0 |
| hi-freq err frac ↓ | 9.420e-03±0.0e+00 | 1.165e-02±0.0e+00 | **1.633e-04±0.0e+00** | 1.972e-01±0.0e+00 | 2.754e-01±0.0e+00 | 4.366e-01±0.0e+00 |
| rel-L2 @T/4 ↓ | **1.544e-02±0.0e+00** | 1.657e-02±0.0e+00 | 1.621e-02±0.0e+00 | 2.760e-02±0.0e+00 | 2.761e-02±0.0e+00 | 3.036e-02±0.0e+00 |
| rel-L2 @T/2 ↓ | 1.953e-02±0.0e+00 | 1.401e-02±0.0e+00 | **1.230e-02±0.0e+00** | 3.287e-02±0.0e+00 | 3.369e-02±0.0e+00 | 4.298e-02±0.0e+00 |
| rel-L2 @3T/4 ↓ | 2.339e-02±0.0e+00 | 1.434e-02±0.0e+00 | **1.268e-02±0.0e+00** | 3.802e-02±0.0e+00 | 4.084e-02±0.0e+00 | 5.853e-02±0.0e+00 |
| rel-L2 @T ↓ | 5.205e-02±0.0e+00 | 5.571e-02±0.0e+00 | 5.650e-02±0.0e+00 | **4.243e-02±0.0e+00** | 4.840e-02±0.0e+00 | 7.550e-02±0.0e+00 |
| err-growth T/(T/4) ↓ | 3.37±0 | 3.36±0 | 3.49±0 | **1.54±0** | 1.75±0 | 2.49±0 |
| mass-cons err ↓ | 5.636e+00±0.0e+00 | **1.373e-04±0.0e+00** | 5.188e-01±0.0e+00 | 2.147e-01±0.0e+00 | 3.686e+00±0.0e+00 | 2.276e-01±0.0e+00 |
| periodic-BC res ↓ | 1.276e-01±0.0e+00 | 1.264e-01±0.0e+00 | **1.261e-01±0.0e+00** | 2.617e-01±0.0e+00 | 2.581e-01±0.0e+00 | 2.423e-01±0.0e+00 |
| grad-energy | 8.318e-02±0.0e+00 | 8.534e-02±0.0e+00 | 8.596e-02±0.0e+00 | 3.736e-01±0.0e+00 | 3.818e-01±0.0e+00 | 4.062e-01±0.0e+00 |
| params ↓ | 7264 | 10464 | 592946 | 4928 | 5484 | **1** |
| train wall(s) ↓ | 87.6±0 | 121±0 | 106±0 | 1.85±0 | 1.99±0 | **1.2±0** |
| infer s/step ↓ | 1.480e-03±0.0e+00 | 2.564e-03±0.0e+00 | 4.585e-03±0.0e+00 | 2.913e-04±0.0e+00 | 8.297e-04±0.0e+00 | **1.175e-05±0.0e+00** |
| throughput cells/s ↑ | 3.11e+06 | 1.80e+06 | 1.00e+06 | 3.95e+06 | 1.39e+06 | **9.80e+07** |

### wave - rel-L2 with uncertainty and paired tests

n = 8 paired evaluations (1 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **pi_nca** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 2 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs pi_nca (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| pi_nca | 3.7099e-02 | [3.248e-02, 4.406e-02] | 3.3405e-02 | - | - | **reference** |
| resnet_iso | 4.4462e-02 | [4.010e-02, 5.013e-02] | 4.0696e-02 | +7.363e-03 | 0.0078 | worse |
| identity | 7.5524e-02 | [7.180e-02, 8.039e-02] | 7.3203e-02 | +3.843e-02 | 0.0078 | worse |
