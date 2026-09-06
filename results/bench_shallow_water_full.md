### shallow_water  (grid=12, train_steps=3, eval_steps=8, epochs=25, seeds=1, clip=None)

| metric | plain_nca | mc_flux_nca | fno | pi_nca | resnet_iso |
|---|---|---|---|---|---|
| rel-L2 ↓ | 2.618e-02±0.0e+00 | 1.613e-02±0.0e+00 | 2.445e-02±0.0e+00 | **1.252e-02±0.0e+00** | 1.991e-02±0.0e+00 |
| MSE ↓ | 2.554e-04±0.0e+00 | 9.693e-05±0.0e+00 | 2.228e-04±0.0e+00 | **5.837e-05±0.0e+00** | 1.476e-04±0.0e+00 |
| RMSE ↓ | 1.598e-02±0.0e+00 | 9.845e-03±0.0e+00 | 1.492e-02±0.0e+00 | **7.640e-03±0.0e+00** | 1.215e-02±0.0e+00 |
| MAE ↓ | 1.056e-02±0.0e+00 | 5.000e-03±0.0e+00 | 8.093e-03±0.0e+00 | **4.609e-03±0.0e+00** | 7.360e-03±0.0e+00 |
| L∞ ↓ | 1.289e-01±0.0e+00 | 9.347e-02±0.0e+00 | 1.211e-01±0.0e+00 | **5.053e-02±0.0e+00** | 7.947e-02±0.0e+00 |
| PSNR(dB) ↑ | 39.9±0 | 44.1±0 | 40.5±0 | **46.3±0** | 42.3±0 |
| SSIM ↑ | 0.999±0 | 1±0 | 1±0 | **1±0** | 1±0 |
| hi-freq err frac ↓ | 1.336e-02±0.0e+00 | 2.528e-03±0.0e+00 | **6.850e-05±0.0e+00** | 2.281e-02±0.0e+00 | 1.269e-02±0.0e+00 |
| rel-L2 @T/4 ↓ | 8.766e-03±0.0e+00 | 5.658e-03±0.0e+00 | 8.984e-03±0.0e+00 | **4.236e-03±0.0e+00** | 5.388e-03±0.0e+00 |
| rel-L2 @T/2 ↓ | 1.109e-02±0.0e+00 | **5.573e-03±0.0e+00** | 8.965e-03±0.0e+00 | 7.710e-03±0.0e+00 | 1.051e-02±0.0e+00 |
| rel-L2 @3T/4 ↓ | 1.086e-02±0.0e+00 | **5.562e-03±0.0e+00** | 8.518e-03±0.0e+00 | 1.046e-02±0.0e+00 | 1.535e-02±0.0e+00 |
| rel-L2 @T ↓ | 2.618e-02±0.0e+00 | 1.613e-02±0.0e+00 | 2.445e-02±0.0e+00 | **1.252e-02±0.0e+00** | 1.991e-02±0.0e+00 |
| err-growth T/(T/4) ↓ | 2.99±0 | 2.85±0 | **2.72±0** | 2.95±0 | 3.7±0 |
| mass-cons err ↓ | 9.869e+00±0.0e+00 | 1.373e-04±0.0e+00 | 2.140e+00±0.0e+00 | **4.768e-05±0.0e+00** | 6.059e-01±0.0e+00 |
| periodic-BC res ↓ | 1.369e-02±0.0e+00 | **1.223e-02±0.0e+00** | 1.326e-02±0.0e+00 | 1.587e-02±0.0e+00 | 1.330e-02±0.0e+00 |
| grad-energy | 8.574e-04±0.0e+00 | 6.789e-04±0.0e+00 | 7.624e-04±0.0e+00 | 1.194e-03±0.0e+00 | 1.187e-03±0.0e+00 |
| params ↓ | 7744 | 10992 | 592995 | **5280** | 5604 |
| train wall(s) ↓ | 96.3±0 | 107±0 | 116±0 | **1.96±0** | 2.04±0 |
| infer s/step ↓ | 1.185e-03±0.0e+00 | 1.456e-03±0.0e+00 | 1.016e-02±0.0e+00 | **3.326e-04±0.0e+00** | 8.371e-04±0.0e+00 |
| throughput cells/s ↑ | **3.89e+06** | 3.16e+06 | 4.53e+05 | 3.46e+06 | 1.38e+06 |

### shallow_water - rel-L2 with uncertainty and paired tests

n = 8 paired evaluations (1 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **pi_nca** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs pi_nca (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| pi_nca | 1.2185e-02 | [1.038e-02, 1.376e-02] | 1.2707e-02 | - | - | **reference** |
| resnet_iso | 1.9307e-02 | [1.627e-02, 2.206e-02] | 1.9551e-02 | +7.122e-03 | 0.0078 | worse |
