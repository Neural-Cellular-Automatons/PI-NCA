### heat  (grid=12, train_steps=3, eval_steps=8, epochs=25, seeds=1, clip=None)

| metric | plain_nca | pi_nca | multiscale_flux_nca | spectral_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|---|
| rel-L2 ↓ | 2.242e-01±0.0e+00 | 3.100e-02±0.0e+00 | 2.749e-02±0.0e+00 | **6.365e-03±0.0e+00** | 2.102e-02±0.0e+00 | 2.239e-01±0.0e+00 | 3.608e-01±0.0e+00 |
| MSE ↓ | 1.318e-01±0.0e+00 | 2.521e-03±0.0e+00 | 1.983e-03±0.0e+00 | **1.063e-04±0.0e+00** | 1.159e-03±0.0e+00 | 1.955e-01±0.0e+00 | 5.076e-01±0.0e+00 |
| RMSE ↓ | 3.630e-01±0.0e+00 | 5.021e-02±0.0e+00 | 4.453e-02±0.0e+00 | **1.031e-02±0.0e+00** | 3.404e-02±0.0e+00 | 4.421e-01±0.0e+00 | 7.125e-01±0.0e+00 |
| MAE ↓ | 1.809e-01±0.0e+00 | 3.299e-02±0.0e+00 | 3.187e-02±0.0e+00 | **7.284e-03±0.0e+00** | 2.616e-02±0.0e+00 | 2.615e-01±0.0e+00 | 4.136e-01±0.0e+00 |
| L∞ ↓ | 4.522e+00±0.0e+00 | 3.048e-01±0.0e+00 | 2.643e-01±0.0e+00 | **6.355e-02±0.0e+00** | 1.093e-01±0.0e+00 | 2.582e+00±0.0e+00 | 4.004e+00±0.0e+00 |
| PSNR(dB) ↑ | 24.6±0 | 41.8±0 | 42.9±0 | **55.6±0** | 45.2±0 | 26.3±0 | 22.1±0 |
| SSIM ↑ | 0.953±0 | 0.999±0 | 0.999±0 | **1±0** | 1±0 | 0.965±0 | 0.928±0 |
| hi-freq err frac ↓ | 8.842e-01±0.0e+00 | 3.974e-01±0.0e+00 | 4.347e-01±0.0e+00 | 4.416e-01±0.0e+00 | **6.798e-02±0.0e+00** | 2.636e-01±0.0e+00 | 1.058e-01±0.0e+00 |
| rel-L2 @T/4 ↓ | 2.657e-02±0.0e+00 | 1.765e-02±0.0e+00 | 1.771e-02±0.0e+00 | **6.678e-03±0.0e+00** | 2.176e-02±0.0e+00 | 7.726e-02±0.0e+00 | 1.029e-01±0.0e+00 |
| rel-L2 @T/2 ↓ | 3.974e-02±0.0e+00 | 2.355e-02±0.0e+00 | 2.210e-02±0.0e+00 | **6.121e-03±0.0e+00** | 1.882e-02±0.0e+00 | 1.331e-01±0.0e+00 | 1.940e-01±0.0e+00 |
| rel-L2 @3T/4 ↓ | 8.872e-02±0.0e+00 | 2.662e-02±0.0e+00 | 2.417e-02±0.0e+00 | **5.866e-03±0.0e+00** | 1.366e-02±0.0e+00 | 1.826e-01±0.0e+00 | 2.809e-01±0.0e+00 |
| rel-L2 @T ↓ | 2.242e-01±0.0e+00 | 3.100e-02±0.0e+00 | 2.749e-02±0.0e+00 | **6.365e-03±0.0e+00** | 2.102e-02±0.0e+00 | 2.239e-01±0.0e+00 | 3.608e-01±0.0e+00 |
| err-growth T/(T/4) ↓ | 8.44±0 | 1.76±0 | 1.55±0 | **0.953±0** | 0.966±0 | 2.9±0 | 3.51±0 |
| mass-cons err ↓ | 3.972e+01±0.0e+00 | **3.128e-04±0.0e+00** | 1.850e-03±0.0e+00 | 1.728e-03±0.0e+00 | 9.846e+00±0.0e+00 | 8.920e+00±0.0e+00 | 1.140e+00±0.0e+00 |
| periodic-BC res ↓ | 4.862e-01±0.0e+00 | 2.369e-01±0.0e+00 | 2.369e-01±0.0e+00 | **2.314e-01±0.0e+00** | 2.318e-01±0.0e+00 | 7.451e-01±0.0e+00 | 8.636e-01±0.0e+00 |
| grad-energy | 2.444e-01±0.0e+00 | 1.913e-01±0.0e+00 | 1.887e-01±0.0e+00 | 1.898e-01±0.0e+00 | 1.875e-01±0.0e+00 | 1.901e+00±0.0e+00 | 2.937e+00±0.0e+00 |
| params ↓ | 6784 | 4576 | 5520 | 134225 | 592897 | 5364 | **1** |
| train wall(s) ↓ | 55.4±0 | 47.2±0 | 42.4±0 | 51.8±0 | 65.2±0 | 2.61±0 | **1.27±0** |
| infer s/step ↓ | 9.823e-04±0.0e+00 | 5.981e-04±0.0e+00 | 8.546e-04±0.0e+00 | 2.424e-03±0.0e+00 | 4.667e-03±0.0e+00 | 8.898e-04±0.0e+00 | **8.850e-06±0.0e+00** |
| throughput cells/s ↑ | 4.69e+06 | 7.71e+06 | 5.39e+06 | 1.90e+06 | 9.87e+05 | 1.29e+06 | **1.30e+08** |

### heat - rel-L2 with uncertainty and paired tests

n = 8 paired evaluations (1 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **resnet_iso** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs resnet_iso (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| resnet_iso | 2.3209e-01 | [2.072e-01, 2.572e-01] | 2.3457e-01 | - | - | **reference** |
| identity | 3.7669e-01 | [3.423e-01, 4.150e-01] | 3.7278e-01 | +1.446e-01 | 0.0078 | worse |
