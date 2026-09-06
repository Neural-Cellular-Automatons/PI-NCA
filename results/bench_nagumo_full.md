### nagumo  (grid=12, train_steps=3, eval_steps=8, epochs=25, seeds=1, clip=None)

| metric | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| rel-L2 ↓ | 7.333e-02±0.0e+00 | 3.768e-01±0.0e+00 | 3.763e-01±0.0e+00 | 8.068e-02±0.0e+00 | **2.204e-02±0.0e+00** | 8.043e-02±0.0e+00 |
| MSE ↓ | 3.592e-03±0.0e+00 | 9.487e-02±0.0e+00 | 9.461e-02±0.0e+00 | 4.349e-03±0.0e+00 | **1.572e-04±0.0e+00** | 2.095e-03±0.0e+00 |
| RMSE ↓ | 5.994e-02±0.0e+00 | 3.080e-01±0.0e+00 | 3.076e-01±0.0e+00 | 6.595e-02±0.0e+00 | **1.254e-02±0.0e+00** | 4.577e-02±0.0e+00 |
| MAE ↓ | 4.366e-02±0.0e+00 | 3.014e-01±0.0e+00 | 3.014e-01±0.0e+00 | 6.020e-02±0.0e+00 | **9.927e-03±0.0e+00** | 4.320e-02±0.0e+00 |
| L∞ ↓ | 3.271e-01±0.0e+00 | 5.171e-01±0.0e+00 | 5.227e-01±0.0e+00 | 1.857e-01±0.0e+00 | **4.984e-02±0.0e+00** | 9.010e-02±0.0e+00 |
| PSNR(dB) ↑ | 22.4±0 | 8.22±0 | 8.24±0 | 21.6±0 | **34.9±0** | 23.7±0 |
| SSIM ↑ | 0.949±0 | 0.807±0 | 0.826±0 | 0.978±0 | **0.996±0** | 0.989±0 |
| hi-freq err frac ↓ | 8.935e-02±0.0e+00 | 3.015e-03±0.0e+00 | **2.190e-03±0.0e+00** | 2.230e-02±0.0e+00 | 3.421e-01±0.0e+00 | 2.939e-02±0.0e+00 |
| rel-L2 @T/4 ↓ | 1.682e-02±0.0e+00 | 1.140e-01±0.0e+00 | 1.123e-01±0.0e+00 | 2.646e-02±0.0e+00 | **5.733e-03±0.0e+00** | 2.032e-02±0.0e+00 |
| rel-L2 @T/2 ↓ | 2.712e-02±0.0e+00 | 2.186e-01±0.0e+00 | 2.158e-01±0.0e+00 | 2.434e-02±0.0e+00 | **1.129e-02±0.0e+00** | 4.052e-02±0.0e+00 |
| rel-L2 @3T/4 ↓ | 4.041e-02±0.0e+00 | 3.071e-01±0.0e+00 | 3.046e-01±0.0e+00 | 4.234e-02±0.0e+00 | **1.671e-02±0.0e+00** | 6.057e-02±0.0e+00 |
| rel-L2 @T ↓ | 7.333e-02±0.0e+00 | 3.768e-01±0.0e+00 | 3.763e-01±0.0e+00 | 8.068e-02±0.0e+00 | **2.204e-02±0.0e+00** | 8.043e-02±0.0e+00 |
| err-growth T/(T/4) ↓ | 4.36±0 | 3.31±0 | 3.35±0 | **3.05±0** | 3.84±0 | 3.96±0 |
| mass-cons err ↓ | 1.930e+02±0.0e+00 | 1.106e-04±0.0e+00 | 6.828e-04±0.0e+00 | 1.392e+02±0.0e+00 | 7.270e+00±0.0e+00 | **0.000e+00±0.0e+00** |
| periodic-BC res ↓ | 8.383e-02±0.0e+00 | 8.779e-02±0.0e+00 | 8.529e-02±0.0e+00 | **7.491e-02±0.0e+00** | 8.145e-02±0.0e+00 | 7.896e-02±0.0e+00 |
| grad-energy | 1.317e-02±0.0e+00 | 1.318e-02±0.0e+00 | 1.489e-02±0.0e+00 | 1.017e-02±0.0e+00 | 8.935e-03±0.0e+00 | 7.990e-03±0.0e+00 |
| params ↓ | 6784 | 4576 | 5520 | 592897 | 5364 | **1** |
| train wall(s) ↓ | 65.9±0 | 49.5±0 | 67.3±0 | 112±0 | 2.25±0 | **1.58±0** |
| infer s/step ↓ | 6.538e-04±0.0e+00 | 7.949e-04±0.0e+00 | 1.391e-03±0.0e+00 | 5.983e-03±0.0e+00 | 8.552e-04±0.0e+00 | **1.350e-05±0.0e+00** |
| throughput cells/s ↑ | 7.05e+06 | 5.80e+06 | 3.31e+06 | 7.70e+05 | 1.35e+06 | **8.53e+07** |

### nagumo - rel-L2 with uncertainty and paired tests

n = 8 paired evaluations (1 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **resnet_iso** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs resnet_iso (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| resnet_iso | 2.2080e-02 | [2.002e-02, 2.414e-02] | 2.2095e-02 | - | - | **reference** |
| identity | 7.9681e-02 | [7.644e-02, 8.291e-02] | 7.9629e-02 | +5.760e-02 | 0.0078 | worse |
