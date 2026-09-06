### cahn_hilliard  (grid=16, train_steps=6, eval_steps=12, epochs=150, seeds=2, clip=None)

| metric | plain_nca | pi_nca |
|---|---|---|
| rel-L2 ↓ | 8.324e-02±1.9e-03 | **8.294e-02±4.3e-03** |
| MSE ↓ | 2.659e-04±2.5e-05 | **2.633e-04±1.4e-05** |
| RMSE ↓ | 1.630e-02±7.7e-04 | **1.622e-02±4.4e-04** |
| MAE ↓ | **1.232e-02±7.5e-04** | 1.247e-02±3.9e-04 |
| L∞ ↓ | 8.877e-02±1.3e-02 | **8.114e-02±1.6e-03** |
| PSNR(dB) ↑ | 36.6±0.66 | **36.7±0.014** |
| SSIM ↑ | 0.996±0.00075 | **0.997±0.00034** |
| hi-freq err frac ↓ | **8.062e-01±4.1e-03** | 8.934e-01±9.7e-03 |
| rel-L2 @T/4 ↓ | 1.521e-02±3.1e-04 | **1.451e-02±6.4e-04** |
| rel-L2 @T/2 ↓ | 3.362e-02±7.2e-04 | **3.270e-02±1.5e-03** |
| rel-L2 @3T/4 ↓ | 5.606e-02±1.3e-03 | **5.531e-02±2.7e-03** |
| rel-L2 @T ↓ | 8.324e-02±1.9e-03 | **8.294e-02±4.3e-03** |
| err-growth T/(T/4) ↓ | **5.47±0.014** | 5.72±0.042 |
| mass-cons err ↓ | 1.712e-01±6.6e-02 | **1.673e-06±2.3e-07** |
| periodic-BC res ↓ | 1.658e-01±9.5e-03 | **1.641e-01±8.9e-03** |
| grad-energy | 2.735e-02±3.2e-04 | 2.740e-02±4.2e-04 |
| params ↓ | 6784 | **4576** |
| train wall(s) ↓ | **12.3±1.8** | 13.4±0.49 |
| infer s/step ↓ | 6.109e-04±1.3e-05 | **6.055e-04±1.4e-04** |
| throughput cells/s ↑ | 3.35e+06 | **3.48e+06** |

### cahn_hilliard - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **pi_nca** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs pi_nca (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| pi_nca | 8.3050e-02 | [7.996e-02, 8.613e-02] | 8.2511e-02 | - | - | **reference** |
| plain_nca | 8.3264e-02 | [8.080e-02, 8.574e-02] | 8.4333e-02 | +2.135e-04 | 1 | tie |

Statistically indistinguishable from pi_nca at this sample size: `plain_nca`.
