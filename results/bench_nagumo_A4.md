### nagumo  (grid=16, train_steps=6, eval_steps=12, epochs=150, seeds=2, clip=None)

| metric | abl_flux | abl_residual |
|---|---|---|
| rel-L2 ↓ | 1.103e-01±1.4e-03 | **1.392e-02±6.6e-04** |
| MSE ↓ | 3.871e-03±2.1e-04 | **6.175e-05±7.6e-06** |
| RMSE ↓ | 6.221e-02±1.7e-03 | **7.851e-03±4.8e-04** |
| MAE ↓ | 5.972e-02±2.2e-03 | **6.175e-03±2.7e-04** |
| L∞ ↓ | 1.098e-01±3.1e-03 | **4.757e-02±2.1e-02** |
| PSNR(dB) ↑ | 20.9±0.022 | **38.9±0.28** |
| SSIM ↑ | 0.985±0.00076 | **0.998±0.00041** |
| hi-freq err frac ↓ | **6.641e-03±5.5e-04** | 4.127e-01±1.3e-01 |
| rel-L2 @T/4 ↓ | 2.775e-02±3.6e-04 | **4.048e-03±6.5e-05** |
| rel-L2 @T/2 ↓ | 5.546e-02±7.1e-04 | **7.671e-03±2.0e-04** |
| rel-L2 @3T/4 ↓ | 8.301e-02±1.1e-03 | **1.094e-02±4.0e-04** |
| rel-L2 @T ↓ | 1.103e-01±1.4e-03 | **1.392e-02±6.6e-04** |
| err-growth T/(T/4) ↓ | 3.97±0.00011 | **3.44±0.11** |
| mass-cons err ↓ | **3.052e-05±5.4e-06** | 1.561e+01±5.1e-01 |
| periodic-BC res ↓ | **7.232e-02±6.4e-03** | 7.385e-02±5.7e-03 |
| grad-energy | 1.037e-02±1.0e-04 | 1.066e-02±3.5e-04 |
| params ↓ | 4576 | **4544** |
| train wall(s) ↓ | 9.95±0.49 | **9.21±0.39** |
| infer s/step ↓ | **5.124e-04±2.0e-04** | 6.272e-04±8.7e-05 |
| throughput cells/s ↑ | **4.31e+06** | 3.30e+06 |

### nagumo - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_residual** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_residual (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_residual | 1.3950e-02 | [1.309e-02, 1.491e-02] | 1.3868e-02 | - | - | **reference** |
| abl_flux | 1.0947e-01 | [1.064e-01, 1.124e-01] | 1.0989e-01 | +9.552e-02 | 3.1e-05 | worse |
