### navier_stokes  (grid=16, train_steps=6, eval_steps=12, epochs=150, seeds=2, clip=None)

| metric | abl_k3 |
|---|---|
| rel-L2 ↓ | **1.716e-01±3.7e-02** |
| MSE ↓ | **9.565e-03±5.0e-03** |
| RMSE ↓ | **9.602e-02±2.6e-02** |
| MAE ↓ | **5.012e-02±7.8e-03** |
| L∞ ↓ | **7.243e-01±2.5e-01** |
| PSNR(dB) ↑ | **35.4±0.12** |
| SSIM ↑ | **0.985±0.0065** |
| hi-freq err frac ↓ | **9.909e-02±1.2e-02** |
| rel-L2 @T/4 ↓ | **4.379e-02±8.9e-03** |
| rel-L2 @T/2 ↓ | **8.669e-02±1.8e-02** |
| rel-L2 @3T/4 ↓ | **1.291e-01±2.7e-02** |
| rel-L2 @T ↓ | **1.716e-01±3.7e-02** |
| err-growth T/(T/4) ↓ | **3.91±0.064** |
| mass-cons err ↓ | **8.205e-06±1.6e-06** |
| periodic-BC res ↓ | **1.769e-01±1.8e-02** |
| grad-energy | 1.115e-01±1.6e-02 |
| params ↓ | **4576** |
| train wall(s) ↓ | **10.5±2.2** |
| infer s/step ↓ | **4.737e-04±6.6e-05** |
| throughput cells/s ↑ | **4.37e+06** |

### navier_stokes - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_k3** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 0 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_k3 (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_k3 | 1.4430e-01 | [1.212e-01, 1.674e-01] | 1.3816e-01 | - | - | **reference** |
