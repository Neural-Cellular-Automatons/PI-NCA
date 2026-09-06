### cahn_hilliard  (grid=16, train_steps=6, eval_steps=12, epochs=150, seeds=2, clip=None)

| metric | multiscale_flux_nca | bounded_multiscale_nca |
|---|---|---|
| rel-L2 ↓ | 8.782e-02±2.0e-03 | **8.782e-02±2.0e-03** |
| MSE ↓ | 2.953e-04±1.2e-06 | **2.953e-04±1.2e-06** |
| RMSE ↓ | 1.718e-02±3.5e-05 | **1.718e-02±3.5e-05** |
| MAE ↓ | 1.344e-02±2.8e-05 | **1.344e-02±2.8e-05** |
| L∞ ↓ | **7.061e-02±9.3e-03** | 7.062e-02±9.3e-03 |
| PSNR(dB) ↑ | 36.2±0.27 | **36.2±0.27** |
| SSIM ↑ | 0.996±0.00017 | **0.996±0.00017** |
| hi-freq err frac ↓ | **8.571e-01±3.9e-05** | 8.571e-01±3.4e-05 |
| rel-L2 @T/4 ↓ | 1.637e-02±5.0e-04 | **1.637e-02±5.0e-04** |
| rel-L2 @T/2 ↓ | 3.621e-02±9.9e-04 | **3.621e-02±9.9e-04** |
| rel-L2 @3T/4 ↓ | 5.995e-02±1.4e-03 | **5.995e-02±1.4e-03** |
| rel-L2 @T ↓ | 8.782e-02±2.0e-03 | **8.782e-02±2.0e-03** |
| err-growth T/(T/4) ↓ | **5.36±0.045** | 5.36±0.045 |
| mass-cons err ↓ | 2.831e-06±8.4e-07 | **2.034e-06±6.4e-07** |
| periodic-BC res ↓ | **1.642e-01±1.2e-02** | 1.642e-01±1.2e-02 |
| grad-energy | 2.751e-02±2.4e-04 | 2.751e-02±2.4e-04 |
| params ↓ | **5520** | 5520 |
| train wall(s) ↓ | **8.25±0.48** | 8.71±0.16 |
| infer s/step ↓ | 9.847e-04±3.4e-04 | **8.380e-04±3.1e-04** |
| throughput cells/s ↑ | 2.21e+06 | **2.63e+06** |

### cahn_hilliard - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **bounded_multiscale_nca** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs bounded_multiscale_nca (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| bounded_multiscale_nca | 8.7533e-02 | [8.537e-02, 8.970e-02] | 8.8169e-02 | - | - | **reference** |
| multiscale_flux_nca | 8.7533e-02 | [8.537e-02, 8.970e-02] | 8.8170e-02 | +1.118e-07 | 0.82 | tie |

Statistically indistinguishable from bounded_multiscale_nca at this sample size: `multiscale_flux_nca`.
