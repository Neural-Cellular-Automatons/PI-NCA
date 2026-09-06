### allen_cahn  (grid=16, train_steps=6, eval_steps=12, epochs=150, seeds=2, clip=None)

| metric | multiscale_flux_nca | bounded_multiscale_nca |
|---|---|---|
| rel-L2 ↓ | **3.291e-02±8.2e-04** | 3.291e-02±8.2e-04 |
| MSE ↓ | **1.053e-03±5.3e-05** | 1.053e-03±5.3e-05 |
| RMSE ↓ | **3.245e-02±8.2e-04** | 3.245e-02±8.2e-04 |
| MAE ↓ | **2.580e-02±6.0e-04** | 2.580e-02±6.0e-04 |
| L∞ ↓ | 1.259e-01±1.5e-03 | **1.259e-01±1.5e-03** |
| PSNR(dB) ↑ | **36.1±0.24** | 36.1±0.24 |
| SSIM ↑ | **0.994±0.0074** | 0.994±0.0074 |
| hi-freq err frac ↓ | 6.815e-01±8.0e-03 | **6.815e-01±8.0e-03** |
| rel-L2 @T/4 ↓ | **1.183e-02±3.2e-04** | 1.183e-02±3.2e-04 |
| rel-L2 @T/2 ↓ | **2.076e-02±5.6e-04** | 2.076e-02±5.6e-04 |
| rel-L2 @3T/4 ↓ | **2.761e-02±7.2e-04** | 2.761e-02±7.2e-04 |
| rel-L2 @T ↓ | **3.291e-02±8.2e-04** | 3.291e-02±8.2e-04 |
| err-growth T/(T/4) ↓ | 2.78±0.0057 | **2.78±0.0057** |
| mass-cons err ↓ | 8.784e-06±3.7e-06 | **5.834e-06±1.8e-06** |
| periodic-BC res ↓ | **9.907e-01±4.7e-02** | 9.907e-01±4.7e-02 |
| grad-energy | 9.595e-01±1.9e-02 | 9.595e-01±1.9e-02 |
| params ↓ | **5520** | 5520 |
| train wall(s) ↓ | 8.15±1.5 | **7.93±1.4** |
| infer s/step ↓ | 9.438e-04±2.3e-04 | **5.718e-04±3.3e-05** |
| throughput cells/s ↑ | 2.23e+06 | **3.59e+06** |

### allen_cahn - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **multiscale_flux_nca** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs multiscale_flux_nca (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| multiscale_flux_nca | 3.2864e-02 | [3.199e-02, 3.374e-02] | 3.2960e-02 | - | - | **reference** |
| bounded_multiscale_nca | 3.2864e-02 | [3.199e-02, 3.374e-02] | 3.2960e-02 | +1.665e-08 | 0.088 | tie |

Statistically indistinguishable from multiscale_flux_nca at this sample size: `bounded_multiscale_nca`.
