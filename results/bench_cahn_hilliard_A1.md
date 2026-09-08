### cahn_hilliard  (grid=48, train_steps=12, eval_steps=48, epochs=1200, seeds=5, clip=None)

| metric | multiscale_flux_nca | bounded_multiscale_nca |
|---|---|---|
| rel-L2 ↓ | 5.153e-01±3.4e-02 | **3.584e-01±3.2e-02** |
| MSE ↓ | 1.634e-01±2.2e-02 | **7.928e-02±1.4e-02** |
| RMSE ↓ | 4.035e-01±2.7e-02 | **2.807e-01±2.5e-02** |
| MAE ↓ | 3.517e-01±2.5e-02 | **2.196e-01±1.8e-02** |
| L∞ ↓ | 1.682e+00±1.4e-01 | **1.535e+00±1.2e-01** |
| PSNR(dB) ↑ | 13.9±0.59 | **17.1±0.78** |
| SSIM ↑ | 0.543±0.35 | **0.944±0.009** |
| hi-freq err frac ↓ | 9.471e-01±4.8e-03 | **8.698e-01±4.1e-03** |
| rel-L2 @T/4 ↓ | **1.743e-02±1.6e-03** | 1.900e-02±3.8e-03 |
| rel-L2 @T/2 ↓ | **6.693e-02±5.1e-03** | 6.795e-02±5.4e-03 |
| rel-L2 @3T/4 ↓ | 2.448e-01±3.0e-02 | **1.979e-01±2.2e-02** |
| rel-L2 @T ↓ | 5.153e-01±3.4e-02 | **3.584e-01±3.2e-02** |
| err-growth T/(T/4) ↓ | 29.6±1.6 | **19.2±2.5** |
| mass-cons err ↓ | 1.992e+01±1.1e+01 | **5.227e-05±2.1e-05** |
| periodic-BC res ↓ | 1.540e+00±7.7e-02 | **1.300e+00±6.3e-02** |
| grad-energy | 6.600e-01±1.4e-02 | 5.019e-01±7.3e-03 |
| params ↓ | **5520** | 5520 |
| train wall(s) ↓ | **63.7±0.31** | 64.2±0.3 |
| infer s/step ↓ | 1.956e-03±1.3e-03 | **1.892e-03±1.2e-03** |
| throughput cells/s ↑ | 1.16e+07 | **1.17e+07** |

### cahn_hilliard - rel-L2 with uncertainty and paired tests

n = 40 paired evaluations (5 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **bounded_multiscale_nca** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs bounded_multiscale_nca (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| bounded_multiscale_nca | 3.5835e-01 | [3.493e-01, 3.676e-01] | 3.5492e-01 | - | - | **reference** |
| multiscale_flux_nca | 5.1518e-01 | [5.055e-01, 5.251e-01] | 5.0803e-01 | +1.568e-01 | 1.8e-12 | worse |
