### navier_stokes  (grid=48, train_steps=12, eval_steps=48, epochs=1200, seeds=5, clip=None)

| metric | abl_k3 | abl_k5 | abl_multiscale |
|---|---|---|---|
| rel-L2 ↓ | 6.793e-01±2.9e-01 | 6.674e-01±3.1e-01 | **5.825e-01±3.8e-01** |
| MSE ↓ | 6.733e+02±1.5e+03 | **6.732e+02±1.5e+03** | 6.733e+02±1.5e+03 |
| RMSE ↓ | 1.240e+01±2.5e+01 | 1.240e+01±2.5e+01 | **1.235e+01±2.6e+01** |
| MAE ↓ | 5.904e-01±9.2e-01 | 5.828e-01±9.2e-01 | **5.479e-01±9.3e-01** |
| L∞ ↓ | 8.248e+02±1.8e+03 | 8.249e+02±1.8e+03 | **8.246e+02±1.8e+03** |
| PSNR(dB) ↑ | 32.4±7.7 | 32.6±7.5 | **34.6±5.7** |
| SSIM ↑ | 0.537±0.47 | 0.544±0.48 | **0.579±0.51** |
| hi-freq err frac ↓ | **4.010e-01±5.3e-01** | 4.231e-01±5.1e-01 | 4.202e-01±5.1e-01 |
| rel-L2 @T/4 ↓ | 1.623e-01±2.6e-02 | 1.382e-01±2.4e-02 | **1.043e-01±1.2e-02** |
| rel-L2 @T/2 ↓ | 2.968e-01±4.8e-02 | 2.692e-01±4.3e-02 | **1.906e-01±2.8e-02** |
| rel-L2 @3T/4 ↓ | 4.066e-01±6.1e-02 | 3.803e-01±6.0e-02 | **2.724e-01±4.6e-02** |
| rel-L2 @T ↓ | 6.793e-01±2.9e-01 | 6.674e-01±3.1e-01 | **5.825e-01±3.8e-01** |
| err-growth T/(T/4) ↓ | **4.11±1.4** | 4.72±1.7 | 5.45±3.3 |
| mass-cons err ↓ | **4.068e-06±1.6e-06** | 4.831e-06±1.8e-06 | 4.922e-06±1.6e-06 |
| periodic-BC res ↓ | 5.711e-02±3.9e-03 | 7.273e-02±1.4e-02 | **5.589e-02±6.5e-03** |
| grad-energy | 1.285e-02±9.4e-04 | 1.782e-02±2.7e-03 | 1.532e-02±3.1e-03 |
| params ↓ | **4576** | 5088 | 9312 |
| train wall(s) ↓ | 18.1±0.16 | **18.1±0.25** | 74.1±0.17 |
| infer s/step ↓ | **1.267e-03±7.7e-05** | 1.318e-03±8.5e-05 | 2.217e-03±1.8e-03 |
| throughput cells/s ↑ | **1.46e+07** | 1.40e+07 | 1.11e+07 |

### navier_stokes - rel-L2 with uncertainty and paired tests

n = 40 paired evaluations (5 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_multiscale** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 2 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_multiscale (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_multiscale | 2.8191e-01 | [2.312e-01, 3.459e-01] | 2.4677e-01 | - | - | **reference** |
| abl_k5 | 3.7543e-01 | [3.184e-01, 4.403e-01] | 3.3163e-01 | +9.353e-02 | 2.5e-11 | worse |
| abl_k3 | 3.9313e-01 | [3.341e-01, 4.575e-01] | 3.6393e-01 | +1.112e-01 | 6.7e-10 | worse |
