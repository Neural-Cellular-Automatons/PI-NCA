### heat  (grid=48, train_steps=12, eval_steps=48, epochs=1200, seeds=5, clip=None)

| metric | abl_k3 | abl_k5 | abl_multiscale |
|---|---|---|---|
| rel-L2 ↓ | 8.957e-03±4.7e-03 | 5.842e-03±2.2e-03 | **4.121e-03±1.4e-03** |
| MSE ↓ | 4.323e-04±3.7e-04 | 1.751e-04±1.4e-04 | **8.157e-05±5.2e-05** |
| RMSE ↓ | 1.880e-02±9.9e-03 | 1.241e-02±5.1e-03 | **8.644e-03±2.9e-03** |
| MAE ↓ | 1.031e-02±4.7e-03 | 7.103e-03±2.9e-03 | **5.364e-03±1.9e-03** |
| L∞ ↓ | 1.556e-01±1.2e-01 | 1.158e-01±5.9e-02 | **6.811e-02±1.7e-02** |
| PSNR(dB) ↑ | 56.6±5.9 | 59.4±3.2 | **62.4±3.8** |
| SSIM ↑ | 1±6.1e-05 | 1±2.1e-05 | **1±9.6e-06** |
| hi-freq err frac ↓ | 3.804e-01±5.0e-01 | 4.748e-01±3.6e-01 | **2.375e-01±1.8e-01** |
| rel-L2 @T/4 ↓ | 1.568e-03±8.3e-04 | **1.373e-03±2.5e-04** | 1.468e-03±3.7e-04 |
| rel-L2 @T/2 ↓ | 2.949e-03±1.6e-03 | **2.419e-03±3.9e-04** | 2.504e-03±7.7e-04 |
| rel-L2 @3T/4 ↓ | 4.670e-03±2.2e-03 | 3.564e-03±4.6e-04 | **3.353e-03±1.1e-03** |
| rel-L2 @T ↓ | 8.957e-03±4.7e-03 | 5.842e-03±2.2e-03 | **4.121e-03±1.4e-03** |
| err-growth T/(T/4) ↓ | 6.76±5.1 | 4.4±2 | **2.75±0.33** |
| mass-cons err ↓ | **5.798e-05±2.7e-05** | 8.545e-05±4.4e-05 | 1.129e-04±4.0e-05 |
| periodic-BC res ↓ | 1.951e-01±2.6e-02 | 1.942e-01±2.6e-02 | **1.937e-01±2.6e-02** |
| grad-energy | 1.687e-01±1.7e-02 | 1.690e-01±1.7e-02 | 1.678e-01±1.8e-02 |
| params ↓ | **4576** | 5088 | 9312 |
| train wall(s) ↓ | **14.9±0.17** | 15.1±1 | 70.8±1.1 |
| infer s/step ↓ | **1.264e-03±5.1e-05** | 2.113e-03±2.1e-03 | 1.909e-03±1.1e-03 |
| throughput cells/s ↑ | **1.46e+07** | 1.34e+07 | 1.13e+07 |

### heat - rel-L2 with uncertainty and paired tests

n = 40 paired evaluations (5 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_multiscale** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 2 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_multiscale (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_multiscale | 4.1620e-03 | [3.747e-03, 4.581e-03] | 3.9149e-03 | - | - | **reference** |
| abl_k5 | 5.7931e-03 | [5.149e-03, 6.511e-03] | 5.1153e-03 | +1.631e-03 | 0.015 | worse |
| abl_k3 | 8.7937e-03 | [7.471e-03, 1.012e-02] | 9.3769e-03 | +4.632e-03 | 2.4e-06 | worse |
