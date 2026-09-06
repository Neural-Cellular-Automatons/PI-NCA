### allen_cahn  (grid=16, train_steps=6, eval_steps=12, epochs=150, seeds=2, clip=None)

| metric | abl_proj_none | abl_proj_uniform | abl_proj_headroom |
|---|---|---|---|
| rel-L2 ↓ | **3.291e-02±8.2e-04** | 3.291e-02±8.2e-04 | 3.291e-02±8.2e-04 |
| MSE ↓ | **1.053e-03±5.3e-05** | 1.053e-03±5.3e-05 | 1.053e-03±5.3e-05 |
| RMSE ↓ | **3.245e-02±8.2e-04** | 3.245e-02±8.2e-04 | 3.245e-02±8.2e-04 |
| MAE ↓ | 2.580e-02±6.0e-04 | **2.580e-02±6.0e-04** | 2.580e-02±6.0e-04 |
| L∞ ↓ | 1.259e-01±1.5e-03 | 1.259e-01±1.5e-03 | **1.259e-01±1.5e-03** |
| PSNR(dB) ↑ | **36.1±0.24** | 36.1±0.24 | 36.1±0.24 |
| SSIM ↑ | 0.994±0.0074 | **0.994±0.0074** | 0.994±0.0074 |
| hi-freq err frac ↓ | **6.815e-01±8.0e-03** | 6.815e-01±8.0e-03 | 6.815e-01±8.0e-03 |
| rel-L2 @T/4 ↓ | **1.183e-02±3.2e-04** | 1.183e-02±3.2e-04 | 1.183e-02±3.2e-04 |
| rel-L2 @T/2 ↓ | **2.076e-02±5.6e-04** | 2.076e-02±5.6e-04 | 2.076e-02±5.6e-04 |
| rel-L2 @3T/4 ↓ | **2.761e-02±7.2e-04** | 2.761e-02±7.2e-04 | 2.761e-02±7.2e-04 |
| rel-L2 @T ↓ | **3.291e-02±8.2e-04** | 3.291e-02±8.2e-04 | 3.291e-02±8.2e-04 |
| err-growth T/(T/4) ↓ | 2.78±0.0057 | 2.78±0.0057 | **2.78±0.0057** |
| mass-cons err ↓ | **4.776e-06±5.8e-07** | 8.784e-06±3.7e-06 | 5.834e-06±1.8e-06 |
| periodic-BC res ↓ | 9.907e-01±4.7e-02 | **9.907e-01±4.7e-02** | 9.907e-01±4.7e-02 |
| grad-energy | 9.595e-01±1.9e-02 | 9.595e-01±1.9e-02 | 9.595e-01±1.9e-02 |
| params ↓ | **5520** | 5520 | 5520 |
| train wall(s) ↓ | 10.4±0.76 | 9.17±0.34 | **8.72±1.1** |
| infer s/step ↓ | 1.181e-03±4.1e-04 | 9.575e-04±1.2e-04 | **9.046e-04±3.5e-04** |
| throughput cells/s ↑ | 1.84e+06 | 2.16e+06 | **2.45e+06** |

### allen_cahn - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_proj_none** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 2 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_proj_none (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_proj_none | 3.2864e-02 | [3.199e-02, 3.374e-02] | 3.2960e-02 | - | - | **reference** |
| abl_proj_uniform | 3.2864e-02 | [3.199e-02, 3.374e-02] | 3.2960e-02 | +7.451e-09 | 0.2 | tie |
| abl_proj_headroom | 3.2864e-02 | [3.199e-02, 3.374e-02] | 3.2960e-02 | +2.410e-08 | 0.018 | tie |

Statistically indistinguishable from abl_proj_none at this sample size: `abl_proj_uniform`, `abl_proj_headroom`.
