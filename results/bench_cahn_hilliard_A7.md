### cahn_hilliard  (grid=48, train_steps=12, eval_steps=48, epochs=1200, seeds=5, clip=None)

| metric | abl_proj_none | abl_proj_uniform | abl_proj_headroom |
|---|---|---|---|
| rel-L2 ↓ | 3.589e-01±2.7e-02 | 3.614e-01±4.7e-02 | **3.583e-01±3.0e-02** |
| MSE ↓ | 7.936e-02±1.2e-02 | 8.123e-02±2.1e-02 | **7.917e-02±1.3e-02** |
| RMSE ↓ | 2.811e-01±2.1e-02 | 2.830e-01±3.7e-02 | **2.806e-01±2.4e-02** |
| MAE ↓ | 2.204e-01±1.4e-02 | 2.211e-01±2.6e-02 | **2.191e-01±1.6e-02** |
| L∞ ↓ | **1.543e+00±7.6e-02** | 1.547e+00±1.6e-01 | 1.547e+00±9.2e-02 |
| PSNR(dB) ↑ | 17.1±0.67 | 17±1.2 | **17.1±0.76** |
| SSIM ↑ | 0.606±0.38 | 0.943±0.013 | **0.944±0.0085** |
| hi-freq err frac ↓ | **8.688e-01±2.9e-03** | 8.743e-01±5.8e-03 | 8.728e-01±3.6e-03 |
| rel-L2 @T/4 ↓ | 1.778e-02±2.3e-03 | 1.936e-02±2.7e-03 | **1.694e-02±1.1e-03** |
| rel-L2 @T/2 ↓ | 6.604e-02±3.6e-03 | 7.127e-02±6.3e-03 | **6.600e-02±3.6e-03** |
| rel-L2 @3T/4 ↓ | 1.961e-01±1.6e-02 | 1.955e-01±2.6e-02 | **1.933e-01±1.6e-02** |
| rel-L2 @T ↓ | 3.589e-01±2.7e-02 | 3.614e-01±4.7e-02 | **3.583e-01±3.0e-02** |
| err-growth T/(T/4) ↓ | 20.3±1.4 | **19±3.8** | 21.1±1.1 |
| mass-cons err ↓ | 1.855e+01±1.1e+01 | 1.065e-04±2.3e-05 | **5.395e-05±9.0e-06** |
| periodic-BC res ↓ | 1.301e+00±6.4e-02 | 1.300e+00±8.2e-02 | **1.296e+00±6.2e-02** |
| grad-energy | 5.053e-01±4.6e-03 | 4.958e-01±1.4e-02 | 4.946e-01±7.8e-03 |
| params ↓ | **5520** | 5520 | 5520 |
| train wall(s) ↓ | **63.8±0.17** | 64±0.23 | 64.2±0.22 |
| infer s/step ↓ | 1.910e-03±1.2e-03 | 1.901e-03±1.1e-03 | **1.405e-03±1.0e-04** |
| throughput cells/s ↑ | 1.18e+07 | 1.15e+07 | **1.32e+07** |

### cahn_hilliard - rel-L2 with uncertainty and paired tests

n = 40 paired evaluations (5 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_proj_headroom** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 2 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_proj_headroom (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_proj_headroom | 3.5819e-01 | [3.495e-01, 3.668e-01] | 3.5656e-01 | - | - | **reference** |
| abl_proj_none | 3.5885e-01 | [3.510e-01, 3.665e-01] | 3.5987e-01 | +6.571e-04 | 0.38 | tie |
| abl_proj_uniform | 3.6132e-01 | [3.481e-01, 3.743e-01] | 3.5551e-01 | +3.129e-03 | 0.31 | tie |

Statistically indistinguishable from abl_proj_headroom at this sample size: `abl_proj_none`, `abl_proj_uniform`.
