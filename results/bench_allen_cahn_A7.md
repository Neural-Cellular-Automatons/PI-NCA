### allen_cahn  (grid=48, train_steps=12, eval_steps=48, epochs=1200, seeds=5, clip=None)

| metric | abl_proj_none | abl_proj_uniform | abl_proj_headroom |
|---|---|---|---|
| rel-L2 ↓ | 4.949e-02±2.6e-04 | 4.952e-02±2.5e-04 | **4.942e-02±1.7e-04** |
| MSE ↓ | 2.353e-03±2.5e-05 | 2.356e-03±2.4e-05 | **2.347e-03±1.6e-05** |
| RMSE ↓ | 4.850e-02±2.6e-04 | 4.854e-02±2.5e-04 | **4.844e-02±1.7e-04** |
| MAE ↓ | 3.870e-02±2.6e-04 | 3.872e-02±2.4e-04 | **3.864e-02±2.0e-04** |
| L∞ ↓ | 1.928e-01±8.9e-03 | 1.936e-01±9.6e-03 | **1.910e-01±1.2e-02** |
| PSNR(dB) ↑ | 32.3±0.045 | 32.3±0.044 | **32.3±0.03** |
| SSIM ↑ | 0.996±0.0023 | 0.996±0.0023 | **0.996±0.0023** |
| hi-freq err frac ↓ | 7.252e-01±5.7e-03 | 7.256e-01±5.7e-03 | **7.245e-01±5.8e-03** |
| rel-L2 @T/4 ↓ | 3.316e-02±2.5e-04 | **3.313e-02±3.1e-04** | 3.313e-02±2.1e-04 |
| rel-L2 @T/2 ↓ | 4.429e-02±4.0e-04 | 4.425e-02±4.7e-04 | **4.424e-02±2.6e-04** |
| rel-L2 @3T/4 ↓ | 4.805e-02±4.0e-04 | 4.803e-02±4.5e-04 | **4.799e-02±2.3e-04** |
| rel-L2 @T ↓ | 4.949e-02±2.6e-04 | 4.952e-02±2.5e-04 | **4.942e-02±1.7e-04** |
| err-growth T/(T/4) ↓ | 1.49±0.0039 | 1.49±0.0091 | **1.49±0.0067** |
| mass-cons err ↓ | 1.070e-05±3.5e-06 | **8.059e-06±3.3e-06** | 1.035e-05±4.0e-06 |
| periodic-BC res ↓ | 1.010e+00±2.2e-02 | 1.009e+00±2.1e-02 | **1.009e+00±1.9e-02** |
| grad-energy | 9.602e-01±6.8e-03 | 9.596e-01±7.8e-03 | 9.598e-01±7.0e-03 |
| params ↓ | **5520** | 5520 | 5520 |
| train wall(s) ↓ | **64±0.24** | 64.3±0.31 | 64.6±0.25 |
| infer s/step ↓ | **1.367e-03±4.6e-05** | 1.412e-03±5.7e-05 | 1.407e-03±6.4e-05 |
| throughput cells/s ↑ | **1.35e+07** | 1.31e+07 | 1.31e+07 |

### allen_cahn - rel-L2 with uncertainty and paired tests

n = 40 paired evaluations (5 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **abl_proj_headroom** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 2 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs abl_proj_headroom (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| abl_proj_headroom | 4.9416e-02 | [4.921e-02, 4.962e-02] | 4.9345e-02 | - | - | **reference** |
| abl_proj_none | 4.9481e-02 | [4.927e-02, 4.970e-02] | 4.9401e-02 | +6.453e-05 | 0.23 | tie |
| abl_proj_uniform | 4.9513e-02 | [4.931e-02, 4.973e-02] | 4.9439e-02 | +9.622e-05 | 0.004 | worse |

Statistically indistinguishable from abl_proj_headroom at this sample size: `abl_proj_none`.
