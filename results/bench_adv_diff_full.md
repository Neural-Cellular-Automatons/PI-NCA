### adv_diff  (grid=12, train_steps=3, eval_steps=8, epochs=25, seeds=1, clip=None)

| metric | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| rel-L2 ↓ | 2.220e-02±0.0e+00 | 1.231e-02±0.0e+00 | 1.816e-02±0.0e+00 | **6.575e-03±0.0e+00** | 7.185e-02±0.0e+00 | 1.351e-01±0.0e+00 |
| MSE ↓ | 2.111e-03±0.0e+00 | 6.490e-04±0.0e+00 | 1.412e-03±0.0e+00 | **1.851e-04±0.0e+00** | 2.852e-02±0.0e+00 | 1.008e-01±0.0e+00 |
| RMSE ↓ | 4.594e-02±0.0e+00 | 2.548e-02±0.0e+00 | 3.757e-02±0.0e+00 | **1.361e-02±0.0e+00** | 1.689e-01±0.0e+00 | 3.175e-01±0.0e+00 |
| MAE ↓ | 2.693e-02±0.0e+00 | 1.388e-02±0.0e+00 | 2.487e-02±0.0e+00 | **9.328e-03±0.0e+00** | 1.062e-01±0.0e+00 | 1.749e-01±0.0e+00 |
| L∞ ↓ | 2.731e-01±0.0e+00 | 2.302e-01±0.0e+00 | 2.068e-01±0.0e+00 | **7.529e-02±0.0e+00** | 1.804e+00±0.0e+00 | 1.723e+00±0.0e+00 |
| PSNR(dB) ↑ | 46.6±0 | 51.7±0 | 48.3±0 | **57.2±0** | 38.2±0 | 32.7±0 |
| SSIM ↑ | 1±0 | 1±0 | 1±0 | **1±0** | 0.996±0 | 0.988±0 |
| hi-freq err frac ↓ | 2.489e-01±0.0e+00 | 2.472e-01±0.0e+00 | 3.601e-01±0.0e+00 | 1.567e-01±0.0e+00 | 3.123e-01±0.0e+00 | **8.708e-02±0.0e+00** |
| rel-L2 @T/4 ↓ | 6.495e-03±0.0e+00 | 3.853e-03±0.0e+00 | 6.452e-03±0.0e+00 | **2.224e-03±0.0e+00** | 4.192e-02±0.0e+00 | 4.993e-02±0.0e+00 |
| rel-L2 @T/2 ↓ | 1.195e-02±0.0e+00 | 6.863e-03±0.0e+00 | 1.118e-02±0.0e+00 | **3.210e-03±0.0e+00** | 4.853e-02±0.0e+00 | 7.390e-02±0.0e+00 |
| rel-L2 @3T/4 ↓ | 1.715e-02±0.0e+00 | 9.626e-03±0.0e+00 | 1.499e-02±0.0e+00 | **4.333e-03±0.0e+00** | 5.915e-02±0.0e+00 | 1.036e-01±0.0e+00 |
| rel-L2 @T ↓ | 2.220e-02±0.0e+00 | 1.231e-02±0.0e+00 | 1.816e-02±0.0e+00 | **6.575e-03±0.0e+00** | 7.185e-02±0.0e+00 | 1.351e-01±0.0e+00 |
| err-growth T/(T/4) ↓ | 3.42±0 | 3.2±0 | 2.81±0 | 2.96±0 | **1.71±0** | 2.71±0 |
| mass-cons err ↓ | 9.508e+00±0.0e+00 | **3.891e-04±0.0e+00** | 1.595e-03±0.0e+00 | 1.202e+00±0.0e+00 | 6.014e+00±0.0e+00 | 7.451e-01±0.0e+00 |
| periodic-BC res ↓ | 3.990e-01±0.0e+00 | 3.956e-01±0.0e+00 | **3.941e-01±0.0e+00** | 3.958e-01±0.0e+00 | 8.515e-01±0.0e+00 | 8.695e-01±0.0e+00 |
| grad-energy | 7.508e-01±0.0e+00 | 7.332e-01±0.0e+00 | 7.323e-01±0.0e+00 | 7.335e-01±0.0e+00 | 2.712e+00±0.0e+00 | 2.982e+00±0.0e+00 |
| params ↓ | 6784 | 4576 | 5520 | 592897 | 5364 | **1** |
| train wall(s) ↓ | 88±0 | 85.9±0 | 48.6±0 | 96.3±0 | 1.98±0 | **1.29±0** |
| infer s/step ↓ | 1.291e-03±0.0e+00 | 6.815e-04±0.0e+00 | 1.038e-03±0.0e+00 | 1.001e-02±0.0e+00 | 1.229e-03±0.0e+00 | **5.900e-06±0.0e+00** |
| throughput cells/s ↑ | 3.57e+06 | 6.76e+06 | 4.44e+06 | 4.60e+05 | 9.37e+05 | **1.95e+08** |

### adv_diff - rel-L2 with uncertainty and paired tests

n = 8 paired evaluations (1 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **resnet_iso** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs resnet_iso (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| resnet_iso | 6.9907e-02 | [6.466e-02, 7.557e-02] | 6.8086e-02 | - | - | **reference** |
| identity | 1.3732e-01 | [1.290e-01, 1.457e-01] | 1.3854e-01 | +6.741e-02 | 0.0078 | worse |
