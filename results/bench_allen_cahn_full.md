### allen_cahn  (grid=12, train_steps=3, eval_steps=8, epochs=25, seeds=1, clip=None)

| metric | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| rel-L2 ↓ | 4.907e-02±0.0e+00 | 4.922e-02±0.0e+00 | 5.274e-02±0.0e+00 | **6.848e-03±0.0e+00** | 2.642e-02±0.0e+00 | 2.663e-02±0.0e+00 |
| MSE ↓ | 2.313e-03±0.0e+00 | 2.327e-03±0.0e+00 | 2.673e-03±0.0e+00 | **4.506e-05±0.0e+00** | 6.834e-04±0.0e+00 | 6.947e-04±0.0e+00 |
| RMSE ↓ | 4.810e-02±0.0e+00 | 4.824e-02±0.0e+00 | 5.170e-02±0.0e+00 | **6.713e-03±0.0e+00** | 2.614e-02±0.0e+00 | 2.636e-02±0.0e+00 |
| MAE ↓ | 3.858e-02±0.0e+00 | 3.857e-02±0.0e+00 | 4.139e-02±0.0e+00 | **5.330e-03±0.0e+00** | 2.092e-02±0.0e+00 | 2.099e-02±0.0e+00 |
| L∞ ↓ | 1.746e-01±0.0e+00 | 1.708e-01±0.0e+00 | 1.897e-01±0.0e+00 | **2.267e-02±0.0e+00** | 1.092e-01±0.0e+00 | 1.125e-01±0.0e+00 |
| PSNR(dB) ↑ | 32.4±0 | 32.4±0 | 31.8±0 | **49.5±0** | 38.2±0 | 38.1±0 |
| SSIM ↑ | 0.998±0 | 0.995±0 | 0.995±0 | 0.977±0 | 0.987±0 | **1±0** |
| hi-freq err frac ↓ | 7.078e-01±0.0e+00 | 7.147e-01±0.0e+00 | 7.315e-01±0.0e+00 | 7.531e-01±0.0e+00 | **7.030e-01±0.0e+00** | 7.208e-01±0.0e+00 |
| rel-L2 @T/4 ↓ | 3.287e-02±0.0e+00 | 3.255e-02±0.0e+00 | 3.282e-02±0.0e+00 | **7.403e-03±0.0e+00** | 8.480e-03±0.0e+00 | 8.557e-03±0.0e+00 |
| rel-L2 @T/2 ↓ | 4.394e-02±0.0e+00 | 4.351e-02±0.0e+00 | 4.445e-02±0.0e+00 | **7.548e-03±0.0e+00** | 1.554e-02±0.0e+00 | 1.568e-02±0.0e+00 |
| rel-L2 @3T/4 ↓ | 4.768e-02±0.0e+00 | 4.741e-02±0.0e+00 | 4.944e-02±0.0e+00 | **7.028e-03±0.0e+00** | 2.144e-02±0.0e+00 | 2.163e-02±0.0e+00 |
| rel-L2 @T ↓ | 4.907e-02±0.0e+00 | 4.922e-02±0.0e+00 | 5.274e-02±0.0e+00 | **6.848e-03±0.0e+00** | 2.642e-02±0.0e+00 | 2.663e-02±0.0e+00 |
| err-growth T/(T/4) ↓ | 1.49±0 | 1.51±0 | 1.61±0 | **0.925±0** | 3.12±0 | 3.11±0 |
| mass-cons err ↓ | 2.547e-01±0.0e+00 | 1.431e-05±0.0e+00 | 8.100e-05±0.0e+00 | 7.072e-01±0.0e+00 | 4.385e-01±0.0e+00 | **0.000e+00±0.0e+00** |
| periodic-BC res ↓ | 9.890e-01±0.0e+00 | 9.823e-01±0.0e+00 | 9.869e-01±0.0e+00 | **9.642e-01±0.0e+00** | 9.890e-01±0.0e+00 | 9.904e-01±0.0e+00 |
| grad-energy | 9.758e-01±0.0e+00 | 9.683e-01±0.0e+00 | 9.620e-01±0.0e+00 | 9.695e-01±0.0e+00 | 1.004e+00±0.0e+00 | 1.007e+00±0.0e+00 |
| params ↓ | 6784 | 4576 | 5520 | 592897 | 5364 | **1** |
| train wall(s) ↓ | 64.5±0 | 54.4±0 | 45.4±0 | 84.2±0 | 2.58±0 | **1.38±0** |
| infer s/step ↓ | 6.710e-04±0.0e+00 | 8.806e-04±0.0e+00 | 7.864e-04±0.0e+00 | 4.226e-03±0.0e+00 | 1.014e-03±0.0e+00 | **7.700e-06±0.0e+00** |
| throughput cells/s ↑ | 6.87e+06 | 5.23e+06 | 5.86e+06 | 1.09e+06 | 1.14e+06 | **1.50e+08** |

### allen_cahn - rel-L2 with uncertainty and paired tests

n = 8 paired evaluations (1 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **resnet_iso** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 1 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs resnet_iso (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| resnet_iso | 2.6389e-02 | [2.562e-02, 2.723e-02] | 2.6238e-02 | - | - | **reference** |
| identity | 2.6606e-02 | [2.587e-02, 2.742e-02] | 2.6446e-02 | +2.171e-04 | 0.016 | worse |
