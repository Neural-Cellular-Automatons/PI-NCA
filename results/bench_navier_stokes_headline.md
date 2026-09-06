### navier_stokes  (grid=16, train_steps=4, eval_steps=12, epochs=40, seeds=2, clip=None)

| metric | plain_nca | pi_nca | fno | fno_small | mc_flux_nca | resnet | resnet_iso |
|---|---|---|---|---|---|---|---|
| rel-L2 ↓ | 1.965e-01±3.6e-02 | 1.876e-01±3.8e-02 | 1.930e-01±3.6e-02 | 1.932e-01±3.6e-02 | 1.866e-01±3.7e-02 | **1.653e-01±2.6e-02** | 1.838e-01±3.7e-02 |
| MSE ↓ | 1.242e-02±5.8e-03 | 1.138e-02±5.7e-03 | 1.199e-02±5.7e-03 | 1.201e-02±5.7e-03 | 1.125e-02±5.5e-03 | **8.718e-03±3.6e-03** | 1.092e-02±5.4e-03 |
| RMSE ↓ | 1.098e-01±2.7e-02 | 1.050e-01±2.7e-02 | 1.079e-01±2.6e-02 | 1.080e-01±2.6e-02 | 1.044e-01±2.7e-02 | **9.233e-02±2.0e-02** | 1.028e-01±2.6e-02 |
| MAE ↓ | 5.364e-02±6.5e-03 | 5.200e-02±7.4e-03 | 5.363e-02±8.1e-03 | 5.231e-02±7.2e-03 | 5.208e-02±6.6e-03 | **4.656e-02±5.3e-03** | 5.168e-02±7.2e-03 |
| L∞ ↓ | 7.583e-01±2.9e-01 | 7.789e-01±2.8e-01 | 7.337e-01±2.3e-01 | 7.732e-01±2.8e-01 | 7.652e-01±2.4e-01 | **6.438e-01±2.4e-01** | 7.699e-01±2.8e-01 |
| PSNR(dB) ↑ | 34.2±0.17 | 34.6±0.03 | 34.3±0.15 | 34.3±0.15 | 34.6±0.06 | **35.7±0.43** | 34.8±0.034 |
| SSIM ↑ | 0.895±0.11 | 0.982±0.0071 | 0.979±0.0046 | 0.975±0.013 | **0.982±0.0069** | 0.862±0.13 | 0.903±0.09 |
| hi-freq err frac ↓ | 9.351e-03±1.3e-03 | 2.529e-02±9.2e-03 | **7.604e-03±6.0e-04** | 7.708e-03±5.0e-04 | 2.812e-02±2.0e-02 | 2.924e-02±8.7e-03 | 2.047e-02±1.9e-04 |
| rel-L2 @T/4 ↓ | 5.048e-02±9.5e-03 | 4.824e-02±9.4e-03 | 4.998e-02±9.5e-03 | 4.993e-02±9.6e-03 | 4.804e-02±9.5e-03 | **4.279e-02±6.9e-03** | 4.715e-02±9.1e-03 |
| rel-L2 @T/2 ↓ | 1.003e-01±1.9e-02 | 9.577e-02±1.9e-02 | 9.902e-02±1.9e-02 | 9.899e-02±1.9e-02 | 9.531e-02±1.9e-02 | **8.467e-02±1.3e-02** | 9.364e-02±1.8e-02 |
| rel-L2 @3T/4 ↓ | 1.490e-01±2.8e-02 | 1.423e-01±2.8e-02 | 1.468e-01±2.8e-02 | 1.468e-01±2.8e-02 | 1.416e-01±2.8e-02 | **1.255e-01±2.0e-02** | 1.393e-01±2.8e-02 |
| rel-L2 @T ↓ | 1.965e-01±3.6e-02 | 1.876e-01±3.8e-02 | 1.930e-01±3.6e-02 | 1.932e-01±3.6e-02 | 1.866e-01±3.7e-02 | **1.653e-01±2.6e-02** | 1.838e-01±3.7e-02 |
| err-growth T/(T/4) ↓ | 3.89±0.014 | 3.89±0.029 | **3.86±0.009** | 3.87±0.018 | 3.88±0.00094 | 3.86±0.013 | 3.9±0.032 |
| mass-cons err ↓ | 7.105e-01±6.6e-01 | **4.772e-06±1.5e-06** | 3.181e-01±1.7e-01 | 4.589e-01±2.4e-01 | 5.052e-06±2.0e-06 | 1.196e+00±5.0e-01 | 7.895e-01±4.0e-01 |
| periodic-BC res ↓ | 1.742e-01±2.8e-02 | **1.717e-01±2.2e-02** | 1.725e-01±2.7e-02 | 1.730e-01±2.7e-02 | 1.744e-01±2.2e-02 | 1.752e-01±2.1e-02 | 1.744e-01±2.6e-02 |
| grad-energy | 1.137e-01±1.5e-02 | 1.120e-01±1.8e-02 | 1.115e-01±1.6e-02 | 1.118e-01±1.5e-02 | 1.119e-01±1.8e-02 | 1.116e-01±1.7e-02 | 1.114e-01±1.4e-02 |
| params ↓ | 6784 | **4576** | 592897 | 8433 | 9936 | 74336 | 5364 |
| train wall(s) ↓ | 4.42±2.4 | 5.89±1.3 | 8.75±0.88 | **2.18±0.32** | 2.8±0.48 | 17.3±0.086 | 2.71±0.25 |
| infer s/step ↓ | 4.948e-04±1.6e-04 | **4.557e-04±3.8e-06** | 2.925e-03±1.9e-04 | 5.223e-04±1.6e-04 | 8.573e-04±4.9e-04 | 4.950e-03±2.8e-03 | 1.490e-03±3.5e-04 |
| throughput cells/s ↑ | 4.38e+06 | **4.49e+06** | 7.02e+05 | 4.12e+06 | 2.86e+06 | 4.94e+05 | 1.41e+06 |

### navier_stokes - rel-L2 with uncertainty and paired tests

n = 16 paired evaluations (2 seed(s) x 8 held-out initial conditions). CIs are 10,000-sample percentile bootstraps. The paired column tests each architecture against **resnet** (best mean) on the SAME initial conditions, using Wilcoxon signed-rank with Holm-Bonferroni across the 6 comparisons in this table; `tie` means the difference is not resolvable at this sample size, not that the means are equal.

Note: the mean here is the unweighted mean of per-IC relative errors, while the ranking table above reports the batch-reduced ratio of norms (which weights high-energy initial conditions more heavily). The two are different estimators of the same quantity and will not print equal numbers; the paired tests need the per-IC form, so it is the one reported with uncertainty.

| architecture | rel-L2 mean | 95% CI | median | vs resnet (mean diff) | p (Holm) | verdict |
|---|---|---|---|---|---|---|
| resnet | 1.3469e-01 | [1.113e-01, 1.587e-01] | 1.2674e-01 | - | - | **reference** |
| resnet_iso | 1.4514e-01 | [1.179e-01, 1.730e-01] | 1.5292e-01 | +1.045e-02 | 0.044 | tie |
| mc_flux_nca | 1.4804e-01 | [1.193e-01, 1.771e-01] | 1.5944e-01 | +1.334e-02 | 0.013 | tie |
| pi_nca | 1.4835e-01 | [1.184e-01, 1.783e-01] | 1.5235e-01 | +1.366e-02 | 0.016 | tie |
| fno_small | 1.5195e-01 | [1.194e-01, 1.836e-01] | 1.5990e-01 | +1.726e-02 | 0.021 | tie |
| fno | 1.5230e-01 | [1.204e-01, 1.834e-01] | 1.5943e-01 | +1.761e-02 | 0.013 | tie |
| plain_nca | 1.5526e-01 | [1.234e-01, 1.867e-01] | 1.6422e-01 | +2.056e-02 | 0.0042 | worse |

Statistically indistinguishable from resnet at this sample size: `resnet_iso`, `mc_flux_nca`, `pi_nca`, `fno_small`, `fno`.
