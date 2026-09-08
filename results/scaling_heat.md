### Scaling behaviour - heat

Each sweep varies one axis with everything else fixed at {'grid': 48, 'rollout': 12, 'epochs': 1200, 'eval_steps': 48, 'batch': 32, 'n_eval': 32}, seeds [42]. The question is not what the errors are, it is whether the *ordering* survives the change of scale: a ranking that flips along an axis was a statement about that operating point, not about the architectures.

#### grid

| grid | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| 24 | 0.19 | 0.0117 | 0.00439 | 0.00614 | 0.0101 | 0.593 |
| 32 | 0.0836 | 0.00855 | 0.0186 | 0.00434 | 0.0152 | 0.37 |
| 48 | 0.0519 | 0.00509 | 0.00351 | 0.00239 | 0.00693 | 0.189 |

Kendall's tau between the ranking at grid=24 and grid=48: **0.73** -- **ordering changes with scale**. The best architecture changes from `multiscale_flux_nca` to `fno`.

#### rollout

| rollout | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| 4 | 0.0662 | 0.00895 | 0.00501 | 0.00293 | 0.00587 | 0.189 |
| 8 | 0.0578 | 0.00635 | 0.00511 | 0.00268 | 0.00667 | 0.189 |
| 12 | 0.0684 | 0.00517 | 0.00346 | 0.00239 | 0.00693 | 0.189 |

Kendall's tau between the ranking at rollout=4 and rollout=12: **0.87** -- ordering is stable.

#### epochs

| epochs | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| 300 | 0.0368 | 0.0101 | 0.00451 | 0.00266 | 0.00815 | 0.189 |
| 600 | 0.0525 | 0.00536 | 0.00602 | 0.00288 | 0.0076 | 0.189 |
| 1200 | 0.0556 | 0.00509 | 0.00344 | 0.00239 | 0.00693 | 0.189 |

Kendall's tau between the ranking at epochs=300 and epochs=1200: **0.87** -- ordering is stable.

> **Any headline ranking should be read as conditional on the operating point**, because the ordering is not stable along: `grid`. Reporting a winner without the scale it was measured at would be misleading.
