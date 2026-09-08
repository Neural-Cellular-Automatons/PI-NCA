### Scaling behaviour - navier_stokes

Each sweep varies one axis with everything else fixed at {'grid': 48, 'rollout': 12, 'epochs': 1200, 'eval_steps': 48, 'batch': 32, 'n_eval': 32}, seeds [42]. The question is not what the errors are, it is whether the *ordering* survives the change of scale: a ranking that flips along an axis was a statement about that operating point, not about the architectures.

#### grid

| grid | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| 24 | 0.453 | 0.433 | 0.192 | 0.0277 | 0.182 | 0.478 |
| 32 | 0.481 | 0.419 | 0.23 | 0.0259 | 0.282 | 0.499 |
| 48 | 0.466 | 0.441 | 0.301 | 0.0452 | 0.333 | 0.506 |

Kendall's tau between the ranking at grid=24 and grid=48: **0.87** -- ordering is stable.

#### rollout

| rollout | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| 4 | 0.469 | 1.03 | 0.549 | 0.0511 | 1.38 | 0.506 |
| 8 | 0.46 | 0.472 | 0.313 | 0.0452 | 0.738 | 0.506 |
| 12 | 0.458 | 0.437 | 0.301 | 0.0452 | 0.333 | 0.506 |

Kendall's tau between the ranking at rollout=4 and rollout=12: **0.07** -- **ordering changes with scale**.

#### epochs

| epochs | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| 300 | 0.464 | 0.459 | 0.335 | 0.0766 | 0.554 | 0.506 |
| 600 | 0.457 | 0.448 | 0.296 | 0.0554 | 0.455 | 0.506 |
| 1200 | 0.458 | 0.434 | 0.301 | 0.0453 | 0.333 | 0.506 |

Kendall's tau between the ranking at epochs=300 and epochs=1200: **0.60** -- **ordering changes with scale**.

> **Any headline ranking should be read as conditional on the operating point**, because the ordering is not stable along: `rollout`, `epochs`. Reporting a winner without the scale it was measured at would be misleading.
