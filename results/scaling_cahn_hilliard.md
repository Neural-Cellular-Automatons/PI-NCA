### Scaling behaviour - cahn_hilliard

Each sweep varies one axis with everything else fixed at {'grid': 48, 'rollout': 12, 'epochs': 1200, 'eval_steps': 48, 'batch': 32, 'n_eval': 32}, seeds [42]. The question is not what the errors are, it is whether the *ordering* survives the change of scale: a ranking that flips along an axis was a statement about that operating point, not about the architectures.

#### grid

| grid | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| 24 | 0.652 | 0.548 | 0.535 | 0.737 | 0.593 | 0.919 |
| 32 | 0.66 | 0.527 | 0.486 | 0.57 | 0.597 | 0.918 |
| 48 | 0.664 | 0.515 | 0.563 | 0.638 | 0.594 | 0.918 |

Kendall's tau between the ranking at grid=24 and grid=48: **0.73** -- **ordering changes with scale**. The best architecture changes from `multiscale_flux_nca` to `pi_nca`.

#### rollout

| rollout | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| 4 | 0.667 | 0.485 | 0.56 | 0.733 | 0.667 | 0.918 |
| 8 | 0.665 | 0.522 | 0.511 | 0.659 | 0.65 | 0.918 |
| 12 | 0.663 | 0.514 | 0.531 | 0.639 | 0.594 | 0.918 |

Kendall's tau between the ranking at rollout=4 and rollout=12: **0.73** -- **ordering changes with scale**.

#### epochs

| epochs | plain_nca | pi_nca | multiscale_flux_nca | fno | resnet_iso | identity |
|---|---|---|---|---|---|---|
| 300 | 0.634 | 0.61 | 0.583 | 0.733 | 0.732 | 0.918 |
| 600 | 0.652 | 0.608 | 0.553 | 0.727 | 0.663 | 0.918 |
| 1200 | 0.663 | 0.515 | 0.542 | 0.637 | 0.594 | 0.918 |

Kendall's tau between the ranking at epochs=300 and epochs=1200: **0.60** -- **ordering changes with scale**. The best architecture changes from `multiscale_flux_nca` to `pi_nca`.

> **Any headline ranking should be read as conditional on the operating point**, because the ordering is not stable along: `grid`, `rollout`, `epochs`. Reporting a winner without the scale it was measured at would be misleading.
