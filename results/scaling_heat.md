### Scaling behaviour - heat

Each sweep varies one axis with everything else fixed at {'grid': 16, 'rollout': 4, 'epochs': 60, 'eval_steps': 16, 'batch': 4, 'n_eval': 8}, seeds [42]. The question is not what the errors are, it is whether the *ordering* survives the change of scale: a ranking that flips along an axis was a statement about that operating point, not about the architectures.

#### grid

| grid | plain_nca | pi_nca | fno | identity |
|---|---|---|---|---|
| 16 | 0.128 | 0.0595 | 0.0599 | 0.49 |
| 24 | 0.0986 | 0.025 | 0.0225 | 0.256 |

Kendall's tau between the ranking at grid=16 and grid=24: **0.67** -- **ordering changes with scale**. The best architecture changes from `pi_nca` to `fno`.

#### rollout

| rollout | plain_nca | pi_nca | fno | identity |
|---|---|---|---|---|
| 4 | 0.128 | 0.0595 | 0.0599 | 0.49 |
| 8 | 0.152 | 0.0626 | 0.0711 | 0.49 |

Kendall's tau between the ranking at rollout=4 and rollout=8: **1.00** -- ordering is stable.

#### epochs

| epochs | plain_nca | pi_nca | fno | identity |
|---|---|---|---|---|
| 60 | 0.128 | 0.0595 | 0.0599 | 0.49 |
| 150 | 0.0551 | 0.0369 | 0.0462 | 0.49 |

Kendall's tau between the ranking at epochs=60 and epochs=150: **1.00** -- ordering is stable.

> **Any headline ranking should be read as conditional on the operating point**, because the ordering is not stable along: `grid`. Reporting a winner without the scale it was measured at would be misleading.
