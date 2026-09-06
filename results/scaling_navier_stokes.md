### Scaling behaviour - navier_stokes

Each sweep varies one axis with everything else fixed at {'grid': 12, 'rollout': 3, 'epochs': 25, 'eval_steps': 8, 'batch': 4, 'n_eval': 4}, seeds [42]. The question is not what the errors are, it is whether the *ordering* survives the change of scale: a ranking that flips along an axis was a statement about that operating point, not about the architectures.

#### grid

| grid | plain_nca | fno | identity |
|---|---|---|---|
| 12 | 0.0975 | 0.098 | 0.0996 |
| 16 | 0.0912 | 0.0902 | 0.0913 |

Kendall's tau between the ranking at grid=12 and grid=16: **0.33** -- **ordering changes with scale**. The best architecture changes from `plain_nca` to `fno`.

#### rollout

| rollout | plain_nca | fno | identity |
|---|---|---|---|
| 2 | 0.0976 | 0.0979 | 0.0996 |
| 3 | 0.0975 | 0.098 | 0.0996 |

Kendall's tau between the ranking at rollout=2 and rollout=3: **1.00** -- ordering is stable.

#### epochs

| epochs | plain_nca | fno | identity |
|---|---|---|---|
| 15 | 0.0995 | 0.0991 | 0.0996 |
| 25 | 0.0975 | 0.098 | 0.0996 |

Kendall's tau between the ranking at epochs=15 and epochs=25: **0.33** -- **ordering changes with scale**. The best architecture changes from `fno` to `plain_nca`.

> **Any headline ranking should be read as conditional on the operating point**, because the ordering is not stable along: `grid`, `epochs`. Reporting a winner without the scale it was measured at would be misleading.
