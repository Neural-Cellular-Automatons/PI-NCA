### Scaling behaviour - cahn_hilliard

Each sweep varies one axis with everything else fixed at {'grid': 12, 'rollout': 3, 'epochs': 25, 'eval_steps': 8, 'batch': 4, 'n_eval': 4}, seeds [42]. The question is not what the errors are, it is whether the *ordering* survives the change of scale: a ranking that flips along an axis was a statement about that operating point, not about the architectures.

#### grid

| grid | plain_nca | fno | identity |
|---|---|---|---|
| 12 | 0.161 | 0.126 | 0.293 |
| 16 | 0.137 | 0.122 | 0.255 |

Kendall's tau between the ranking at grid=12 and grid=16: **1.00** -- ordering is stable.

#### rollout

| rollout | plain_nca | fno | identity |
|---|---|---|---|
| 2 | 0.163 | 0.144 | 0.293 |
| 3 | 0.161 | 0.126 | 0.293 |

Kendall's tau between the ranking at rollout=2 and rollout=3: **1.00** -- ordering is stable.

#### epochs

| epochs | plain_nca | fno | identity |
|---|---|---|---|
| 15 | 0.186 | 0.197 | 0.293 |
| 25 | 0.161 | 0.126 | 0.293 |

Kendall's tau between the ranking at epochs=15 and epochs=25: **0.33** -- **ordering changes with scale**. The best architecture changes from `plain_nca` to `fno`.

> **Any headline ranking should be read as conditional on the operating point**, because the ordering is not stable along: `epochs`. Reporting a winner without the scale it was measured at would be misleading.
