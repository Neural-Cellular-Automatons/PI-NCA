# Initialization & Training Protocol — "Start From a Better Point"

How the **original** PyTorch implementations initialized and trained, what was missing in the
JAX port, and the protocol now adopted. Grounded in the original source (verified, not assumed).

## What the originals actually did
| Technique | Source (verified) |
|---|---|
| **Single fixed seed = 42** | `torch.manual_seed(42); np.random.seed(42)` in `PI NCA_v1.py`, the Gray-Heat notebook, and `SEED=42` in the v3 notebook. **No multi-seed averaging.** |
| **He / Kaiming-normal init** for ReLU convs | `nn.init.kaiming_normal_(self.perceive.weight, nonlinearity='relu')` (+ proc layers) in the Gray-Heat notebook |
| **Zero-init flux head** (identity start) | `self.process[-1].weight.zero_()` / `.fill_(0.0)` |
| **LR warmup** | `warmup_epochs=200..250, warmup_lr=1e-4..2e-4` (v3 PDE_CONFIGS) |
| **Sample pool + pre-seeding** | `pool_size=256, pool_preseed=300` — *"pre-evolve pool states so training sees developed patterns"* (the v3 "KEY FIX") |

## What the JAX port was doing (and the gap)
The port used Flax's default `lecun_normal` init (not He), **no** warmup, **no** pre-seeding
(fresh random blob ICs every epoch), and **multi-seed mean±std** averaging. The zero-init head
was already present.

## Changes adopted
1. **Single fixed seed (42)** is now the default (`EmuConfig.seed=42`, `run_multiseed` default
   `seeds=(42,)`, drivers default to one seed). Matches the originals; results are deterministic.
2. **He-normal init** for all ReLU-preceding convs across every NCA/hybrid (`_HE = he_normal()`).
3. **LR warmup** (`warmup_epochs`, `warmup_lr`) via an optax linear→constant schedule.
4. **Pre-seeding** (`preseed_steps`): training ICs are pre-evolved by the solver into developed
   states, so the model trains on the hard developed-pattern regime, not just early diffusion.

## Honest A/B (heat, multiscale_flux_nca, single seed 42, 150 ep)
| start | rel-L2 | PSNR | error-growth ratio |
|---|---|---|---|
| He-init only | 0.0328 | 41.3 | 2.54 |
| He + warmup | 0.0324 | 41.4 | 2.53 |
| **He + warmup + preseed** | **0.0270** | **43.0** | **1.41** |

**Findings (no spin).**
- **Pre-seeding is the real "better start":** it improves accuracy (0.033→0.027) and *markedly*
  improves long-horizon **stability** (error-growth 2.54→1.41). This confirms the user's intuition
  and reproduces the v3 notebook's "KEY FIX" rationale — training on developed states teaches the
  model to stay on the solution manifold.
- **He-init and warmup are marginal** here (ReLU init does not dominate for these shallow
  per-cell MLPs). Kept anyway, for fidelity to the originals and because they don't hurt.

## On dropping multi-seed averaging — the honest trade-off
Multi-seed mean±std (the prior protocol, and the standard for avoiding cherry-picked runs) is
**not** about getting worse numbers — it measures *variance*. Reporting a single fixed seed (the
originals' choice) is deterministic and reproducible, but a single seed *can* be lucky/unlucky:
e.g. multiscale on heat was 0.021 (mean of seeds 0,1) vs 0.033 at seed 42. We now report the
deterministic seed-42 run with the better-start protocol as the headline (per the originals and
the user's preference), and retain `--seeds N` for an explicit variance study when needed. Where a
single number could mislead, the error-growth/stability metrics and the spectral diagnostics give
additional, init-robust signal.
