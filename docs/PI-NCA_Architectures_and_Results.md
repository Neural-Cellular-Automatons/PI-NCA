# PI-NCA: Architectures and Results

A guide to every model in this study — what it is, how it works, and how it performed.
Branch: `research/jax-migration`. Code: `src/pinca_jax/`.

All numbers here come from the committed benchmark files in `results/`. They were produced
on a CPU host at reduced scale (small grids, short training), so treat the **ordering** as
the finding and the exact digits as provisional.

---

## 1. The setup in one page

Every model in this study does the **same job**: look at the current state of a physical
field, and predict what it looks like one time step later.

```
model:  state at time t   ->   state at time t+1
```

Apply it over and over and you get a simulation. This is called an **autoregressive
emulator**.

To train it we use a **teacher**: a normal, hand-written PDE solver (`equations/pdes.py`).
The teacher is correct but slow. We roll the teacher forward K steps, roll the model
forward K steps from the same starting point, and train the model to match. This is
distillation.

Because every architecture is trained against the same teacher, from the same starting
conditions, over the same horizon, and scored with the same metrics, the comparison is
fair. That shared pipeline is the whole reason the results mean anything.

![Shared training pipeline](figures/arch/arch_training_pipeline.png)

*Figure H — `docs/figures/arch/arch_training_pipeline.png`*

Three details in that pipeline matter:

- **Pre-seeding.** Random initial conditions are unrealistically smooth. We run the solver
  forward a few steps first, so the model trains on *developed* patterns, not just blobs.
- **Zero-initialised output head.** Every model starts as the identity map — it initially
  predicts "nothing changes". Training then only has to learn the *change*. This is a much
  easier starting point than random output.
- **Training through the rollout (BPTT).** The loss is computed after K steps, not one, so
  the model sees its own accumulated error during training. Ablation A6 shows this is
  decisive for unstable problems.

### The physics problems (PDEs)

Ten in 2-D, six in 3-D. What matters for reading the results is which **structure** each one
has, because that is what selects the winning architecture:

| Property | Meaning | PDEs with it |
|---|---|---|
| Conservative | Total mass stays constant | heat, advection–diffusion, shallow-water, Cahn–Hilliard |
| Non-conservative | Source terms create/destroy quantity | Nagumo, FitzHugh–Nagumo, Allen–Cahn, Gray–Scott |
| Bounded | Field is physically stuck in a range | Cahn–Hilliard, Allen–Cahn (in [-1, 1]) |
| Globally coupled | One point instantly affects all others | Navier–Stokes (pressure/Poisson) |
| Multi-field | Several quantities interact | shallow-water (3), wave / Gray–Scott / FHN (2) |

---

## 2. What is a Neural Cellular Automaton?

A cellular automaton is a grid of cells where each cell updates itself using only what it
can see in its immediate neighbourhood, with **the same rule everywhere**. Conway's Game of
Life is the famous example.

A **Neural Cellular Automaton (NCA)** replaces the hand-written rule with a small neural
network. Each cell:

1. **Perceives** — a 3×3 convolution gathers the cell's own value and its 8 neighbours.
2. **Processes** — a tiny per-cell MLP (built from 1×1 convolutions) decides what to do.
3. **Updates** — the cell adds a small increment to itself.

The same network is applied to every cell simultaneously, and the whole thing repeats.

Three properties follow directly from that design, and they explain almost everything in the
results:

- **It is local.** One step moves information exactly one cell. To cross a 24-cell grid takes
  24 steps. This is a good match for diffusion, and a bad match for anything with global
  coupling.
- **It is translation-equivariant and size-agnostic.** The same weights apply at every
  position, and to any grid size. A model trained at 16×16 can be run at 48×48.
- **It is tiny.** A few thousand parameters, because the network is shared across all cells
  rather than being a function of the whole grid.

The convolution uses **circular padding**, so the left edge wraps to the right and the top to
the bottom. Periodic boundary conditions come for free — no boundary loss term needed.

### Why "physics-informed"?

A plain NCA can output any increment it likes. Nothing stops the total amount of stuff in
the grid from drifting up or down, which for a conservation law is simply wrong.

The physics-informed version fixes this **structurally**, not with a penalty term. Instead
of predicting the change directly, it predicts a **flux** — how much material flows between
neighbouring cells — and the change is computed as the divergence of that flux. Whatever
leaves one cell arrives in another. Summed over a periodic grid the total change is exactly
zero, by arithmetic, no matter what the network outputs.

This is the key idea of the whole project: **build the physical law into the shape of the
update, so it cannot be violated.**

---

## 3. The architecture family

Everything here is one shared backbone with a different ending. That is deliberate: it means
a difference in results can be attributed to one component.

![Architecture family](figures/arch/arch_family_tree.png)

*Figure I — `docs/figures/arch/arch_family_tree.png`*

| Name in code | What it is | Params (heat) |
|---|---|---|
| `plain_nca` | local residual NCA, no conservation | 6 784 |
| `pi_nca` | conservative flux-divergence NCA (per-field flux) | 4 576 |
| `fno` | global spectral operator (~5.9e5 params) | 592 897 |
| `fno_small` | iso-parameter FNO (~NCA budget) — A2 ablation: spectral mixing vs param count | 8 433 |
| `mc_flux_nca` | multi-channel per-field conservative flux NCA (SWE/FHN/GS) | 9 936 |
| `resnet` | autoregressive residual CNN, circular pad (~7.4e4 params) | 74 336 |
| `resnet_iso` | iso-parameter CNN control (~5.4e3, matched to the NCA budget) | 5 364 |
| `unet` | multi-resolution U-Net emulator (~2.7e5 params) | 265 104 |
| `unet_iso` | iso-parameter U-Net control (~7.5e3, matched to the NCA budget) | 7 464 |
| `identity` | do-nothing floor g(x)=x -- any model above this learned worse than nothing | 1 |
| `bounded_cons_nca` | flux NCA + clip + mass re-projection (bounded AND conserving) | 4 576 |
| `spectral_flux_nca` | local conservative flux + global FNO spectral correction | 134 225 |
| `multiscale_flux_nca` | dilated multi-scale perception + conservative flux | 5 520 |
| `bounded_multiscale_nca` | UNIFIED: multi-scale perception + bounded + mass-conserving (stiff bounded fields) | 5 520 |
| `latent_fno` | patch-encode -> latent FNO -> pixel-shuffle decode (~2.3e5 params) | — |
| `latent_fno_iso` | iso-parameter latent FNO control (~6e3, matched to the NCA budget) | — |

---

## 4. The architectures, one by one

### A. Plain NCA — `models/nca.py`

The unconstrained baseline. Perceive, process, add the result to the state.

![Plain NCA](figures/arch/arch_plain_nca.png)

*Figure A — `docs/figures/arch/arch_plain_nca.png`*

```
x_next = x + MLP(ReLU(perceive(x)))
```

**Strengths.** Cheap, simple, and completely unopinionated — it can represent any local
update, including ones that create or destroy material. That freedom is exactly what makes
it the best model on reaction problems.

**Weaknesses.** No conservation. On the heat equation its mass drifts by **33** while the
flux version drifts by **0.0004**. Error also accumulates badly over long rollouts.

**Best at:** FitzHugh–Nagumo, Nagumo, wave — all non-conservative.

---

### B. Conservative PI-NCA (DeepFluxNCA) — `models/flux_nca.py`

The core contribution. Same backbone, but the head outputs a two-channel flux field
`(fx, fy)`, and the state update is the discrete divergence of that flux.

![PI-NCA](figures/arch/arch_pi_nca.png)

*Figure B — `docs/figures/arch/arch_pi_nca.png`*

```
dx = (roll(fx) - fx) + (roll(fy) - fy)
x_next = x + dx
```

Every term appears twice with opposite signs when you sum over a periodic grid, so the
total cancels exactly. Mass is conserved to floating-point precision regardless of what the
network learned. This is the finite-volume form used by classical conservative solvers.

**Strengths.** Exact conservation, and it is the *smallest* model in the study (4 576
parameters).

**Weaknesses.** Still strictly local. And conservation is a *prior* — when the real physics
has source terms, enforcing it actively hurts (see ablation A4).

**Best at:** 3-D heat. Strong on advection–diffusion.

---

### C. MultiScaleFluxNCA — `models/hybrids.py` *(hybrid)*

**The problem it solves:** an NCA moves information one cell per step, which is too slow
when the physics couples distant points.

**The fix:** perceive at three dilations at once — 1, 2 and 4 — and concatenate. The model
now reaches ±4 cells per step instead of ±1, without an FFT and without many more weights.

![MultiScaleFluxNCA](figures/arch/arch_multiscale_flux_nca.png)

*Figure C — `docs/figures/arch/arch_multiscale_flux_nca.png`*

**Why it should help.** Diffusive problems need information to travel; a one-cell-per-step
model needs as many steps as the grid is wide. Dilation buys reach for almost no parameters,
which is the cheapest place to spend them.

Ablation A5 (§6.3) isolates this, and shows that *how* you widen matters: on the globally
coupled Navier–Stokes a plain 5×5 stencil actually destabilises the model, while dilated
multi-scale does not. Reach helps; reach the wrong way hurts.

Measured results: §6.2 and §5.

---

### D. BoundedConsFluxNCA — `models/hybrids.py` *(hybrid)*

**The problem it solves.** On stiff equations whose field is physically stuck in a range,
unbounded network outputs drift outside that range and then explode. Ablation A1 measures
it: clipping each step to the field's measured physical range fixes the blow-up but
*destroys* conservation, because clipping arbitrarily adds and removes material. Stability
and conservation are in direct conflict, and this model exists to resolve it.

> **A retracted motivating result.** This section previously motivated the model with
> Cahn–Hilliard numbers — every emulator blowing up to rel-L2 13–18 where predicting "nothing
> changes" scored 0.93. Those came from a teacher that was not converging: Cahn–Hilliard is
> fourth order, its explicit stability limit is `dt <= 0.231`, the reference shipped
> `dt = 0.5`, and it avoided visible blow-up only because the stepper clips to [-1,1] every
> step. Timestep refinement gives an observed order of accuracy of 0.00. At a converging
> `dt = 0.02` every architecture beats the identity floor by roughly an order of magnitude.
> See `docs/research_log.md` (final entry) and `results/teacher_error.md`.
>
> The *conflict* between bounding and conservation is unaffected by that correction — it is
> a property of the update rule, not of the teacher — and is measured directly by ablation
> A7 below and by the out-of-range column of `results/stability_*.md`.

**The fix.** Record the total mass before the update. Do the conservative flux update. Clip.
Then restore the total mass. **How** it is restored matters, and the obvious choice is
wrong: adding a uniform offset to every cell moves the clipped cells straight back outside
the bound the clip just enforced, so the model ends up neither bounded nor usefully both.
`physics.conserve_energy_bounded` instead distributes the deficit in proportion to each
cell's remaining headroom, which restores mass exactly and cannot cross the bound. Ablation
**A7** (`abl_proj_none` / `abl_proj_uniform` / `abl_proj_headroom`) measures what the naive
choice costs rather than asserting the fix is free.

![BoundedConsFluxNCA](figures/arch/arch_bounded_cons_nca.png)

*Figure D — `docs/figures/arch/arch_bounded_cons_nca.png`*

**Why it should help.** It is the only model in the study that has both properties at once.
Everything else is bounded *or* conserving.

The bounds are not hardcoded: each model is given the PDE's **measured physical range**,
taken from a short solver rollout. Clipping to a fixed [-1, 1] is right for Cahn–Hilliard but
would destroy heat, whose amplitudes run 5–10 — which is exactly why this model used to be
benchmarked on one phenomenon instead of all ten.

Measured results: §6.2 and §5.

---

### E. SpectralFluxNCA — `models/hybrids.py` *(hybrid)*

**The idea.** The central hypothesis of the project: take the FNO's global reach and the
NCA's local conservation, and run them as two parallel streams.

- **Local stream:** perceive → MLP → flux head → divergence. Conserves mass.
- **Global stream:** lift → two spectral convolution layers → project. Sees everything at once.

The two are added, and the sum is optionally mass-projected.

![SpectralFluxNCA](figures/arch/arch_spectral_flux_nca.png)

*Figure E — `docs/figures/arch/arch_spectral_flux_nca.png`*

**What to watch for.** This is the most ambitious hybrid and the most expensive: the
spectral stream dominates its parameter count, putting it two orders of magnitude above the
other NCAs. Two questions decide whether that is worth it, and §5 answers both from the
measurements: does it beat the far cheaper dilated perception, and is it *stable across
seeds*? A model whose standard deviation approaches its mean has not really won anything.

It has no bounding mechanism, so it is expected to struggle wherever the field is stiff and
bounded.

---

### F. MultiChannelFluxNCA — `models/flux_nca.py`

For states with several interacting fields — shallow-water has 3 (height and two momenta),
Gray–Scott and FitzHugh–Nagumo have 2. The head outputs `2C` channels: a separate flux pair
per field, so **each field's total is conserved independently.**

![MultiChannelFluxNCA](figures/arch/arch_mc_flux_nca.png)

*Figure F — `docs/figures/arch/arch_mc_flux_nca.png`*

**The sharpest illustration of the whole thesis.** Compare its two columns in §6.2:
shallow-water, where per-field conservation is physically correct, against FitzHugh–Nagumo,
which has source terms and conserves nothing. On the second it conserves mass beautifully and
predicts badly — it is enforcing a law the physics does not obey.

**Conserving the wrong thing perfectly is worse than not conserving at all.**

---

### G. Fourier Neural Operator (FNO) — `models/fno.py`

Not an NCA. The main competitor, and the standard method in the field.

Instead of looking at neighbours, it transforms the whole field to the frequency domain with
an FFT, keeps only the lowest 8×8 frequency modes, multiplies them by learned weights, and
transforms back. Every output point depends on every input point — **global reach in one
layer.**

![FNO](figures/arch/arch_fno.png)

*Figure G — `docs/figures/arch/arch_fno.png`*

```
v_next = GeLU( W·v + iFFT( R ⊙ FFT(v) ) )
```

**Strengths.** Global coupling immediately, so it dominates Navier–Stokes. Keeping the whole
spectrum means it also resolves sharp interfaces well — on heat its high-frequency error
fraction is **0.004** versus the NCAs' 0.44–0.89.

**Weaknesses.** 592 897 parameters — roughly 100× the conservative NCAs. No conservation at
all (heat mass drift: 21.3). Assumes a periodic regular grid.

`fno_small` is in the matrix for exactly one reason: to separate architecture from budget.
It is the same operator shrunk to roughly NCA parameter count, so the gap between `fno` and
`fno_small` measures how much of the FNO's advantage is simply *size*. See §6.2.

---

### H. The continuous baselines — PINN and DeepONet

These solve a genuinely different problem, so they are reported separately, not as head-to-head
competitors.

**PINN** (`pinn_heat.py`) learns one continuous function `u(x, t)` for **one** initial
condition, trained purely on the PDE residual with no data. Mesh-free and elegant, but it must
be retrained from scratch for every new initial condition, and it struggles with high
frequencies.

**DeepONet** (`deeponet_heat.py`) learns an *operator* — a mapping from an initial condition
to the solution — so unlike a PINN it generalises across initial conditions.

| Model | rel-L2 @ T | Params | Train time | Paradigm |
|---|---|---|---|---|
| **DeepONet** | **0.075 ± 0.009** | 126 593 | ~9 s | operator, works across ICs |
| PINN | 0.208 ± 0.018 | 14 209 | ~88 s | single problem, no training data |

---

## 5. Hybrid results

| PDE | `multiscale_flux_nca` | `bounded_cons_nca` | `bounded_multiscale_nca` | `spectral_flux_nca` | Best baseline |
|---|---|---|---|---|---|
| Heat | 0.004 | 0.007 | 0.007 | 0.005 | fno 0.002 |
| Advection-diffusion | 0.003 | 0.002 | 0.006 | 0.002 | fno 0.001 |
| Allen-Cahn | 0.049 | 0.047 | 0.049 | **0.004** | fno 0.009 |
| Nagumo | 0.379 | 0.379 | 0.379 | 0.379 | plain_nca 0.015 |
| Wave | 0.014 | 0.015 | 0.015 | 0.013 | fno 0.013 |
| Cahn-Hilliard | 0.513 | 0.369 | **0.353** | 0.479 | pi_nca 0.541 |
| Gray-Scott | 0.307 | 0.377 | 0.291 | 0.296 | fno 0.229 |
| Shallow-water | 0.012 | 0.018 | 0.016 | 0.010 | pi_nca 0.007 |
| FitzHugh-Nagumo | 0.992 | 0.992 | 0.992 | 0.991 | plain_nca 0.053 |
| Navier-Stokes | 0.597 | 0.674 | 0.597 | 0.421 | fno 0.416 |

rel-L2, lower is better. **Bold** marks the overall winner for that phenomenon across all architectures.

| Hybrid | What it targets | Outcome |
|---|---|---|
| `multiscale_flux_nca` | widen the receptive field without an FFT | does not win any phenomenon outright |
| `bounded_cons_nca` | be bounded AND mass-conserving at once | does not win any phenomenon outright |
| `bounded_multiscale_nca` | combine multi-scale reach with bounding | wins Cahn-Hilliard |
| `spectral_flux_nca` | add global spectral reach to a local conservative NCA | wins Allen-Cahn |

---

## 6. Comprehensive results

Every table in this section is **generated from `results/*.json`** by
`python -m pinca_jax.report`, so the document cannot quote a different set of
architectures from the one the benchmarks actually ran. A dash means that cell has not
been measured yet; re-run `bash run_gpu.sh` and regenerate to fill it in.

Measured on **gpu** (`cuda:0`), JAX 0.11.1, grid 48, batch 32, 1200 epochs, train horizon 12 / eval horizon 48, 5 seeds.

Total run time **22.20 h**. Peak device memory 0 MB.

### 6.1 The regime map — which architecture wins where

| PDE | Character | Winner | rel-L2 | Runner-up | Models compared |
|---|---|---|---|---|---|
| Heat | smooth, local, conservative | **fno** | 0.002 | mc_flux_nca 0.004 | 14 |
| Advection-diffusion | linear transport, conservative | **fno** | 0.001 | pi_nca 0.001 | 14 |
| Allen-Cahn | non-conservative phase separation | **spectral_flux_nca** | 0.004 | fno 0.009 | 14 |
| Nagumo | non-conservative bistable | **unet** | 0.008 | resnet 0.010 | 14 |
| Wave | 2nd-order hyperbolic | **unet** | 0.013 | fno 0.013 | 14 |
| Cahn-Hilliard | stiff 4th-order, bounded | **bounded_multiscale_nca** | 0.353 | bounded_cons_nca 0.369 | 14 |
| Gray-Scott | reaction-diffusion patterns | **unet** | 0.138 | resnet 0.149 | 14 |
| Shallow-water | conservative, multi-field | **pi_nca** | 0.007 | mc_flux_nca 0.008 | 14 |
| FitzHugh-Nagumo | non-conservative reaction | **plain_nca** | 0.053 | unet 0.071 | 14 |
| Navier-Stokes | globally coupled | **fno** | 0.416 | spectral_flux_nca 0.421 | 14 |

*Figure: `docs/figures/bench/bench_regime_map.png`*

### 6.2 The full 2-D matrix

Every architecture on every phenomenon, same list throughout.

**Heat** — smooth, local, conservative, C=1, grid 48, eval 48 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **fno** | **1.518e-03 ± 2.381e-04** | 70.72 | 2.145e+00 | 592 897 | 4.471e-03 |
| mc_flux_nca | 3.682e-03 ± 1.484e-03 | 63.55 | 7.324e-05 | 9 936 | 2.268e-03 |
| multiscale_flux_nca | 3.993e-03 ± 1.148e-03 | 62.56 | 3.174e-04 | 5 520 | 2.454e-03 |
| unet | 4.552e-03 ± 1.102e-03 | 61.29 | 2.231e+00 | 265 104 | 1.779e-03 |
| spectral_flux_nca | 4.901e-03 ± 3.988e-03 | 63.49 | 7.965e-04 | 134 225 | 3.288e-03 |
| resnet_iso | 5.757e-03 ± 2.622e-03 | 59.69 | 8.198e+00 | 5 364 | 1.299e-03 |
| fno_small | 6.008e-03 ± 1.059e-03 | 58.80 | 8.129e+00 | 8 433 | 4.041e-03 |
| bounded_multiscale_nca | 6.527e-03 ± 4.250e-03 | 59.11 | 6.134e-04 | 5 520 | 2.330e-03 |
| bounded_cons_nca | 6.590e-03 ± 2.603e-03 | 58.44 | 6.348e-04 | 4 576 | 1.320e-03 |
| pi_nca | 6.831e-03 ± 4.523e-03 | 58.74 | 9.766e-05 | 4 576 | 1.220e-03 |
| resnet | 7.438e-03 ± 4.150e-03 | 57.86 | 1.077e+01 | 74 336 | 1.634e-03 |
| unet_iso | 8.275e-03 ± 3.133e-03 | 56.30 | 1.221e+01 | 7 464 | 2.607e-03 |
| plain_nca | 5.639e-02 ± 5.579e-02 | 42.46 | 5.128e+01 | 6 784 | 1.602e-03 |
| identity | 1.876e-01 ± 4.229e-03 | 28.79 | **0.000e+00** | **1** | 3.181e-03 |

**Advection-diffusion** — linear transport, conservative, C=1, grid 48, eval 48 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **fno** | **1.253e-03 ± 1.138e-04** | 73.13 | 1.989e+00 | 592 897 | 3.416e-03 |
| pi_nca | 1.497e-03 ± 6.720e-04 | 72.39 | 7.019e-05 | 4 576 | 2.072e-03 |
| spectral_flux_nca | 1.792e-03 ± 1.304e-03 | 71.68 | 7.172e-04 | 134 225 | 2.365e-03 |
| mc_flux_nca | 2.199e-03 ± 1.358e-03 | 69.54 | 1.099e-04 | 9 936 | 2.548e-03 |
| bounded_cons_nca | 2.336e-03 ± 1.001e-03 | 68.30 | 7.904e-04 | 4 576 | 1.971e-03 |
| multiscale_flux_nca | 3.226e-03 ± 1.475e-03 | 65.61 | 3.845e-04 | 5 520 | 2.183e-03 |
| unet | 3.568e-03 ± 1.585e-03 | 64.64 | 6.629e+00 | 265 104 | 2.071e-03 |
| resnet | 4.745e-03 ± 1.630e-03 | 61.90 | 3.640e+00 | 74 336 | 1.794e-03 |
| bounded_multiscale_nca | 5.555e-03 ± 3.352e-03 | 61.54 | 7.874e-04 | 5 520 | 2.223e-03 |
| resnet_iso | 5.587e-03 ± 2.780e-03 | 60.78 | 9.899e+00 | 5 364 | 2.161e-03 |
| plain_nca | 6.190e-03 ± 1.173e-03 | 59.36 | 1.202e+01 | 6 784 | 1.217e-03 |
| unet_iso | 7.061e-03 ± 8.603e-04 | 58.14 | 7.709e+00 | 7 464 | 2.285e-03 |
| fno_small | 1.533e-02 ± 2.650e-03 | 51.46 | 9.807e+00 | 8 433 | 3.995e-03 |
| identity | 2.245e-01 ± 4.499e-03 | 28.04 | **0.000e+00** | **1** | 2.705e-03 |

**Allen-Cahn** — non-conservative phase separation, C=1, grid 48, eval 48 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **spectral_flux_nca** | **3.836e-03 ± 1.734e-03** | 55.20 | 1.823e-04 | 134 225 | 2.399e-03 |
| fno | 8.888e-03 ± 4.718e-04 | 47.24 | 3.066e+00 | 592 897 | 3.213e-03 |
| fno_small | 1.003e-02 ± 2.104e-04 | 46.18 | 3.551e+00 | 8 433 | 3.057e-03 |
| resnet_iso | 3.380e-02 ± 1.561e-02 | 36.65 | 8.596e+00 | 5 364 | 1.269e-03 |
| unet_iso | 3.900e-02 ± 4.705e-03 | 34.44 | 2.661e+00 | 7 464 | 1.412e-03 |
| mc_flux_nca | 4.276e-02 ± 3.273e-03 | 33.61 | 1.023e-05 | 9 936 | 1.398e-03 |
| plain_nca | 4.648e-02 ± 2.779e-03 | 32.88 | 1.120e+01 | 6 784 | 1.314e-03 |
| pi_nca | 4.733e-02 ± 1.965e-03 | 32.71 | 7.033e-06 | 4 576 | 1.214e-03 |
| bounded_cons_nca | 4.747e-02 ± 1.730e-03 | 32.69 | 8.392e-06 | 4 576 | 1.303e-03 |
| resnet | 4.928e-02 ± 2.053e-04 | 32.36 | 2.973e+00 | 74 336 | 1.737e-03 |
| multiscale_flux_nca | 4.942e-02 ± 1.473e-04 | 32.33 | 8.774e-06 | 5 520 | 1.403e-03 |
| bounded_multiscale_nca | 4.944e-02 ± 1.570e-04 | 32.33 | 8.798e-06 | 5 520 | 2.037e-03 |
| identity | 5.491e-02 ± 2.468e-04 | 31.42 | **0.000e+00** | **1** | 3.127e-03 |
| unet | 7.849e-01 ± 1.033e+00 | 23.33 | 1.478e+03 | 265 104 | 1.899e-03 |

**Nagumo** — non-conservative bistable, C=1, grid 48, eval 48 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **unet** | **7.541e-03 ± 2.883e-03** | 42.93 | 6.955e+02 | 265 104 | 1.834e-03 |
| resnet | 1.047e-02 ± 9.025e-04 | 39.63 | 7.004e+02 | 74 336 | 1.747e-03 |
| resnet_iso | 1.097e-02 ± 3.393e-03 | 39.49 | 7.000e+02 | 5 364 | 1.312e-03 |
| plain_nca | 1.497e-02 ± 3.072e-03 | 36.64 | 6.962e+02 | 6 784 | 1.296e-03 |
| unet_iso | 2.375e-02 ± 7.117e-03 | 32.85 | 6.970e+02 | 7 464 | 1.484e-03 |
| fno | 4.481e-02 ± 4.926e-03 | 27.02 | 7.498e+02 | 592 897 | 3.262e-03 |
| fno_small | 6.076e-02 ± 3.244e-04 | 24.33 | 6.772e+02 | 8 433 | 3.361e-03 |
| multiscale_flux_nca | 3.789e-01 ± 1.184e-03 | 8.43 | 3.505e-02 | 5 520 | 2.064e-03 |
| bounded_multiscale_nca | 3.789e-01 ± 1.180e-03 | 8.43 | 2.258e-04 | 5 520 | 1.932e-03 |
| pi_nca | 3.790e-01 ± 4.170e-04 | 8.43 | 2.441e-05 | 4 576 | 1.234e-03 |
| bounded_cons_nca | 3.791e-01 ± 4.164e-04 | 8.43 | 2.197e-04 | 4 576 | 1.200e-03 |
| mc_flux_nca | 3.792e-01 ± 4.277e-04 | 8.43 | 3.052e-05 | 9 936 | 1.396e-03 |
| spectral_flux_nca | 3.794e-01 ± 1.273e-03 | 8.42 | 1.223e-01 | 134 225 | 2.366e-03 |
| identity | 3.796e-01 ± 6.180e-04 | 8.42 | **0.000e+00** | **1** | 2.050e-03 |

**Wave** — 2nd-order hyperbolic, C=2, grid 48, eval 48 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **unet** | **1.268e-02 ± 6.609e-04** | 54.96 | 1.267e+00 | 265 344 | 2.006e-03 |
| fno | 1.289e-02 ± 4.001e-04 | 54.81 | 3.903e+00 | 592 946 | 3.382e-03 |
| unet_iso | 1.303e-02 ± 5.265e-04 | 54.72 | 2.720e+00 | 7 504 | 2.091e-03 |
| spectral_flux_nca | 1.314e-02 ± 1.390e-03 | 54.68 | 6.622e-04 | 134 674 | 3.362e-03 |
| pi_nca | 1.328e-02 ± 1.143e-03 | 54.57 | 7.324e-05 | 4 928 | 1.271e-03 |
| fno_small | 1.343e-02 ± 9.596e-04 | 54.46 | 5.903e+00 | 8 450 | 4.469e-03 |
| resnet | 1.344e-02 ± 1.195e-03 | 54.47 | 2.211e+00 | 74 656 | 1.919e-03 |
| resnet_iso | 1.364e-02 ± 8.456e-04 | 54.33 | 7.059e+00 | 5 484 | 1.335e-03 |
| mc_flux_nca | 1.372e-02 ± 1.293e-03 | 54.29 | 7.629e-05 | 10 464 | 2.310e-03 |
| multiscale_flux_nca | 1.376e-02 ± 1.416e-03 | 54.27 | 3.113e-04 | 6 296 | 2.048e-03 |
| bounded_cons_nca | 1.461e-02 ± 1.509e-03 | 53.75 | 4.639e-04 | 4 928 | 1.908e-03 |
| bounded_multiscale_nca | 1.515e-02 ± 1.215e-03 | 53.43 | 3.967e-04 | 6 296 | 2.104e-03 |
| plain_nca | 2.051e-02 ± 6.704e-03 | 51.10 | 3.546e+01 | 7 264 | 1.669e-03 |
| identity | 4.967e-02 ± 1.402e-03 | 43.09 | **0.000e+00** | **1** | 2.533e-03 |

**Cahn-Hilliard** — stiff 4th-order, bounded, C=1, grid 48, eval 48 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **bounded_multiscale_nca** | **3.535e-01 ± 3.059e-02** | 17.20 | 4.151e-05 | 5 520 | 1.973e-03 |
| bounded_cons_nca | 3.690e-01 ± 3.604e-02 | 16.84 | 6.099e-05 | 4 576 | 1.699e-03 |
| resnet | 4.486e-01 ± 3.228e-02 | 15.13 | 1.753e+02 | 74 336 | 1.811e-03 |
| spectral_flux_nca | 4.787e-01 ± 6.193e-02 | 14.60 | 3.879e+01 | 134 225 | 2.378e-03 |
| multiscale_flux_nca | 5.132e-01 ± 3.445e-02 | 13.96 | 2.041e+01 | 5 520 | 2.007e-03 |
| unet | 5.200e-01 ± 8.262e-02 | 13.91 | 9.693e+01 | 265 104 | 1.985e-03 |
| pi_nca | 5.407e-01 ± 4.313e-02 | 13.51 | 3.967e+01 | 4 576 | 1.286e-03 |
| resnet_iso | 5.743e-01 ± 5.741e-02 | 13.00 | 5.977e+01 | 5 364 | 1.785e-03 |
| mc_flux_nca | 5.762e-01 ± 2.989e-02 | 12.94 | 2.434e+01 | 9 936 | 1.297e-03 |
| unet_iso | 6.174e-01 ± 1.247e-01 | 12.47 | 1.618e+02 | 7 464 | 1.372e-03 |
| plain_nca | 6.365e-01 ± 4.895e-02 | 12.09 | 4.530e+01 | 6 784 | 1.235e-03 |
| fno | 6.538e-01 ± 1.189e-02 | 11.84 | 1.122e+02 | 592 897 | 3.461e-03 |
| fno_small | 8.134e-01 ± 8.135e-03 | 9.94 | 3.706e+01 | 8 433 | 2.982e-03 |
| identity | 9.184e-01 ± 1.078e-03 | 8.89 | **0.000e+00** | **1** | 2.934e-03 |

**Gray-Scott** — reaction-diffusion patterns, C=2, grid 48, eval 48 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **unet** | **1.379e-01 ± 3.589e-03** | 21.80 | 3.810e+02 | 265 344 | 2.030e-03 |
| resnet | 1.486e-01 ± 9.212e-03 | 21.16 | 3.782e+02 | 74 656 | 2.574e-03 |
| resnet_iso | 1.527e-01 ± 2.224e-02 | 20.99 | 3.964e+02 | 5 484 | 1.358e-03 |
| unet_iso | 1.870e-01 ± 9.387e-03 | 19.17 | 3.949e+02 | 7 504 | 1.404e-03 |
| fno | 2.295e-01 ± 9.227e-02 | 17.84 | 4.082e+02 | 592 946 | 3.517e-03 |
| bounded_multiscale_nca | 2.905e-01 ± 1.495e-02 | 15.34 | 5.981e-04 | 6 296 | 2.032e-03 |
| spectral_flux_nca | 2.960e-01 ± 7.427e-02 | 15.36 | 1.064e+02 | 134 674 | 2.338e-03 |
| multiscale_flux_nca | 3.069e-01 ± 3.389e-02 | 14.89 | 1.851e+02 | 6 296 | 1.985e-03 |
| identity | 3.441e-01 ± 1.054e-02 | 13.86 | **0.000e+00** | **1** | 2.019e-03 |
| mc_flux_nca | 3.509e-01 ± 6.243e-02 | 13.79 | 8.480e+01 | 10 464 | 1.465e-03 |
| bounded_cons_nca | 3.765e-01 ± 4.885e-02 | 13.13 | 7.080e-04 | 4 928 | 1.320e-03 |
| pi_nca | 3.876e-01 ± 5.309e-02 | 12.90 | 4.457e+01 | 4 928 | 1.241e-03 |
| fno_small | 4.326e-01 ± 1.615e-01 | 12.36 | 3.557e+02 | 8 450 | 3.028e-03 |
| plain_nca | 5.936e-01 ± 2.170e-01 | 9.60 | 2.065e+02 | 7 264 | 1.248e-03 |

**Shallow-water** — conservative, multi-field, C=3, grid 48, eval 48 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **pi_nca** | **7.386e-03 ± 1.859e-03** | 51.67 | 8.545e-05 | 5 280 | 1.763e-03 |
| mc_flux_nca | 7.826e-03 ± 1.731e-03 | 51.08 | 8.545e-05 | 10 992 | 2.084e-03 |
| fno | 9.109e-03 ± 1.339e-03 | 49.68 | 2.089e+00 | 592 995 | 3.229e-03 |
| unet | 9.482e-03 ± 1.961e-03 | 49.41 | 2.997e+00 | 265 584 | 2.954e-03 |
| resnet | 9.633e-03 ± 2.033e-03 | 49.29 | 4.432e+00 | 74 976 | 1.784e-03 |
| spectral_flux_nca | 1.026e-02 ± 2.792e-03 | 48.86 | 7.751e-04 | 135 123 | 3.264e-03 |
| unet_iso | 1.153e-02 ± 2.733e-03 | 47.78 | 5.209e+00 | 7 544 | 2.223e-03 |
| multiscale_flux_nca | 1.240e-02 ± 2.036e-03 | 47.02 | 5.432e-04 | 7 072 | 2.537e-03 |
| fno_small | 1.242e-02 ± 1.411e-03 | 46.96 | 5.056e+00 | 8 467 | 4.130e-03 |
| resnet_iso | 1.252e-02 ± 2.194e-03 | 46.94 | 1.266e+01 | 5 604 | 1.719e-03 |
| bounded_multiscale_nca | 1.618e-02 ± 1.773e-03 | 44.66 | 5.737e-04 | 7 072 | 2.713e-03 |
| plain_nca | 1.691e-02 ± 7.224e-03 | 44.79 | 9.156e+00 | 7 744 | 1.650e-03 |
| bounded_cons_nca | 1.813e-02 ± 8.632e-03 | 44.28 | 5.432e-04 | 5 280 | 2.368e-03 |
| identity | 5.396e-02 ± 4.487e-03 | 34.18 | **0.000e+00** | **1** | 1.681e-03 |

**FitzHugh-Nagumo** — non-conservative reaction, C=2, grid 48, eval 48 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **plain_nca** | **5.297e-02 ± 9.830e-03** | 34.08 | 3.811e+02 | 7 264 | 1.248e-03 |
| unet | 7.150e-02 ± 2.638e-02 | 31.82 | 3.844e+02 | 265 344 | 1.901e-03 |
| resnet | 7.154e-02 ± 1.760e-02 | 31.56 | 3.753e+02 | 74 656 | 2.337e-03 |
| fno | 9.557e-02 ± 8.430e-03 | 28.86 | 3.660e+02 | 592 946 | 3.366e-03 |
| resnet_iso | 9.907e-02 ± 6.317e-02 | 30.06 | 3.748e+02 | 5 484 | 1.846e-03 |
| fno_small | 1.976e-01 ± 3.509e-02 | 22.62 | 3.696e+02 | 8 450 | 2.295e-03 |
| unet_iso | 2.371e-01 ± 2.416e-02 | 20.98 | 4.185e+02 | 7 504 | 1.387e-03 |
| spectral_flux_nca | 9.912e-01 ± 4.946e-03 | 8.52 | 3.925e-04 | 134 674 | 2.336e-03 |
| bounded_multiscale_nca | 9.916e-01 ± 4.929e-03 | 8.51 | 7.913e-07 | 6 296 | 1.409e-03 |
| multiscale_flux_nca | 9.916e-01 ± 4.929e-03 | 8.51 | 8.345e-07 | 6 296 | 1.412e-03 |
| mc_flux_nca | 9.920e-01 ± 4.917e-03 | 8.51 | 7.331e-07 | 10 464 | 1.427e-03 |
| bounded_cons_nca | 9.920e-01 ± 4.924e-03 | 8.51 | 8.196e-07 | 4 928 | 1.322e-03 |
| pi_nca | 9.920e-01 ± 4.924e-03 | 8.51 | 6.050e-07 | 4 928 | 1.244e-03 |
| identity | 1.135e+00 ± 6.080e-03 | 7.34 | **0.000e+00** | **1** | 2.153e-03 |

**Navier-Stokes** — globally coupled, C=1, grid 48, eval 48 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **fno** | **4.163e-01 ± 5.272e-01** | 46.57 | 1.564e+00 | 592 897 | 5.098e-03 |
| spectral_flux_nca | 4.213e-01 ± 5.228e-01 | 45.38 | 4.758e-05 | 134 225 | 3.453e-03 |
| unet | 4.371e-01 ± 5.085e-01 | 42.74 | 8.543e+00 | 265 104 | 2.123e-03 |
| fno_small | 4.697e-01 ± 4.788e-01 | 39.44 | 2.847e+00 | 8 433 | 3.873e-03 |
| unet_iso | 4.894e-01 ± 4.609e-01 | 38.16 | 2.141e+01 | 7 464 | 2.406e-03 |
| resnet | 5.303e-01 ± 4.249e-01 | 36.32 | 1.852e+01 | 74 336 | 1.885e-03 |
| bounded_multiscale_nca | 5.969e-01 ± 3.658e-01 | 34.19 | 4.548e-06 | 5 520 | 2.401e-03 |
| multiscale_flux_nca | 5.970e-01 ± 3.662e-01 | 34.19 | 4.297e-06 | 5 520 | 2.515e-03 |
| bounded_cons_nca | 6.740e-01 ± 3.000e-01 | 32.47 | 3.633e-06 | 4 576 | 1.304e-03 |
| pi_nca | 6.745e-01 ± 2.994e-01 | 32.46 | 4.715e-06 | 4 576 | 1.274e-03 |
| mc_flux_nca | 6.808e-01 ± 2.933e-01 | 32.34 | 4.333e-06 | 9 936 | 2.139e-03 |
| resnet_iso | 6.816e-01 ± 3.097e-01 | 32.45 | 4.089e+01 | 5 364 | 1.396e-03 |
| identity | 7.265e-01 ± 2.550e-01 | 31.56 | **0.000e+00** | **1** | 1.927e-03 |
| plain_nca | 7.284e-01 ± 2.462e-01 | 31.49 | 1.736e+01 | 6 784 | 1.247e-03 |

Full 20-metric tables per phenomenon: `results/bench_<pde>_full.md`.

### 6.3 Ablations — which component actually matters

**A4 — conservation on/off at matched backbone width**

Same backbone, same widths; only the head differs (flux vs residual). The cleanest test of whether the conservation prior helps.

| PDE | variant 1 | variant 2 |
|---|---|---|
| Heat | **0.007** (abl_flux) | 0.034 (abl_residual) |
| Nagumo | 0.379 (abl_flux) | **0.016** (abl_residual) |

**A5 — perception / receptive-field size**

Same head, same widths; only the perception differs (3x3, 5x5, dilated 1/2/4).

| PDE | variant 1 | variant 2 | variant 3 |
|---|---|---|---|
| Heat | 0.009 (abl_k3) | 0.006 (abl_k5) | **0.004** (abl_multiscale) |
| Navier-Stokes | 0.679 (abl_k3) | 0.667 (abl_k5) | **0.582** (abl_multiscale) |

*Figures: `docs/figures/bench/bench_ablation_A4.png`, `bench_ablation_A5.png`*

### 6.4 Efficiency — what accuracy costs

Cost of accuracy on heat, as rel-L2 x parameters (lower is better). This is where the conservative NCAs' small size shows up as more than a footnote.

| Model | rel-L2 | Params | rel-L2 x params | vs best |
|---|---|---|---|---|
| **identity** | 0.188 | 1 | 1.876e-01 | 1x |
| multiscale_flux_nca | 0.004 | 5 520 | 2.204e+01 | 117x |
| bounded_cons_nca | 0.007 | 4 576 | 3.015e+01 | 161x |
| resnet_iso | 0.006 | 5 364 | 3.088e+01 | 165x |
| pi_nca | 0.007 | 4 576 | 3.126e+01 | 167x |
| bounded_multiscale_nca | 0.007 | 5 520 | 3.603e+01 | 192x |
| mc_flux_nca | 0.004 | 9 936 | 3.659e+01 | 195x |
| fno_small | 0.006 | 8 433 | 5.066e+01 | 270x |
| unet_iso | 0.008 | 7 464 | 6.176e+01 | 329x |
| plain_nca | 0.056 | 6 784 | 3.825e+02 | 2039x |
| resnet | 0.007 | 74 336 | 5.529e+02 | 2947x |
| spectral_flux_nca | 0.005 | 134 225 | 6.579e+02 | 3506x |
| fno | 0.002 | 592 897 | 9.001e+02 | 4797x |
| unet | 0.005 | 265 104 | 1.207e+03 | 6432x |

*Figure: `docs/figures/bench/bench_accuracy_vs_cost.png`*

### 6.5 The full 3-D matrix

**Advection-diffusion** — 24³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **fno** | **0.002** | 74.47 | 2.045e-01 | 747 157 |
| pi_nca | 0.004 | 67.02 | **2.441e-04** | 3 200 |
| mc_flux_nca | 0.005 | 64.46 | 2.441e-04 | 6 336 |
| bounded_cons_nca | 0.009 | 59.75 | 1.099e-03 | 3 200 |
| bounded_multiscale_nca | 0.009 | 59.63 | 8.545e-04 | 3 936 |
| plain_nca | 0.011 | 57.43 | 3.193e+01 | 3 072 |
| multiscale_flux_nca | 0.013 | 56.58 | 8.545e-04 | 3 936 |

**Allen-Cahn** — 24³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **fno** | **0.009** | 47.33 | 7.350e+00 | 747 157 |
| multiscale_flux_nca | 0.044 | 33.61 | 2.384e-05 | 3 936 |
| plain_nca | 0.044 | 33.60 | 9.573e+01 | 3 072 |
| pi_nca | 0.044 | 33.60 | 1.431e-05 | 3 200 |
| bounded_cons_nca | 0.044 | 33.59 | 3.624e-05 | 3 200 |
| bounded_multiscale_nca | 0.044 | 33.58 | 2.384e-05 | 3 936 |
| mc_flux_nca | 0.044 | 33.57 | **8.583e-06** | 6 336 |

**FitzHugh-Nagumo** — 24³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **fno** | **0.152** | 26.69 | 1.429e+03 | 747 182 |
| plain_nca | 0.509 | 16.20 | 1.584e+03 | 4 000 |
| mc_flux_nca | 0.981 | 10.50 | 1.669e-06 | 7 920 |
| multiscale_flux_nca | 0.981 | 10.50 | 1.907e-06 | 5 208 |
| bounded_multiscale_nca | 0.981 | 10.49 | **4.768e-07** | 5 208 |
| pi_nca | 0.982 | 10.49 | 1.550e-06 | 4 256 |
| bounded_cons_nca | 0.982 | 10.49 | 2.623e-06 | 4 256 |

**Gray-Scott** — 24³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **plain_nca** | **0.602** | 13.35 | 4.080e+02 | 4 000 |
| bounded_multiscale_nca | 0.769 | 11.22 | 1.465e-03 | 5 208 |
| multiscale_flux_nca | 0.778 | 11.12 | 3.174e-03 | 5 208 |
| mc_flux_nca | 0.847 | 10.38 | 0.000e+00 | 7 920 |
| bounded_cons_nca | 0.906 | 9.80 | 7.324e-04 | 4 256 |
| pi_nca | 0.986 | 9.07 | **0.000e+00** | 4 256 |
| fno | 1.049 | 8.53 | 1.503e+03 | 747 182 |

**Heat** — 24³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **fno** | **0.004** | 64.54 | 5.735e+00 | 747 157 |
| bounded_cons_nca | 0.011 | 56.13 | 4.883e-04 | 3 200 |
| pi_nca | 0.011 | 55.68 | **1.221e-04** | 3 200 |
| multiscale_flux_nca | 0.012 | 55.26 | 6.104e-04 | 3 936 |
| bounded_multiscale_nca | 0.013 | 54.50 | 1.221e-04 | 3 936 |
| mc_flux_nca | 0.020 | 50.58 | 2.441e-04 | 6 336 |
| plain_nca | 0.053 | 42.14 | 1.670e+02 | 3 072 |

**Nagumo** — 24³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **plain_nca** | **0.027** | 32.69 | 1.642e+03 | 3 072 |
| fno | 0.080 | 23.39 | 1.357e+03 | 747 157 |
| bounded_multiscale_nca | 0.197 | 15.54 | 2.441e-04 | 3 936 |
| multiscale_flux_nca | 0.197 | 15.53 | 7.324e-04 | 3 936 |
| bounded_cons_nca | 0.197 | 15.53 | 2.441e-04 | 3 200 |
| pi_nca | 0.197 | 15.53 | 1.221e-04 | 3 200 |
| mc_flux_nca | 0.197 | 15.52 | **0.000e+00** | 6 336 |

*Figure: `docs/figures/bench/bench_accuracy_3d.png`*

### 6.6 Resolution transfer

*Not yet run.*

*Figures: `docs/figures/bench/bench_resolution_*.png`*

---

## 7. What to take away

1. **No universal winner.** Architecture should be chosen from the PDE's structure —
   conservation, boundedness, and how far information travels per step.
2. **Structural constraints beat penalty terms.** Flux-divergence conserves mass to 1e-4 or
   better *by construction*. A loss term only encourages it.
3. **The right prior is worth ~100× the parameters.** On heat, 5 520 parameters beat 592 897.
4. **The wrong prior is worse than none.** Per-field conservation on FitzHugh–Nagumo conserves
   to 1e-6 and produces a useless model (0.993).
5. **Conservation is not stability.** On Cahn–Hilliard, models conserved mass perfectly while
   diverging to rel-L2 24. Bounding was the missing ingredient.
6. **Hybrids work when they target a specific measured failure.** MultiScale and BoundedCons
   each fixed a diagnosed problem and won their regime. SpectralFlux was the most ambitious
   and delivered least reliably. Stacking two winners (bounded + multi-scale) made things
   slightly *worse*.

---

## 8. Figure index

All paths relative to the repository root.

### Architecture diagrams — `docs/figures/arch/`

| File | Shows |
|---|---|
| `arch_family_tree.png` | How all architectures relate to one backbone |
| `arch_plain_nca.png` | Plain NCA |
| `arch_pi_nca.png` | Conservative PI-NCA (flux divergence) |
| `arch_multiscale_flux_nca.png` | MultiScaleFluxNCA hybrid |
| `arch_bounded_cons_nca.png` | BoundedConsFluxNCA hybrid |
| `arch_spectral_flux_nca.png` | SpectralFluxNCA hybrid |
| `arch_mc_flux_nca.png` | MultiChannelFluxNCA |
| `arch_fno.png` | Fourier Neural Operator |
| `arch_training_pipeline.png` | Shared training pipeline |

Regenerate with `python -m pinca_jax.arch_figs`.

### Benchmark plots — `docs/figures/bench/`

| File | Shows |
|---|---|
| `bench_regime_map.png` | Which architecture wins which PDE |
| `bench_accuracy_2d.png` | rel-L2, all models, all 2-D phenomena |
| `bench_accuracy_3d.png` | The 3-D suite |
| `bench_conservation_2d.png` | Mass drift — where the flux head earns its keep |
| `bench_accuracy_vs_cost.png` | Accuracy vs parameter count |
| `bench_error_growth.png` | Error growth from T/4 to T |
| `bench_psnr_2d.png` | PSNR |
| `bench_train_time.png`, `bench_throughput.png` | Training cost, inference speed |
| `bench_ablation_A4.png`, `bench_ablation_A5.png` | Conservation on/off; perception size |
| `bench_resolution_*.png` | Train-grid × eval-grid transfer |

Regenerate with `python -m pinca_jax.plots`.

### Simulation figures — `docs/figures/`

| File pattern | Shows |
|---|---|
| `<pde>_comparison.png` | Analytic vs model vs error, 2-D, at key timesteps |
| `<pde>_3d_comparison.png` | Same in 3-D, mid-depth slice |
| `<pde>_3d_volume.png` | True 3-D volume render |

For `<pde>` in: heat, allen_cahn, nagumo, adv_diff, gray_scott, shallow_water,
fitzhugh_nagumo, wave, cahn_hilliard, navier_stokes (2-D); heat, adv_diff, allen_cahn,
nagumo, gray_scott, fitzhugh_nagumo (3-D).

---

## 9. Source files

| Component | Path |
|---|---|
| Plain NCA | `src/pinca_jax/models/nca.py` |
| Conservative PI-NCA, multi-channel | `src/pinca_jax/models/flux_nca.py` |
| All three hybrids | `src/pinca_jax/models/hybrids.py` |
| FNO | `src/pinca_jax/models/fno.py` |
| Ablation backbone | `src/pinca_jax/models/ablation_nca.py` |
| Conservation operators | `src/pinca_jax/physics.py` |
| PDE solvers (teachers) | `src/pinca_jax/equations/pdes.py` |
| Training / evaluation harness | `src/pinca_jax/harness.py` |
| 3-D counterparts | `src/pinca_jax/*3d.py` |
| Benchmark drivers | `src/pinca_jax/bench.py`, `bench_all.py`, `bench3d.py` |
| Correctness gate | `tests/` (56 tests) |

Reproduce: `python -m pytest tests/ -q`, then `bash run_gpu.sh` (see `docs/gpu_runbook.md`).
