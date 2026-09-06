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
| `fno_small` | iso-parameter FNO (~NCA budget) — A2 ablation: spectral mixing vs param count | — |
| `mc_flux_nca` | multi-channel per-field conservative flux NCA (SWE/FHN/GS) | — |
| `resnet` | autoregressive residual CNN, circular pad (~7.4e4 params) | — |
| `resnet_iso` | iso-parameter CNN control (~5.4e3, matched to the NCA budget) | 5 364 |
| `unet` | multi-resolution U-Net emulator (~2.7e5 params) | — |
| `unet_iso` | iso-parameter U-Net control (~7.5e3, matched to the NCA budget) | — |
| `identity` | do-nothing floor g(x)=x -- any model above this learned worse than nothing | 1 |
| `bounded_cons_nca` | flux NCA + clip + mass re-projection (bounded AND conserving) | — |
| `spectral_flux_nca` | local conservative flux + global FNO spectral correction | 134 225 |
| `multiscale_flux_nca` | dilated multi-scale perception + conservative flux | 5 520 |
| `bounded_multiscale_nca` | UNIFIED: multi-scale perception + bounded + mass-conserving (stiff bounded fields) | — |

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
| Heat | 0.027 | — | — | **0.006** | fno 0.021 |
| Advection-diffusion | 0.018 | — | — | — | fno 0.007 |
| Allen-Cahn | 0.053 | — | — | — | fno 0.007 |
| Nagumo | 0.376 | — | — | — | plain_nca 0.073 |
| Wave | — | — | — | — | pi_nca 0.042 |
| Cahn-Hilliard | 0.088 | 0.083 | 0.088 | 0.046 | fno 0.051 |
| Gray-Scott | — | — | — | — | pi_nca 0.410 |
| Shallow-water | — | — | — | — | pi_nca 0.013 |
| FitzHugh-Nagumo | — | — | — | — | plain_nca 0.125 |
| Navier-Stokes | — | — | — | — | pi_nca 0.192 |

rel-L2, lower is better. **Bold** marks the overall winner for that phenomenon across all architectures.

| Hybrid | What it targets | Outcome |
|---|---|---|
| `multiscale_flux_nca` | widen the receptive field without an FFT | does not win any phenomenon outright |
| `bounded_cons_nca` | be bounded AND mass-conserving at once | does not win any phenomenon outright |
| `bounded_multiscale_nca` | combine multi-scale reach with bounding | does not win any phenomenon outright |
| `spectral_flux_nca` | add global spectral reach to a local conservative NCA | wins Heat |

---

## 6. Comprehensive results

Every table in this section is **generated from `results/*.json`** by
`python -m pinca_jax.report`, so the document cannot quote a different set of
architectures from the one the benchmarks actually ran. A dash means that cell has not
been measured yet; re-run `bash run_gpu.sh` and regenerate to fill it in.

Measured on **cpu** (`cpu:0`), JAX 0.10.1, grid 12, batch 4, 25 epochs, train horizon 3 / eval horizon 8, 1 seed.

Total run time **0.68 h**. Peak device memory 0 MB.

Single fixed seed (42), so no ± is shown; the protocol is the originals' He-init + zero-init heads + LR warm-up + pre-seeding.

### 6.1 The regime map — which architecture wins where

| PDE | Character | Winner | rel-L2 | Runner-up | Models compared |
|---|---|---|---|---|---|
| Heat | smooth, local, conservative | **spectral_flux_nca** | 0.006 | fno 0.021 | 7 |
| Advection-diffusion | linear transport, conservative | **fno** | 0.007 | pi_nca 0.012 | 6 |
| Allen-Cahn | non-conservative phase separation | **fno** | 0.007 | resnet_iso 0.026 | 6 |
| Nagumo | non-conservative bistable | **resnet_iso** | 0.022 | plain_nca 0.073 | 6 |
| Wave | 2nd-order hyperbolic | **pi_nca** | 0.042 | resnet_iso 0.048 | 6 |
| Cahn-Hilliard | stiff 4th-order, bounded | **resnet** | 0.044 | spectral_flux_nca 0.046 | 14 |
| Gray-Scott | reaction-diffusion patterns | **resnet_iso** | 0.213 | identity 0.389 | 6 |
| Shallow-water | conservative, multi-field | **pi_nca** | 0.013 | mc_flux_nca 0.016 | 6 |
| FitzHugh-Nagumo | non-conservative reaction | **plain_nca** | 0.125 | fno 0.199 | 6 |
| Navier-Stokes | globally coupled | **resnet_iso** | 0.190 | pi_nca 0.192 | 4 |

*Figure: `docs/figures/bench/bench_regime_map.png`*

### 6.2 The full 2-D matrix

Every architecture on every phenomenon, same list throughout.

**Heat** — smooth, local, conservative, C=1, grid 12, eval 8 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **spectral_flux_nca** | **6.365e-03** | 55.56 | 1.728e-03 | 134 225 | 2.424e-03 |
| fno | 2.102e-02 | 45.19 | 9.846e+00 | 592 897 | 4.667e-03 |
| multiscale_flux_nca | 2.749e-02 | 42.85 | 1.850e-03 | 5 520 | 8.546e-04 |
| pi_nca | 3.100e-02 | 41.81 | **3.128e-04** | 4 576 | 5.981e-04 |
| resnet_iso | 2.239e-01 | 26.27 | 8.920e+00 | 5 364 | 8.898e-04 |
| plain_nca | 2.242e-01 | 24.63 | 3.972e+01 | 6 784 | 9.823e-04 |
| identity | 3.608e-01 | 22.13 | 1.140e+00 | **1** | 8.850e-06 |

**Advection-diffusion** — linear transport, conservative, C=1, grid 12, eval 8 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **fno** | **6.575e-03** | 57.16 | 1.202e+00 | 592 897 | 1.001e-02 |
| pi_nca | 1.231e-02 | 51.71 | **3.891e-04** | 4 576 | 6.815e-04 |
| multiscale_flux_nca | 1.816e-02 | 48.34 | 1.595e-03 | 5 520 | 1.038e-03 |
| plain_nca | 2.220e-02 | 46.59 | 9.508e+00 | 6 784 | 1.291e-03 |
| resnet_iso | 7.185e-02 | 38.19 | 6.014e+00 | 5 364 | 1.229e-03 |
| identity | 1.351e-01 | 32.71 | 7.451e-01 | **1** | 5.900e-06 |

**Allen-Cahn** — non-conservative phase separation, C=1, grid 12, eval 8 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **fno** | **6.848e-03** | 49.50 | 7.072e-01 | 592 897 | 4.226e-03 |
| resnet_iso | 2.642e-02 | 38.18 | 4.385e-01 | 5 364 | 1.014e-03 |
| identity | 2.663e-02 | 38.11 | **0.000e+00** | **1** | 7.700e-06 |
| plain_nca | 4.907e-02 | 32.39 | 2.547e-01 | 6 784 | 6.710e-04 |
| pi_nca | 4.922e-02 | 32.36 | 1.431e-05 | 4 576 | 8.806e-04 |
| multiscale_flux_nca | 5.274e-02 | 31.76 | 8.100e-05 | 5 520 | 7.864e-04 |

**Nagumo** — non-conservative bistable, C=1, grid 12, eval 8 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **resnet_iso** | **2.204e-02** | 34.90 | 7.270e+00 | 5 364 | 8.552e-04 |
| plain_nca | 7.333e-02 | 22.44 | 1.930e+02 | 6 784 | 6.538e-04 |
| identity | 8.043e-02 | 23.66 | **0.000e+00** | **1** | 1.350e-05 |
| fno | 8.068e-02 | 21.61 | 1.392e+02 | 592 897 | 5.983e-03 |
| multiscale_flux_nca | 3.763e-01 | 8.24 | 6.828e-04 | 5 520 | 1.391e-03 |
| pi_nca | 3.768e-01 | 8.22 | 1.106e-04 | 4 576 | 7.949e-04 |

**Wave** — 2nd-order hyperbolic, C=2, grid 12, eval 8 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **pi_nca** | **4.243e-02** | 45.33 | 2.147e-01 | 4 928 | 2.913e-04 |
| resnet_iso | 4.840e-02 | 44.19 | 3.686e+00 | 5 484 | 8.297e-04 |
| plain_nca | 5.205e-02 | 41.53 | 5.636e+00 | 7 264 | 1.480e-03 |
| mc_flux_nca | 5.571e-02 | 40.94 | **1.373e-04** | 10 464 | 2.564e-03 |
| fno | 5.650e-02 | 40.81 | 5.188e-01 | 592 946 | 4.585e-03 |
| identity | 7.550e-02 | 40.33 | 2.276e-01 | **1** | 1.175e-05 |

**Cahn-Hilliard** — stiff 4th-order, bounded, C=1, grid 16, eval 12 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **resnet** | **4.409e-02 ± 2.382e-05** | 42.16 | 5.678e-02 | 74 336 | 3.203e-03 |
| spectral_flux_nca | 4.592e-02 ± 1.944e-03 | 41.81 | 1.069e-05 | 134 225 | 1.400e-03 |
| fno | 5.078e-02 ± 1.738e-03 | 40.94 | 2.513e-01 | 592 897 | 6.200e-03 |
| unet | 5.138e-02 ± 2.609e-03 | 40.84 | 5.708e-02 | 265 104 | 4.012e-03 |
| mc_flux_nca | 6.938e-02 ± 4.533e-03 | 38.23 | 1.766e-06 | 9 936 | 1.442e-03 |
| resnet_iso | 7.253e-02 ± 2.939e-03 | 37.84 | 1.264e-01 | 5 364 | 1.827e-03 |
| pi_nca | 8.294e-02 ± 4.272e-03 | 36.68 | 1.673e-06 | 4 576 | 6.055e-04 |
| bounded_cons_nca | 8.294e-02 ± 4.273e-03 | 36.68 | 2.477e-06 | 4 576 | 3.004e-04 |
| plain_nca | 8.324e-02 ± 1.926e-03 | 36.64 | 1.712e-01 | 6 784 | 6.109e-04 |
| bounded_multiscale_nca | 8.782e-02 ± 1.954e-03 | 36.18 | 2.034e-06 | 5 520 | 7.244e-04 |
| multiscale_flux_nca | 8.782e-02 ± 1.954e-03 | 36.18 | 2.831e-06 | 5 520 | 5.786e-04 |
| fno_small | 1.510e-01 ± 9.795e-03 | 31.48 | 2.171e-01 | 8 433 | 7.293e-04 |
| unet_iso | 1.632e-01 ± 3.003e-03 | 30.80 | 3.571e-01 | 7 464 | 1.460e-03 |
| identity | 4.104e-01 ± 7.468e-03 | 22.79 | **0.000e+00** | **1** | 1.185e-05 |

**Gray-Scott** — reaction-diffusion patterns, C=2, grid 12, eval 8 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **resnet_iso** | **2.127e-01** | 19.90 | 1.749e+01 | 5 484 | 9.527e-04 |
| identity | 3.889e-01 | 14.65 | **0.000e+00** | **1** | 7.350e-06 |
| pi_nca | 4.103e-01 | 14.19 | 1.831e-04 | 4 928 | 3.290e-04 |
| fno | 6.741e-01 | 10.11 | 1.729e+02 | 592 946 | 8.026e-03 |
| mc_flux_nca | 6.916e-01 | 9.89 | 1.450e-04 | 10 464 | 1.317e-03 |
| plain_nca | 1.762e+00 | 1.77 | 1.113e+02 | 7 264 | 1.009e-03 |

**Shallow-water** — conservative, multi-field, C=3, grid 12, eval 8 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **pi_nca** | **1.252e-02** | 46.34 | 4.768e-05 | 5 280 | 3.326e-04 |
| mc_flux_nca | 1.613e-02 | 44.14 | 1.373e-04 | 10 992 | 1.456e-03 |
| resnet_iso | 1.991e-02 | 42.31 | 6.059e-01 | 5 604 | 8.371e-04 |
| fno | 2.445e-02 | 40.53 | 2.140e+00 | 592 995 | 1.016e-02 |
| plain_nca | 2.618e-02 | 39.93 | 9.869e+00 | 7 744 | 1.185e-03 |
| identity | 2.918e-02 | 38.99 | **0.000e+00** | **1** | 1.370e-05 |

**FitzHugh-Nagumo** — non-conservative reaction, C=2, grid 12, eval 8 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **plain_nca** | **1.247e-01** | 26.48 | 9.217e+01 | 7 264 | 2.270e-03 |
| fno | 1.986e-01 | 22.44 | 8.621e+01 | 592 946 | 1.051e-02 |
| pi_nca | 9.121e-01 | 17.41 | 3.874e-07 | 4 928 | 2.644e-04 |
| resnet_iso | 9.709e-01 | 16.87 | 1.260e+00 | 5 484 | 1.419e-03 |
| mc_flux_nca | 9.986e-01 | 8.41 | 1.668e-06 | 10 464 | 1.097e-03 |
| identity | 1.045e+00 | 16.23 | **0.000e+00** | **1** | 7.500e-06 |

**Navier-Stokes** — globally coupled, C=1, grid 16, eval 12 steps

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ | Infer s/step ↓ |
|---|---|---|---|---|---|
| **resnet_iso** | **1.902e-01 ± 3.953e-02** | 34.48 | 7.337e-01 | 5 364 | 1.029e-03 |
| pi_nca | 1.918e-01 ± 3.887e-02 | 34.40 | 6.020e-06 | 4 576 | 1.841e-04 |
| identity | 1.970e-01 ± 3.667e-02 | 34.16 | **0.000e+00** | **1** | 5.775e-06 |
| plain_nca | 1.983e-01 ± 3.743e-02 | 34.10 | 1.235e+00 | 6 784 | 2.521e-04 |

Full 20-metric tables per phenomenon: `results/bench_<pde>_full.md`.

### 6.3 Ablations — which component actually matters

**A4 — conservation on/off at matched backbone width**

Same backbone, same widths; only the head differs (flux vs residual). The cleanest test of whether the conservation prior helps.

| PDE | variant 1 | variant 2 |
|---|---|---|
| Heat | **0.024** (abl_flux) | 0.046 (abl_residual) |
| Nagumo | 0.110 (abl_flux) | **0.014** (abl_residual) |

**A5 — perception / receptive-field size**

Same head, same widths; only the perception differs (3x3, 5x5, dilated 1/2/4).

| PDE | variant 1 | variant 2 | variant 3 |
|---|---|---|---|
| Heat | **0.024** (abl_k3) | 0.050 (abl_k5) | 0.037 (abl_multiscale) |
| Navier-Stokes | 0.172 (abl_k3) | 0.139 (abl_k5) | **0.110** (abl_multiscale) |

*Figures: `docs/figures/bench/bench_ablation_A4.png`, `bench_ablation_A5.png`*

### 6.4 Efficiency — what accuracy costs

Cost of accuracy on heat, as rel-L2 x parameters (lower is better). This is where the conservative NCAs' small size shows up as more than a footnote.

| Model | rel-L2 | Params | rel-L2 x params | vs best |
|---|---|---|---|---|
| **identity** | 0.361 | 1 | 3.608e-01 | 1x |
| pi_nca | 0.031 | 4 576 | 1.419e+02 | 393x |
| multiscale_flux_nca | 0.027 | 5 520 | 1.518e+02 | 421x |
| spectral_flux_nca | 0.006 | 134 225 | 8.544e+02 | 2368x |
| resnet_iso | 0.224 | 5 364 | 1.201e+03 | 3328x |
| plain_nca | 0.224 | 6 784 | 1.521e+03 | 4215x |
| fno | 0.021 | 592 897 | 1.246e+04 | 34541x |

*Figure: `docs/figures/bench/bench_accuracy_vs_cost.png`*

### 6.5 The full 3-D matrix

**Advection-diffusion** — 8³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **pi_nca** | **0.186** | 31.59 | 3.433e-05 | 3 200 |
| bounded_cons_nca | 0.186 | 31.59 | 8.011e-05 | 3 200 |
| mc_flux_nca | 0.218 | 30.19 | **1.907e-05** | 6 336 |
| multiscale_flux_nca | 0.224 | 29.97 | 1.221e-04 | 3 936 |
| bounded_multiscale_nca | 0.224 | 29.97 | 1.221e-04 | 3 936 |
| plain_nca | 0.235 | 29.57 | 1.056e+01 | 3 072 |
| fno | 0.300 | 27.43 | 4.549e+00 | 277 141 |

**Allen-Cahn** — 8³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **fno** | **0.042** | 33.78 | 6.084e-01 | 277 141 |
| multiscale_flux_nca | 0.077 | 28.56 | 5.722e-06 | 3 936 |
| bounded_multiscale_nca | 0.077 | 28.56 | 5.722e-06 | 3 936 |
| plain_nca | 0.079 | 28.39 | 2.683e+01 | 3 072 |
| mc_flux_nca | 0.092 | 27.01 | **3.338e-06** | 6 336 |
| bounded_cons_nca | 0.098 | 26.50 | 7.391e-06 | 3 200 |
| pi_nca | 0.098 | 26.49 | 4.172e-06 | 3 200 |

**FitzHugh-Nagumo** — 8³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **fno** | **0.950** | 9.44 | 7.206e+01 | 277 166 |
| mc_flux_nca | 1.121 | 8.01 | 1.967e-06 | 7 920 |
| pi_nca | 1.275 | 6.89 | **1.818e-06** | 4 256 |
| bounded_cons_nca | 1.275 | 6.89 | 3.874e-06 | 4 256 |
| multiscale_flux_nca | 1.310 | 6.65 | 4.947e-06 | 5 208 |
| bounded_multiscale_nca | 1.310 | 6.65 | 4.947e-06 | 5 208 |
| plain_nca | 1.864 | 3.59 | 1.574e+02 | 4 000 |

**Gray-Scott** — 8³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **fno** | **0.345** | 17.18 | 4.596e+00 | 277 166 |
| plain_nca | 0.408 | 15.73 | 6.951e+01 | 4 000 |
| bounded_multiscale_nca | 0.630 | 11.95 | 1.221e-03 | 5 208 |
| multiscale_flux_nca | 0.654 | 11.62 | 3.357e-04 | 5 208 |
| bounded_cons_nca | 0.654 | 11.62 | 2.121e-03 | 4 256 |
| mc_flux_nca | 0.680 | 11.28 | **3.052e-05** | 7 920 |
| pi_nca | 0.709 | 10.93 | 5.341e-05 | 4 256 |

**Heat** — 8³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **mc_flux_nca** | **1.155** | 8.06 | 4.578e-05 | 6 336 |
| plain_nca | 1.286 | 7.13 | 4.975e+01 | 3 072 |
| fno | 1.403 | 6.37 | 5.786e+00 | 277 141 |
| bounded_cons_nca | 1.427 | 6.22 | 1.945e-04 | 3 200 |
| pi_nca | 1.687 | 4.77 | **1.526e-05** | 3 200 |
| bounded_multiscale_nca | 2.186 | 2.52 | 4.463e-04 | 3 936 |
| multiscale_flux_nca | 3.541 | -1.67 | 1.221e-04 | 3 936 |

**Nagumo** — 8³

| Model | rel-L2 ↓ | PSNR ↑ | Mass drift ↓ | Params ↓ |
|---|---|---|---|---|
| **mc_flux_nca** | **0.214** | 13.97 | 9.918e-05 | 6 336 |
| bounded_cons_nca | 0.217 | 13.85 | 9.155e-05 | 3 200 |
| pi_nca | 0.217 | 13.85 | **3.815e-05** | 3 200 |
| plain_nca | 0.226 | 13.51 | 3.437e+01 | 3 072 |
| multiscale_flux_nca | 0.228 | 13.42 | 1.602e-04 | 3 936 |
| bounded_multiscale_nca | 0.228 | 13.42 | 1.602e-04 | 3 936 |
| fno | 0.262 | 12.21 | 8.006e+01 | 277 141 |

*Figure: `docs/figures/bench/bench_accuracy_3d.png`*

### 6.6 Resolution transfer

**Allen-Cahn / `multiscale_flux_nca`** (rel-L2)

|  | eval 16² | eval 24² | eval 32² | eval 48² |
|---|---|---|---|---|
| train 16² | **0.054** | 0.055 | 0.056 | 0.056 |
| train 24² | 0.051 | **0.051** | 0.052 | 0.052 |
| train 32² | 0.049 | 0.050 | **0.051** | 0.050 |
| train 48² | 0.051 | 0.052 | 0.053 | **0.053** |

**Allen-Cahn / `fno`** (rel-L2)

|  | eval 16² | eval 24² | eval 32² | eval 48² |
|---|---|---|---|---|
| train 16² | **0.021** | 0.023 | 0.024 | 0.026 |
| train 24² | 0.016 | **0.012** | 0.014 | 0.016 |
| train 32² | 0.020 | 0.012 | **0.012** | 0.013 |
| train 48² | 0.032 | 0.020 | 0.016 | **0.015** |

**Heat / `pi_nca`** (rel-L2)

|  | eval 16² | eval 24² | eval 32² | eval 48² |
|---|---|---|---|---|
| train 16² | **0.086** | 0.053 | 0.048 | 0.045 |
| train 24² | 0.110 | **0.034** | 0.036 | 0.039 |
| train 32² | 0.178 | 0.053 | **0.022** | 0.021 |
| train 48² | 0.380 | 0.122 | 0.048 | **0.017** |

**Heat / `multiscale_flux_nca`** (rel-L2)

|  | eval 16² | eval 24² | eval 32² | eval 48² |
|---|---|---|---|---|
| train 16² | **0.076** | 0.046 | 0.037 | 0.029 |
| train 24² | 0.111 | **0.036** | 0.030 | 0.026 |
| train 32² | 0.209 | 0.054 | **0.018** | 0.018 |
| train 48² | 0.349 | 0.125 | 0.042 | **0.008** |

**Heat / `fno`** (rel-L2)

|  | eval 16² | eval 24² | eval 32² | eval 48² |
|---|---|---|---|---|
| train 16² | **0.070** | 0.302 | 0.398 | 0.504 |
| train 24² | 0.335 | **0.040** | 0.159 | 0.294 |
| train 32² | 0.542 | 0.174 | **0.017** | 0.160 |
| train 48² | 0.782 | 0.381 | 0.164 | **0.006** |

**Navier-Stokes / `multiscale_flux_nca`** (rel-L2)

|  | eval 16² | eval 24² | eval 32² | eval 48² |
|---|---|---|---|---|
| train 16² | **0.295** | 0.519 | 0.506 | 0.639 |
| train 24² | 0.462 | **0.344** | 0.399 | 0.590 |
| train 32² | 0.737 | 0.651 | **0.547** | 0.666 |
| train 48² | 2.387 | 1.919 | 1.309 | **1.252** |

**Navier-Stokes / `fno`** (rel-L2)

|  | eval 16² | eval 24² | eval 32² | eval 48² |
|---|---|---|---|---|
| train 16² | **0.211** | 0.208 | 0.233 | 0.215 |
| train 24² | 0.189 | **0.183** | 0.223 | 0.210 |
| train 32² | 0.214 | 0.223 | **0.237** | 0.213 |
| train 48² | 0.202 | 0.203 | 0.218 | **0.219** |

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
