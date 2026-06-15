# Literature Review: NCA, PINNs, Operator Learning, and Hybrids for PDE-Governed Systems

**Project:** PI-NCA — Physics-Informed Neural Cellular Automata vs. PINNs and Neural Operators
**Status:** Living document. Citations are real arXiv/journal works; arXiv IDs given as `[arXiv:ID]`.
**Scope:** This review grounds the architecture and experiment design. Each family is given (1) a mathematical formulation, (2) computational complexity, (3) memory complexity, (4) expected strengths, (5) expected weaknesses, and (6) citations.

> Reviewer's note on honesty (per research standards): the goal is *not* to show NCAs win. The point is to characterize the regimes where PINNs, NCAs, operators, and hybrids each win. Claims below are flagged as **established**, **plausible**, or **to-verify** so the experimental program can target the open questions.

---

## 0. Problem setting

We study time-dependent PDEs of the form

    ∂u/∂t = N[u; θ_phys],    x ∈ Ω,  t ∈ [0,T],
    B[u] = 0 on ∂Ω,   u(x,0) = u₀(x),

where `N` is a (possibly nonlinear) spatial operator. The repository's equations are concrete instances:

| Equation | Operator `N[u]` | Type | Branch |
|---|---|---|---|
| Heat / diffusion | `α ∇²u` | linear parabolic | `main`, all |
| Heterogeneous heat | `∇·(α(x) ∇u)` | linear, variable-coeff | `Heterogenous-simulations` |
| Gray–Scott RD | `D_u∇²u − uv² + F(1−u)`, `D_v∇²v + uv² − (F+k)v` | nonlinear reaction–diffusion | `PI-NCA-Gray-Heat-Equation` |
| Shallow-water (SWE) | hyperbolic conservation laws (h, hu, hv) | nonlinear hyperbolic | `new-update-of-physics-informed-models` |
| FitzHugh–Nagumo (FHN) | excitable RD (activator/inhibitor) | nonlinear RD | same |
| Cahn–Hilliard (CH) | `∇²(−ε²∇²u + W'(u))` | 4th-order parabolic | same |

Two distinct *learning problems* must not be conflated:

- **Solving one IVP** (PINN's native setting): fit `u(x,t)` for a *single* `u₀`, `θ_phys`.
- **Learning the solution operator** (operator/NCA-emulator setting): learn `G: u₀ ↦ u(·,t)` (or one-step `u(t) ↦ u(t+Δt)`) that *generalizes across* `u₀`.

The repo's NCA learns a **local one-step update** trained to imitate a numerical solver — i.e. it is an **autoregressive neural emulator** of the solution operator, the same regime benchmarked by APEBench `[arXiv:2411.00180]`. This distinction drives the entire comparison: PINNs and emulators are not measured on the same axis unless the protocol is designed carefully (see `docs/experimental_methodology.md`).

---

## 1. Neural Cellular Automata (NCA)

### Formulation
An NCA maintains a state grid `S ∈ R^{H×W×C}` (C channels: visible + hidden). One step applies an identical learned local rule `f_θ` to every cell using a fixed *perception* stencil (Sobel/Laplacian/identity depth-wise convolutions giving `P(S)`), then a pointwise MLP (1×1 convolutions):

    S_{t+1} = S_t + m ⊙ f_θ(P(S_t)),

where `m` is a stochastic per-cell update mask (asynchrony) and the residual "+S_t" makes the rule an **explicit Euler step of a learned dynamical system**. Training is by backprop-through-time (BPTT) over a rollout, often with a *sample pool* for long-horizon stability.

- **Foundational:** Mordvintsev et al., "Growing Neural Cellular Automata," *Distill* 2020 (the canonical differentiable NCA); Mordvintsev et al., "Self-classifying MNIST" 2020. Classical CA↔PDE links predate this `[arXiv:1003.1983]`.
- **Texture / dynamics:** μNCA `[arXiv:2111.13545]`, DyNCA `[arXiv:2211.11417]`, NoiseNCA `[arXiv:2404.06279]` (continuous space-time; directly relevant to PDE NCAs).
- **Geometry / robustness:** Isotropic `[arXiv:2205.01681]`, Steerable `[arXiv:2302.10197]`.
- **Surveys / frontiers:** "From Cells to Pixels" `[arXiv:2506.22899]`, "Path to Universal NCA" `[arXiv:2505.13058]`, Neural Particle Automata (Lagrangian NCA) `[arXiv:2601.16096]`, DiffLogic CA `[arXiv:2506.04912]`.
- **Accelerator:** CAX `[arXiv:2410.02651]` — JAX library for hardware-accelerated (N)CA; we evaluate it directly.

### Complexity
Per step, the cost is dominated by the 1×1-conv MLP: **O(H·W·C·F)** time for hidden width `F`, i.e. *linear in the number of cells*. A `K`-step rollout costs **O(K·H·W·C·F)**.

### Memory
- Inference: **O(H·W·C)** (one state grid).
- Training (naïve BPTT over `K` steps): **O(K·H·W·C)** activations — the dominant cost, and the reason the repo uses truncated BPTT (`truncate_depth`). Gradient checkpointing trades compute for **O(√K)** memory.

### Strengths (expected)
- **Locality / translation equivariance** is the correct inductive bias for local PDE stencils — matches finite-difference structure exactly (heat kernel = discrete Laplacian). **Established** for diffusion-like operators.
- **Resolution/temporal extrapolation** and **regeneration/robustness** to perturbations. **Plausible** for PDEs (NoiseNCA, isotropic NCA).
- **Cheap inference**, parameter-efficient (μNCA: <1k params).

### Weaknesses (expected)
- **Long-horizon instability / error accumulation** in autoregressive rollouts; needs pool training, normalization, or stability regularization. **Established** (APEBench `[arXiv:2411.00180]`).
- **Locality cannot represent global/fast operators** (pressure projection in SWE, 4th-order CH coupling) in one step — information propagates one cell/step. **To-verify**: this is our central hypothesis for where NCAs underperform operators.
- **BPTT memory** limits horizon at training time.

---

## 2. Physics-Informed Neural Networks (PINNs)

### Formulation
A network `u_φ(x,t)` is trained to minimize a composite residual loss using autodiff for derivatives:

    L = λ_r ‖∂_t u_φ − N[u_φ]‖²_Ω×[0,T] + λ_b ‖B[u_φ]‖²_∂Ω + λ_0 ‖u_φ(·,0) − u₀‖².

No mesh, no labeled interior data; the PDE *is* the supervision.

- **Foundational:** Raissi, Perdikaris, Karniadakis, *J. Comput. Phys.* 2019 `[arXiv:1711.10561, 1711.10566]`.
- **Reviews:** "Where we are and what's next" `[arXiv:2201.05624]`; adaptive-PINN survey `[arXiv:2503.18181]`; limitations `[arXiv:2411.18240]`.
- **Pathologies & fixes:** gradient/loss balancing & AutoBalance `[arXiv:2510.06684]`; causal/temporal sweeping `[arXiv:2302.14227]`; Fourier-feature & spectral-bias mitigation `[arXiv:2410.03496, arXiv:2602.19265]`; preconditioning `[arXiv:2402.00531]`; convergence/error analysis `[arXiv:2305.01240]`.

### Complexity
Cost is per **collocation point**, not per grid cell. Each residual needs higher-order autodiff (Laplacian ⇒ 2nd derivatives). Time per step ≈ **O(N_col · cost(∇²u_φ))**; for an MLP of width `W`, depth `L`, a forward is O(W²L) and 2nd-order AD multiplies this by a constant factor and the input dimension. Training is the expensive phase; **inference is a cheap forward pass at any (x,t)** — mesh-free, continuous.

### Memory
**O(N_col · W·L)** for the AD graph over a batch; independent of any grid. Higher-order derivatives enlarge the graph.

### Strengths
- **Mesh-free, continuous** representation; trivially queried at arbitrary `(x,t)`; handles irregular domains and inverse problems. **Established.**
- **No training data** beyond the equation; strong in data-scarce regimes.
- Excellent for **smooth, low-frequency** solutions.

### Weaknesses
- **Spectral bias**: struggles with high-frequency / multiscale / sharp features `[arXiv:2602.19265, arXiv:2410.03496]`. **Established.**
- **Training pathologies**: stiff loss balancing, gradient conflict, ill-conditioning `[arXiv:2510.06684, arXiv:2402.00531]`. **Established.**
- **Causality violations** for long time horizons unless explicitly enforced `[arXiv:2302.14227]`.
- Solves **one IVP per training run** — no cross-IC generalization without re-training (unlike operators/emulators).
- Hyperbolic/shock problems (SWE) are hard for vanilla PINNs.

---

## 3. Physics-Informed Neural Cellular Automata (PINCA)

### Formulation
A PINCA is an NCA whose update is constrained by physics, either (a) by training against a differentiable numerical solver (the repo's `DeepFluxNCA` + `HeatEquationSolver`), (b) by adding a PDE-residual loss on the NCA state, and/or (c) by **structurally encoding conservation** in the update. The repo's design is notable: it predicts a **flux** `(f_x, f_y)` and applies a discrete divergence

    S_{t+1} = S_t + (roll(f_x) − f_x) + (roll(f_y) − f_y),

making the update a **discrete conservation law** (a finite-volume flux form) — mass is conserved by construction up to boundary terms; the repo additionally projects energy (`conserve_energy`). This is a strong, physically-motivated inductive bias and a genuine contribution to characterize.

- Classical CA-as-PDE basis `[arXiv:1003.1983]`; continuous-time NCA (NoiseNCA `[arXiv:2404.06279]`) gives the dt→0 link; reference on the Heterogenous branch `[arXiv:2407.06151]` (to be read for the heterogeneity setup).
- Differentiable-physics / hybrid neural-physics framing: implicit neural differential models `[arXiv:2504.02260]`, differentiable solvers for ROM `[arXiv:2505.14595]`.

### Complexity / Memory
Same order as NCA (§1), plus the solver's cost during training if used as a teacher. Flux/divergence parameterization adds only constant channels.

### Strengths (expected)
- **Conservation by construction** (flux form) → far better long-horizon energy behavior than an unconstrained NCA. **Plausible→to-verify**; the repo already shows energy-drift tracking.
- Inherits NCA locality/equivariance; physics loss reduces data needs.

### Weaknesses (expected)
- Still local ⇒ shares NCA's difficulty with global/stiff operators.
- Flux form encodes *conservation* but not *correct dynamics* automatically; can conserve mass while getting the rate wrong.

---

## 4. Fourier Neural Operator (FNO)

### Formulation
FNO learns an operator `G: a ↦ u` between function spaces via spectral convolution. Each layer:

    v_{l+1}(x) = σ( W v_l(x) + F⁻¹( R_l ⊙ F[v_l] )(x) ),

where `F` is the (FFT-based) Fourier transform and `R_l` is a learned complex multiplier on the lowest `k_max` modes (higher modes truncated). Global convolution in one layer.

- **Foundational:** Li et al. `[arXiv:2010.08895]`; neural-operator framework `[arXiv:2108.08481]`; universality `[arXiv:2107.07562]`; Rademacher bounds `[arXiv:2209.05150]`.
- **Physics-informed variant:** PINO `[arXiv:2111.03794]` (adds PDE residual to FNO → less data, better generalization).

### Complexity
Per layer: FFT is **O(H·W·log(HW)·C)**; mode multiply **O(k_max·C²)**. **Quasi-linear** in grid size and, crucially, **global receptive field in O(1) layers** — the opposite of NCA's one-cell-per-step locality.

### Memory
**O(H·W·C)** for fields plus **O(k_max·C²)** for spectral weights per layer; modest.

### Strengths
- **Global, multiscale** operator in few layers; **resolution-invariant** (zero-shot super-resolution). **Established.**
- Strong **cross-IC generalization**; fast inference once trained.
- Natural for **smooth periodic** domains (the repo's periodic BCs fit FNO's FFT perfectly).

### Weaknesses
- **Periodic/regular-grid bias** from FFT; non-periodic BCs and complex geometry need care.
- **Spectral truncation** smooths sharp features/shocks (SWE) `[arXiv:2602.19265]`.
- Needs **paired training data** (solver trajectories) unless physics-informed (PINO).
- Less locally interpretable; larger spectral weights.

---

## 5. DeepONet

### Formulation
Learns `G(a)(y) ≈ Σ_k b_k(a) · t_k(y)` with a **branch** net encoding the input function `a` (sampled at sensors) and a **trunk** net encoding the query location `y`; output is their inner product. Grounded in the operator universal-approximation theorem.

- **Foundational:** Lu, Jin, Karniadakis `[arXiv:1910.03193]` (*Nat. Mach. Intell.* 2021).
- **Physics-informed / variants:** physics-informed DeepONet `[arXiv:2207.05748]`; resolution-independent variant `[arXiv:2407.13010]`; statistical view `[arXiv:2504.03503]`.

### Complexity / Memory
Branch+trunk forwards: **O(W²L)** each; output is a cheap inner product. Mesh-free in the query (trunk) like a PINN; memory dominated by branch sensor count and net widths.

### Strengths
- **Flexible output queries** (mesh-free trunk); solid theory; good for parametric operators and varying sensors. **Established.**
- Composes naturally with physics losses.

### Weaknesses
- **Fixed sensor layout** in vanilla form; can underperform FNO on structured grids; trunk can inherit PINN-style spectral bias.

## 5b. Graph Neural Operators (GNO)
Kernel-integral operators on graphs for **irregular meshes**; multipole variant gives near-linear global interaction `[arXiv:2006.09535]`; equivariant GNO for 3D dynamics `[arXiv:2401.11037]`. Strength: arbitrary geometry. Weakness: graph construction overhead; heavier than FNO on regular grids (our repo is regular-grid + periodic, so GNO is lower priority — noted for completeness).

---

## 6. Operator learning (general) & Differentiable PDE solvers / JAX SciML

- **Operator learning** unifies §4–5: learn maps between function spaces; statistical perspective `[arXiv:2504.03503]`, generalization limits `[arXiv:2602.23113]`.
- **Differentiable solvers / differentiable physics:** end-to-end gradients through a numerical solver enable hybrid learning and inverse design — TORAX (JAX tokamak solver) `[arXiv:2406.06718]`, implicit neural differential models `[arXiv:2504.02260]`, differentiable-solver ROM `[arXiv:2505.14595]`. The repo's `HeatEquationSolver` is exactly a differentiable solver used as a teacher.
- **JAX SciML ecosystem (to use):** JAX (XLA jit/vmap/scan/sharding); Flax (`nnx`/`linen`) for modules; Optax for optimization; Diffrax (ODE/PDE integration); Equinox (PyTree modules); **Exponax**/APEBench `[arXiv:2411.00180]` for spectral PDE references and autoregressive-emulator benchmarking; **CAX** `[arXiv:2410.02651]` for accelerated NCA. These define our migration target and best practices.

### Why JAX for this project (best-practice rationale)
- `jax.jit` + XLA fuses the NCA step; `jax.lax.scan` expresses rollouts with O(1) Python overhead and enables reverse-mode through the whole trajectory.
- `jax.vmap` batches over ICs/seeds without manual batching code; `jax.checkpoint` (rematerialization) controls BPTT memory.
- `pmap`/`shard_map`/`jax.sharding` give multi-device scaling **when hardware exists** (no-op here: single CPU — documented honestly).
- Functional purity → reproducibility via explicit PRNG keys (`jax.random`).

---

## 7. Hybrid NCA × Operator architectures (the central investigation)

Motivating gap: **NCA = local, cheap, conservation-friendly, but slow to propagate global information; FNO/operators = global, multiscale, resolution-invariant, but data-hungry and weak on sharp/conserved features.** They are complementary, suggesting hybrids:

1. **FNO-latent + NCA-refinement** — operator provides a global coarse estimate; NCA does local correction/conservation. (cf. multiscale neural emulators.)
2. **NCA local + operator global (two-stream / additive split)** — split the update into a local divergence-flux term and a spectral global term.
3. **Multi-scale / hierarchical NCA** — pyramid of NCAs (or dilated perception) to widen the receptive field per step, partially closing the locality gap without full FFT.
4. **Learned PDE operator coupled to NCA evolution** — operator predicts effective coefficients/forcing fed into a conservative NCA step (links to implicit neural differential models `[arXiv:2504.02260]`).
5. **Differentiable-physics-in-the-loop NCA** — NCA correction on top of a cheap differentiable solver step (residual/closure learning; cf. PINO `[arXiv:2111.03794]`, differentiable solvers `[arXiv:2406.06718]`).

These are testable hypotheses, not assumed wins. The ablations (§ ablation report) will isolate which component (locality, spectral global mixing, conservation structure, physics loss) actually drives any gain.

---

## 8. Synthesis → predictions to test

| Regime | Predicted best | Rationale | Confidence |
|---|---|---|---|
| Single smooth IVP, irregular query points | PINN / DeepONet | mesh-free, continuous | med |
| Local diffusion, long rollout, conservation matters | Conservative PI-NCA | flux-form mass conservation + locality matches stencil | med |
| Cross-IC generalization on periodic grid | FNO | global spectral op, resolution-invariant | high |
| Sharp/hyperbolic (SWE shocks) | FNO/operator or hybrid; PINNs weak | spectral global + data; locality too slow | med |
| Stiff 4th-order (Cahn–Hilliard) | FNO or hybrid | global coupling needed per step | med |
| Best overall robustness | **Hybrid (NCA local + operator/global)** | complementary biases | **to-verify (core hypothesis)** |

The experimental program (next deliverables) exists to confirm/refute this table with multi-seed mean±std metrics, **not** to assume it.

---

## References (arXiv IDs verified via discovery)
- NCA: Mordvintsev et al. *Growing NCA*, Distill 2020; `1003.1983`, `2111.13545`, `2211.11417`, `2404.06279`, `2205.01681`, `2302.10197`, `2506.22899`, `2505.13058`, `2601.16096`, `2506.04912`.
- PINN: Raissi et al. `1711.10561`, `1711.10566`; `2201.05624`, `2503.18181`, `2411.18240`, `2510.06684`, `2302.14227`, `2410.03496`, `2602.19265`, `2402.00531`, `2305.01240`.
- Operators: FNO `2010.08895`, `2108.08481`, `2107.07562`, `2209.05150`; PINO `2111.03794`; DeepONet `1910.03193`, `2207.05748`, `2407.13010`; GNO `2006.09535`, `2401.11037`; operator-learning theory `2504.03503`, `2602.23113`.
- Differentiable / JAX SciML: CAX `2410.02651`, APEBench `2411.00180`, TORAX `2406.06718`, `2504.02260`, `2505.14595`.
- Repo reference (heterogeneity branch): `2407.06151` (to be read in detail).
