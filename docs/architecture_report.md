# Architecture Report

For each implemented architecture: (1) mathematical formulation, (2) computational
complexity, (3) memory complexity, (4) expected strengths, (5) expected weaknesses,
(6) literature, (7) implementation notes. Symbols: grid `H×W=N` cells, channels `C`,
hidden width `F`, rollout/eval horizon `K`, FNO modes `m`, depth `L`.

All neural models are used as **one-step autoregressive emulators**
`g_θ: state_t ↦ state_{t+1}` trained by distillation against a differentiable
solver teacher (`equations/pdes.py`) under a shared harness (`harness.py`). PINNs
are the exception (continuous `(x,t)↦u`, single-IVP) and are evaluated separately.

---

## A. Plain NCA — `models/nca.py`
**Formulation.** Perceive with a learnable 3×3 circular conv `P(s)`, then a per-cell
MLP (1×1 convs) producing a residual increment:
`s_{t+1} = s_t + MLP(relu(P(s_t)))`, update head zero-initialised (starts as identity).
No conservation constraint.

**Compute.** O(N·C·F) per step (1×1-conv MLP dominates); rollout O(K·N·C·F). **Linear in cells.**
**Memory.** Inference O(N·C). Training BPTT O(K·N·F) activations (truncatable; `jax.checkpoint` → O(√K)).
**Strengths.** Locality/translation-equivariance matches local PDE stencils; cheap; parameter-light.
**Weaknesses.** No conservation ⇒ mass/energy drift; locality ⇒ slow global-information propagation; long-horizon error accumulation.
**Literature.** Mordvintsev et al. *Growing NCA* (Distill 2020); NoiseNCA `arXiv:2404.06279`; survey `arXiv:2506.22899`.

## B. Conservative PI-NCA (DeepFluxNCA) — `models/flux_nca.py`
**Formulation.** Same perceive+MLP backbone, but the head predicts a **flux field**
`(f_x,f_y)` and the update is its discrete divergence:
`s_{t+1} = s_t + (roll(f_x)−f_x) + (roll(f_y)−f_y)`.
Summed over a periodic grid the increment telescopes to **0** ⇒ **mass conserved by
construction** (finite-volume form). Optional explicit energy projection (`physics.conserve_energy`).

**Compute / Memory.** Same order as plain NCA (flux head adds one channel).
**Strengths.** Exact discrete conservation (verified: heat conservation error ~1e-4 vs ~10¹ for unconstrained); strong long-horizon mass behaviour; same cheap locality.
**Weaknesses.** Still local (global operators slow); conserves mass but not necessarily *correct rate*; flux form is native to scalar conserved fields (multi-channel needs per-field fluxes).
**Literature.** Classical CA↔PDE `arXiv:1003.1983`; conservation-structured/differentiable-physics framing `arXiv:2504.02260`.

## C. Fourier Neural Operator (FNO2d) — `models/fno.py`
**Formulation.** `v_{l+1} = σ(W v_l + F⁻¹(R_l ⊙ F[v_l]))`, truncating to the lowest
`m×m` Fourier modes; lift/project with 1×1 convs; residual one-step output.

**Compute.** Per layer FFT O(N log N · C) + mode-mix O(m²C²); rollout O(K·L·(N log N·C + m²C²)). **Global receptive field in O(1) layers.**
**Memory.** O(N·C) fields + O(L·m²·C²) spectral weights (here ~10⁵ params, ≫ the NCAs' ~5×10³).
**Strengths.** Global/multiscale mixing in few layers; resolution-invariant; native to periodic grids (FFT); strong cross-IC generalisation.
**Weaknesses.** Spectral truncation smooths sharp/shock features; periodic/regular-grid bias; many more parameters; no built-in conservation.
**Literature.** Li et al. `arXiv:2010.08895`; universality `arXiv:2107.07562`; PINO `arXiv:2111.03794`.

## D. PINN (planned, `research/baseline-pinn`)
**Formulation.** `u_φ(x,t)` minimising `λ_r‖∂_t u_φ − N[u_φ]‖² + λ_b‖B[u_φ]‖² + λ_0‖u_φ(·,0)−u_0‖²`
via autodiff derivatives; **single IVP per training run**, mesh-free.
**Compute.** O(N_col·cost(∇²u_φ)) per step (2nd-order AD); cheap inference at any (x,t).
**Memory.** O(N_col·W·L) AD graph; grid-free.
**Strengths.** Continuous, mesh-free, no labelled data, irregular query points, inverse problems.
**Weaknesses.** Spectral bias (high-freq hard), loss-balancing pathologies, causality for long T, no cross-IC reuse.
**Literature.** Raissi et al. `arXiv:1711.10561`; pathologies/fixes `arXiv:2510.06684`, `arXiv:2302.14227`.

> **Data-flow diagrams** for every hybrid (DeepFluxNCA, MultiScale, BoundedCons, SpectralFlux,
> MultiChannel) + the shared training pipeline are in
> [docs/architecture_diagrams.md](architecture_diagrams.md) (Mermaid, renders on GitHub).

## E. Hybrids (implemented — see diagrams)
FNO-latent + NCA refinement; multi-scale/dilated NCA; operator-coupled conservative NCA;
differentiable-physics-in-the-loop NCA (lit review §7). Hypothesis: combine FNO global
mixing with NCA local conservation. **To verify**, not assumed.

---

## Comparison axes the experiments isolate
| Axis | plain NCA | PI-NCA | FNO | PINN |
|---|---|---|---|---|
| Receptive field / step | local (1 cell) | local (1 cell) | global | global (continuous) |
| Conservation built-in | no | **yes (mass)** | no | only via loss |
| Params (heat, reduced) | ~6.8k | **~4.6k** | ~1.1×10⁵ | small MLP |
| Cross-IC generalisation | emulator | emulator | **operator** | none (per-IV) |
| Native BC | periodic (circular) | periodic | periodic (FFT) | soft (loss) |

Numbers above for params are measured (smoke); full mean±std in `docs/experimental_report.md`.
