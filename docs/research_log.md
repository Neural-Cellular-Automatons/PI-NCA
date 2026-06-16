# Research Log — PI-NCA: NCA vs PINN vs Operator Learning for PDEs

Running, append-only log. Newest entries at the bottom of each day. Times are the session's working dates.

---

## 2026-06-16 — Day 0: Scoping, environment, foundation

### Ground truth established (no assumptions)
- **Repository = PyTorch.** Branches and contents (verified via `git ls-tree`):
  - `main`: `PI NCA_v1.py` — a fused Physics-Informed NCA for the 2-D heat equation. Key design: `DeepFluxNCA` predicts a **flux** `(f_x,f_y)` and applies a **discrete divergence** update (finite-volume / conservation form); `HeatEquationSolver` is a differentiable 5-point-Laplacian teacher with periodic (circular) padding; `conserve_energy` projects total mass. Trains by truncated BPTT against the solver, AMP, `torch.compile`.
  - `Heterogenous-simulations`: + `heat-simulation-for-heterogenity.ipynb`, + reference paper `2407.06151v2.pdf`.
  - `PI-NCA-Gray-Heat-Equation`: + `PINCA Heat_Gray_Scott.ipynb` (Gray–Scott reaction–diffusion + heat).
  - `new-update-of-physics-informed-models`: + `PINCA_v3plus_SWE_FHN_CH_PSNR.ipynb` (Shallow-Water, FitzHugh–Nagumo, Cahn–Hilliard, PSNR metrics) + `pinca_v3_outputs.zip` (79 MB outputs).
- **Compute reality:** Python **3.14.5**, `torch 2.12.0+cpu`, **no GPU** (`nvidia-smi` absent, `torch.cuda.is_available()=False`).
- **JAX feasibility:** verified by installing — `jax 0.10.1` (CpuDevice), `optax 0.2.8`, `flax 0.12.7` (`nnx`+`linen` import OK), `cax 0.3.3`. Flax needed a `--no-deps` workaround because `orbax`'s deeply nested test dirs exceed Windows `MAX_PATH` (no admin to set `LongPathsEnabled`). Recorded in `docs/environment.md`.

### Decisions (with user)
- **Compute scope:** *CPU-feasible reduced scale* — run the full pipeline (PINN, NCA, PI-NCA, FNO, hybrids) end-to-end at deliberately small grids/steps/seeds as a rigorous, reproducible methodology demonstration; configs written to re-run unchanged on GPU. Absolute numbers documented as reduced-scale.
- **Cadence:** *Run mostly autonomously* through phases; commit per-branch; maintain this log; stop only for genuine blockers.
- **Branching model:** foundational work (scaffolding, lit review, shared JAX core, migration + correctness) on `research/jax-migration`; baseline/hybrid branches fork from it to inherit the migrated core; `research/final-comparison` merges results.

### Implications of CPU-only (documented honestly)
- `pmap` / multi-device `sharding` are **no-ops on a single CPU host** — they will be implemented as code paths and documented, but provide no speedup here. `jit`, `vmap`, `lax.scan`, and `jax.checkpoint` *do* help on CPU and are the focus.
- Full reproduction of the 89 MB SWE/FHN/CH notebook at original scale is impractical in-session; we reduce scale and document the gap.

### Work done today
1. Mapped all branches & files; read `PI NCA_v1.py` in full.
2. Installed and verified the JAX/Flax/Optax/CAX stack on Python 3.14 CPU.
3. Created branch `research/jax-migration`.
4. **Literature review** drafted — `docs/literature_review.md`. Real citations gathered via paper-discovery across NCA, PINN, PINCA, FNO, DeepONet, GNO, operator learning, differentiable solvers, JAX SciML. Includes per-family math formulation, computational & memory complexity, strengths/weaknesses, and a falsifiable predictions table.
5. Reproducibility scaffolding — `docs/environment.md`, `requirements-jax.txt`, `README_RESEARCH.md`.

### Open questions / next steps
- [ ] **Phase 1 — Migration:** port `PI NCA_v1.py` (heat PI-NCA + solver) to JAX/Flax in `src/pinca_jax/`, with a **numerical-correctness harness** comparing the JAX `HeatEquationSolver` step against the PyTorch one to tight tolerance (this is the gate before any architecture changes).
- [ ] Read reference paper `2407.06151` (heterogeneity branch) and extract the heterogeneity setup.
- [ ] Per-branch migration docs under `docs/migration/`.
- [ ] Define the shared equation suite + reduced-scale configs (`src/pinca_jax/equations/`).
- [ ] Then baselines: PINN, NCA, PI-NCA, FNO; then hybrids; then ablations; then `research/final-comparison`.

### Update — Phase 1 (heat/`main` branch migration) COMPLETE
- Built shared JAX core `src/pinca_jax/`: `equations/heat.py` (roll-based periodic
  Laplacian + `lax.scan` rollouts), `models/flux_nca.py` (Flax linen DeepFluxNCA,
  NHWC, circular padding, zero-init flux head), `physics.py` (divergence update +
  energy projection), `data.py` (vectorised periodic-blob ICs), `train_nca.py`
  (jit/optax/scan trainer), `configs.py` (SMOKE/CPU_REDUCED/GPU_FULL presets).
- **Correctness gate: `pytest tests/` → 7/7 pass.** JAX Laplacian == PyTorch circular
  conv; single-step + 25-step rollout match; weight-ported NCA matches with non-zero
  flux head (divergence path exercised, atol 1e-5); mass conservation + energy
  projection verified; end-to-end training reduces loss.
- **End-to-end:** `train_nca --smoke` → loss 1.11e-1→7.22e-3 (15.4×) in 2.0s CPU.
- Migration docs: `docs/migration/README.md`, `docs/migration/01_heat_pinca_main.md`
  (port table, correctness evidence, intentional behavioural diffs, hardware honesty).
- **Gate satisfied → architecture work on the heat branch is now unblocked.**
- Next: migrate the three notebook branches' equations (heterogeneous heat, Gray–Scott,
  SWE/FHN/Cahn–Hilliard) into `equations/`, then fork baseline branches.

### Update — Phase 2 (8-PDE suite migration) + a correctness bug found & fixed
- **Bug found (honest record):** the Phase-1 `heat.laplacian_periodic` differenced the
  *last two axes* (`-1,-2`). That is correct for NCHW (what the unit test fed) but WRONG
  for the NHWC `(B,H,W,1)` arrays the training pipeline actually uses — there axis `-1` is
  the size-1 channel, so the teacher rollout during training was a **degenerate 1-D
  Laplacian**. The unit test passed only because it used NCHW; the convention mismatch hid
  the defect. This is exactly the failure mode correctness gates exist to catch — mine had
  a convention gap with the real code path.
- **Fix:** introduced `equations/operators.py` (NHWC, spatial axes (1,2): `laplacian`,
  `grad_x`, `grad_y`); `heat.py` now aliases the NHWC laplacian. Standardised the whole
  codebase on NHWC. Added a **2-D isotropy regression test** (Laplacian of a radially
  symmetric bump must be x↔y symmetric and concave at the peak) that fails on the old
  degenerate operator. Re-ran heat training: converges correctly on the true isotropic
  teacher (loss 2.72e-1→1.47e-2, 18.5×; absolute loss higher than before precisely because
  2-D diffusion is the harder, correct target).
- **8-PDE suite migrated** (`equations/pdes.py`): heat, wave, advection-diffusion,
  Allen-Cahn, Gray-Scott, shallow-water (RK4), Cahn-Hilliard, FitzHugh-Nagumo — each a pure
  NHWC `step(state, params)` with a `PDESpec` registry (channels, params, conserves_mass)
  and `lax.scan` rollouts. Full bodies/params extracted verbatim from the notebook
  (no assumptions; Wave u-update and SWE RK4 re-extracted in full to avoid guessing).
- **Correctness gate now 25/25** (`tests/`): every PDE's single step AND 10-step rollout
  matches a verbatim PyTorch reference (atol 1e-5 step / 1e-4 rollout); isotropy test;
  NHWC-corrected heat tests; NCA weight-port; conservation; training smoke.
- Next: fork baseline branches from `research/jax-migration` (PINN, NCA, PI-NCA, FNO);
  build shared IC generators + metrics; then hybrids and ablations.

### Update — Shared evaluation infrastructure + a stability finding
- **`ic.py`** — vectorised JAX initial-condition generators for all 8 PDEs, ported from
  the notebook `make_state`/`make_gaussian_blobs` (gray_scott seed-boxes vectorised at
  fixed count; FHN IC flagged as standard since its source wasn't captured).
- **`metrics.py`** — full mandated metric set: MSE/RMSE/MAE, relative-L2, PSNR,
  one-step residual, mass-conservation error, periodic-BC residual, gradient-energy
  (stability proxy), parameter count, wall-clock timer, and multi-seed `Agg`/`aggregate`
  (mean ± std — so we never report single runs).
- **Finding (honest record): Gray-Scott `dt=2.0` is numerically unstable.** The verbatim
  notebook param exceeds the explicit-Euler diffusion stability limit
  `dt ≤ dx²/(4·Du) = 1.25`; on sharp box ICs the solver diverges by ~step 12
  (verified: max|u| 1→9.3→inf). Kept the faithful `dt=2.0` in `REGISTRY` (migration
  fidelity) but added `pdes.STABLE["gray_scott"]` (dt=1.0) and `override_params(...)` for
  experiments. Locked in via `test_gray_scott_dt2_is_unstable_finding`. This is a
  candidate regime where *all* learned emulators inherit teacher instability — to revisit
  when comparing architectures on stiff dynamics.
- Gate now **36/36**. Shared infra ready → baseline architecture branches next.

### Update — Phase 3: emulator harness + first multi-seed benchmark (heat)
- Built model-agnostic emulator harness (`harness.py`, multi-seed mean±std),
  `models/nca.py` (plain NCA), `models/fno.py` (FNO2d), `models/registry.py`, `bench.py`.
- **Heat, 3 seeds (grid 24, train 12 / eval 48 steps, 200 ep):**
  - fno: rel-L2 **3.52e-2 ± 3.5e-3** (best + low variance), consErr 21.3, **592,897 params**, 4.3e-3 s/step
  - pi_nca: rel-L2 4.38e-2 ± 3.5e-2, **consErr 3.8e-4**, **4,576 params**, **8.4e-4 s/step**
  - plain_nca: rel-L2 0.234 ± 0.28 (unstable; ≥1 seed diverged), consErr 33.3
- **Findings:** H1 confirmed (conservation structure → 5 orders better mass error; removing it
  destabilises the NCA). On local diffusion FNO wins accuracy+stability but at 130× params /
  5× slower inference and no conservation; PI-NCA is the efficiency/conservation winner,
  accuracy-competitive but higher seed variance (needs pool/longer training). Honest split,
  not an NCA win. FNO param bloat → iso-parameter ablation queued.
- Next: Cahn-Hilliard (H2: global coupling should favour FNO), then SWE/FHN multi-channel
  (plain_nca + fno only; build multi-channel conservative NCA), PINN track, hybrids, ablations.

### Update — Phase 4: Cahn-Hilliard failure, PINN, and first ablation
- **CH benchmark:** all emulators diverge (rel-L2 14-18 vs identity floor 0.93; FNO
  catastrophic variance; pi_nca conserves mass while least accurate). H2 NOT supported.
  Honest negative result → docs/experimental_report.md.
- **PINN (heat):** implemented continuous single-IVP PINN (`pinn_heat.py`), Fourier-feature
  periodic BCs, nested-autodiff residual; validated (loss 36→0.23 in 300 it). Branch
  `research/baseline-pinn`.
- **Ablation 1 (output bounding, CH):** clipping emulator outputs to [-1,1] (solver's range)
  gives **24-27× improvement** — plain_nca 12.9→0.54, pi_nca 16.5→0.60, both **below the
  0.93 floor**. CH failure was output blow-up, architecturally fixable. **New tension:** hard
  clip destroys pi_nca's exact conservation (3e-5→7.6). Once bounded, conservation no longer
  the accuracy differentiator (regime-dependent). → docs/ablation_report.md (A2-A6 queued).
- **Key cross-cutting result:** *different regimes reward different components* — conservation
  wins on smooth local heat; bounding wins on stiff CH. No single architecture dominates.
- Deliverables now drafted: lit review, migration (4 reports), architecture, experimental,
  ablation, reproducibility, research log. Remaining: multi-channel NCA + SWE/FHN runs,
  hybrids, full multi-seed PINN sweep, final paper-style summary.
