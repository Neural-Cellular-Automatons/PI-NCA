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
