# Environment & Reproducibility

## Host (this session)
- OS: Windows 11 (no admin rights).
- Python: **3.14.5** (`C:\Users\tharu\AppData\Local\Python\pythoncore-3.14-64`).
- Hardware: **CPU only** — `nvidia-smi` not present; `torch.cuda.is_available() == False`.

## Verified package stack (installed & import-checked this session)
| Package | Version | Notes |
|---|---|---|
| jax / jaxlib | 0.10.1 | CPU build. `jax.devices() == [CpuDevice(id=0)]`. |
| optax | 0.2.8 | optimizers |
| flax | 0.12.7 | `flax.nnx` and `flax.linen` both import |
| cax | 0.3.3 | Cellular Automata Accelerated in JAX (needs `pillow` for image utils) |
| torch | 2.12.0+cpu | original reference implementation only |
| numpy, matplotlib, imageio | (present) | data/plots/gifs |

## Known install gotcha (documented for reproducibility)
On this Windows host without admin rights, `pip install flax optax` **fails** while extracting
`orbax-checkpoint` because its deeply nested test directories exceed the Windows `MAX_PATH`
(260-char) limit, and `HKLM\...\FileSystem\LongPathsEnabled` cannot be set without admin.

**Workaround used:**
```
python -m pip install "jax[cpu]"
python -m pip install optax msgpack tensorstore rich PyYAML typing_extensions
python -m pip install --no-deps flax          # Flax imports orbax lazily; modeling (nnx/linen) needs no orbax
python -m pip install --no-deps cax            # deps (flax/jax/optax) already present
python -m pip install pillow                   # for cax image utilities
```
On a machine with admin / long-paths enabled (or Linux/macOS), plain
`pip install -r requirements-jax.txt` works without the `--no-deps` dance.

## GPU host (final benchmark run)
- Target: single **NVIDIA RTX 4090** (24 GB), Linux, headless.
- Install `requirements-gpu.txt` instead of `requirements-jax.txt` — the only difference is
  `jax[cuda12]==0.10.1` in place of `jax[cpu]==0.10.1`. The CUDA/cuDNN runtime ships inside the
  wheel; only an NVIDIA driver new enough for CUDA 12 is required (>= 525), no system toolkit.
- `pip install -e .` (pyproject, src layout) so `python -m pinca_jax.<driver>` resolves from any
  working directory. `PYTHONPATH=src` is the no-install fallback.
- `run_gpu.sh` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so XLA does not grab ~90% of VRAM up
  front, and `--xla_gpu_deterministic_ops=true` for repeatable reductions.
- Scale knobs used for the full run (2-D): `--grid 64 --batch 64 --epochs 2000 --seeds 3`;
  3-D: `--grid 32 --epochs 800 --batch 16`. Identical code path to the CPU presets.
- Step-by-step: `docs/gpu_runbook.md`.

## Reproducibility policy
- **PRNG:** every experiment takes an explicit integer `seed`; JAX `jax.random.PRNGKey(seed)` is
  threaded functionally (no global RNG state). Multiple seeds → mean ± std (never single-run).
- **Configs:** every run is fully specified by a dataclass config (grid size, steps, channels,
  lr, epochs, seed). Reduced-scale CPU configs and full-scale GPU configs differ **only** in
  numeric fields, so the same code reproduces both.
- **Determinism caveat:** XLA CPU reductions are not bitwise-deterministic across thread counts;
  correctness is asserted to tolerances, not bit-equality.
- **Migration correctness gate:** JAX ports are validated against the PyTorch reference
  (`tests/`/correctness harness) before any architecture change.
