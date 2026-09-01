# GPU Runbook — final benchmark run on an RTX 4090 (headless, no IDE)

Everything here is terminal-only. Nothing needs a display, a notebook, or an editor.
All output lands in `results/` (tables + JSON) and `docs/figures/bench/` (plots).

---

## 0. Prerequisites on the GPU box

```bash
nvidia-smi                 # driver >= 525 for CUDA 12; note the VRAM figure
python3 --version          # 3.10 - 3.13 (3.14 works too; jax wheels exist for 3.13+)
```

JAX ships its own CUDA/cuDNN in the `jax[cuda12]` wheel — you do **not** need a system
CUDA toolkit, only an NVIDIA driver new enough for CUDA 12.

## 1. Get the code and the branch

```bash
git clone https://github.com/Neural-Cellular-Automatons/PI-NCA.git
cd PI-NCA
git checkout feature/final-benchmark-run
```

## 2. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
pip install -e .                 # puts src/pinca_jax on the path — needed for `python -m pinca_jax.*`
```

Confirm the GPU is actually the backend before spending hours on it:

```bash
python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"
```

Expected: `0.10.1 gpu [CudaDevice(id=0)]`. If it prints `cpu`, the CUDA wheel did not
install — re-run the pip step and read the error; do not proceed.

## 3. Correctness gate (always first)

```bash
python -m pytest tests/ -q
```

56 tests. They assert the JAX ports match the PyTorch reference to tolerance. Red gate ⇒
the benchmarks below are meaningless.

## 4. The run

```bash
bash run_gpu.sh smoke      # ~2 min, CPU-scale numbers, proves every stage wires up
bash run_gpu.sh            # the real thing
```

Long run over SSH — keep it alive and logged:

```bash
tmux new -s bench
bash run_gpu.sh 2>&1 | tee run_gpu.log
# detach: Ctrl-b then d      reattach: tmux attach -t bench
```

No tmux available:

```bash
nohup bash run_gpu.sh > run_gpu.log 2>&1 &
tail -f run_gpu.log
```

Watch the GPU from a second shell: `watch -n2 nvidia-smi`.

### What `run_gpu.sh` does

| Stage | Command it runs | Writes |
|---|---|---|
| 1 | `pytest tests/ -q` | — |
| 2 | `pinca_jax.bench_all --group all` | `results/bench_<pde>_full.{md,json}`, `bench_*_A4/A5.*` |
| 3 | `pinca_jax.bench3d` | `results/bench3d_<pde>.{md,json}` |
| 4 | `pinca_jax.res_study` | `results/bench_resolution_<pde>.{md,json}` |
| 5 | `pinn_heat`, `deeponet_heat`, `darcy` | console + `results/` |
| 6 | `viz`, `viz3d`, `viz3d_volume` | `docs/figures/*.png`, `results/gifs/*.gif` |
| 7 | `pinca_jax.plots` | `docs/figures/bench/*.png` |

### Full-scale vs smoke settings

| knob | smoke | full (GPU) | why |
|---|---|---|---|
| seeds | 1 (fixed 42) | 3 | GPU makes real mean±std affordable |
| epochs (2-D) | 60 | 2000 | CPU runs were truncated at 150 |
| grid (2-D) | 24 | 64 | the scale the paper claims |
| batch | 16 | 64 | 4090 VRAM is not the constraint here |
| grid (3-D) | 16³ | 32³ | 8× the cells of the CPU run |

Only numeric fields change — the code path is identical, which is the whole point of
`docs/reproducibility.md` §6.

## 5. Running pieces by hand

```bash
# one phenomenon, all applicable architectures
python -m pinca_jax.bench --pde heat --seeds 3 --epochs 2000 --grid 64 --batch 64 \
                          --rollout 12 --eval 48

# one group only
python -m pinca_jax.bench_all --group local        --seeds 3 --epochs 2000 --grid 64 --batch 64
python -m pinca_jax.bench_all --group multichannel --seeds 3 --epochs 2000 --grid 64 --batch 64
python -m pinca_jax.bench_all --group special      --seeds 3 --epochs 2000 --grid 64 --batch 64
python -m pinca_jax.bench_all --group ablation     --seeds 3 --epochs 2000 --grid 64 --batch 64

# 3-D
python -m pinca_jax.bench3d --grid 32 --epochs 800 --batch 16

# replot from whatever JSON is already in results/ (no training, seconds)
python -m pinca_jax.plots
```

`python -m pinca_jax.plots` is safe to run at any time — it only reads `results/*.json`.

## 6. Reading the output

```bash
cat results/bench_heat_full.md          # per-phenomenon table, winner per metric bolded
ls docs/figures/bench/                  # every benchmark plot
```

| Figure | Shows |
|---|---|
| `bench_accuracy_2d.png` | rel-L2 @T, every architecture, every 2-D phenomenon |
| `bench_psnr_2d.png` | PSNR (dB) |
| `bench_conservation_2d.png` | mass-conservation drift — where PI-NCA's flux head earns its keep |
| `bench_error_growth.png` | rel-L2 at T/4 → T: who degrades over a long rollout |
| `bench_accuracy_vs_cost.png` | params vs error, the Pareto view |
| `bench_train_time.png`, `bench_throughput.png` | wall-clock cost and inference speed |
| `bench_regime_map.png` | normalised rel-L2 — the "no universal winner" claim in one image |
| `bench_accuracy_3d.png` | the 3-D suite |
| `bench_ablation_A4.png`, `bench_ablation_A5.png` | conservation on/off, perception size |
| `bench_resolution_*.png` | train-grid × eval-grid zero-shot transfer |

Copy them off the box with `scp -r user@box:PI-NCA/docs/figures/bench ./`.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `jax.default_backend() == 'cpu'` | CPU wheel installed | `pip uninstall -y jax jaxlib && pip install -r requirements-gpu.txt` |
| `ModuleNotFoundError: pinca_jax` | src layout not on path | `pip install -e .` (or `export PYTHONPATH=src`) |
| `RESOURCE_EXHAUSTED` / OOM | batch or grid too large | lower `--batch` first, then `--grid` |
| GPU sits at ~0% util | grid too small — kernels finish faster than they launch | raise `--grid`/`--batch`; small grids are launch-bound, not a bug |
| `CUDA_ERROR_NO_DEVICE` inside tmux | stale session predates the driver | start a fresh tmux session |
| run dies on SSH disconnect | no tmux/nohup | see §4 |

Timings from the run land in the tables as `train wall(s)` and `infer s/step`, and the
device stamp (`jax` version, backend, device list, peak VRAM) is written into every
`results/*.json` under `"device"` — so a GPU run is self-identifying after the fact.
