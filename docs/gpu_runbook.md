# GPU Runbook — final benchmark run on an RTX 4090 (headless, no IDE)

Everything here is terminal-only. Nothing needs a display, a notebook, or an editor.
All output lands in `results/` (tables + JSON) and `docs/figures/bench/` (plots).

---

## Windows hosts: use WSL2, not cmd.exe

JAX has **no native-Windows GPU support**. Per the official install matrix
(docs.jax.dev/en/latest/installation.html):

| Platform | NVIDIA GPU |
|---|---|
| Linux, x86_64 | yes |
| Linux, aarch64 | yes |
| Windows, x86_64 | **no** |
| Windows WSL2, x86_64 | experimental (works in practice, CUDA 12 + recent driver) |

Install `jax[cuda12]` in native Windows and you get the CPU backend no matter what card is
in the box. So on a Windows 4090 machine, run the benchmark inside WSL2.

One-time setup, from an **Administrator** cmd.exe:

```bat
wsl --install -d Ubuntu
```

Reboot when it asks, then set the Ubuntu username/password it prompts for. The Windows NVIDIA
driver already provides GPU passthrough — do **not** install an NVIDIA driver inside WSL.

Every session after that, from ordinary cmd.exe:

```bat
wsl
```

You are now in Linux. Your Windows drives are mounted under `/mnt/c`, so:

```bash
cd /mnt/c/Users/<you>/PI-NCA
nvidia-smi          # must list the 4090; if not, update the Windows driver
```

Then follow §1 onward exactly as written. Clone into the WSL filesystem
(`~/PI-NCA`) rather than `/mnt/c` if you care about I/O speed — `/mnt/c` is slow.

`run_gpu.bat` exists for running the suite from cmd.exe, but it can only ever
produce **CPU** numbers, and it says so before it starts. Use it for
`python -m pinca_jax.plots` and for reduced-scale sanity checks, not the final run.

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

One command. It picks an interpreter, builds `.venv`, installs the pinned stack, and
refuses to report success unless the GPU backend is actually live:

```bash
bash setup_gpu.sh
```

```bash
source .venv/bin/activate        # every new shell afterwards
```

There is deliberately **no committed virtualenv** — a venv bakes in absolute paths, ships
platform-specific binaries, and the CUDA wheels run to several GB. `setup_gpu.sh` is the
portable equivalent.

<details><summary>What it does, if you prefer to run the steps by hand</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu   # reference impl only
pip install -r requirements-gpu.txt
pip install -e .                 # puts src/pinca_jax on the path — needed for `python -m pinca_jax.*`
python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"
```

Expected: `0.10.1 gpu [CudaDevice(id=0)]`. If it prints `cpu`, the CUDA wheel did not
install — read the pip error; do not proceed.
</details>

**Python version:** the pins need **>= 3.12**; **3.12, 3.13 and 3.14 all work**, and
`setup_gpu.sh` prefers the newest it finds. Verified against PyPI: jax 0.10.1 `>=3.12`,
cax 0.3.3 `>=3.11`, and `jaxlib` / `jax-cuda12-plugin` / `torch` 2.12.0 all publish cp312,
cp313 and cp314 manylinux x86_64 wheels. flax/optax are pure Python.

Incidentally the wheel index is where the Windows limitation becomes concrete: `jaxlib`
publishes a `win_amd64` wheel, but `jax-cuda12-plugin` publishes **manylinux only**. There is
no Windows CUDA backend to install.

Ubuntu ships `python3` without the `venv` module. `setup_gpu.sh` detects that and installs
`python3.<N>-venv` for you (it will ask for your sudo password). To do it yourself first:

```bash
sudo apt update && sudo apt install -y python3.13 python3.13-venv python3-pip git
```

CPU-only box (plots and sanity checks, not the final run):

```bash
bash setup_gpu.sh cpu
```

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
| Windows: backend is `cpu` however you install | native Windows has no JAX GPU wheels | run inside WSL2 — see the Windows section above |
| `nvidia-smi` empty inside WSL2 | Windows driver too old, or a driver was installed *inside* WSL | update the Windows NVIDIA driver; never install one in WSL |

Timings from the run land in the tables as `train wall(s)` and `infer s/step`, and the
device stamp (`jax` version, backend, device list, peak VRAM) is written into every
`results/*.json` under `"device"` — so a GPU run is self-identifying after the fact.
