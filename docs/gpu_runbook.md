# GPU Runbook — the benchmark run on an RTX 4090 (headless, no IDE)

Terminal only. Nothing here needs a display, a notebook, or an editor. Output lands in
`results/` (tables + JSON), `docs/figures/` (plots and figures), and
`docs/PI-NCA_Architectures_and_Results.pdf` (the report, regenerated from the results).

---

## The one command

```bash
bash run_gpu.sh
```

That is the whole run: correctness gate → uniform 2-D matrix → ablations → uniform 3-D
matrix → resolution study → baselines → trajectory capture → field figures → plots →
report. It is resumable, tolerates a model running out of memory, and refuses to
produce CPU numbers.

Everything below is context for when something goes wrong.

---

## Windows hosts: use WSL2, not cmd.exe

JAX has **no native-Windows GPU support**. Per the official install matrix:

| Platform | NVIDIA GPU |
|---|---|
| Linux, x86_64 | yes |
| Linux, aarch64 | yes |
| Windows, x86_64 | **no** |
| Windows WSL2, x86_64 | experimental (works in practice with CUDA 12 + a recent driver) |

The wheel index makes it concrete: `jaxlib` publishes a `win_amd64` wheel, but
`jax-cuda12-plugin` is **manylinux only**. There is no Windows CUDA backend to install,
so `pip install jax[cuda12]` under Windows silently leaves you on CPU whatever card is
in the box.

One-time setup, from an **Administrator** cmd.exe:

```bat
wsl --install -d Ubuntu
```

Reboot if asked, then set the Ubuntu username/password. The Windows NVIDIA driver
already provides GPU passthrough — do **not** install an NVIDIA driver inside WSL.

Every session after that, from ordinary cmd.exe or PowerShell:

```bat
wsl
```

You are now in Linux. Check the card is visible:

```bash
nvidia-smi
```

### Shell check

`source` is a bash builtin. If `source .venv/bin/activate` says *not recognised*, you
are still in PowerShell or cmd, not WSL.

| Shell | Activate |
|---|---|
| WSL / Linux | `source .venv/bin/activate` |
| cmd.exe | `.venv\Scripts\activate.bat` |
| PowerShell | `.\.venv\Scripts\Activate.ps1` |

`nvidia-smi` working in PowerShell proves only that the **driver** sees the card. It
says nothing about whether JAX can use it — on Windows it cannot.

---

## 1. Get the code

```bash
git clone -b feature/final-benchmark-run https://github.com/Neural-Cellular-Automatons/PI-NCA.git ~/PI-NCA
cd ~/PI-NCA
```

Clone into `~`, not `/mnt/c`. The Windows filesystem through WSL's translation layer is
slow for the many small reads a benchmark does.

## 2. Environment

**Python 3.12, 3.13 or 3.14** — all three work. `jax`, `jaxlib` and `jax-cuda12-plugin`
publish cp312/cp313/cp314 manylinux wheels, and the CUDA runtime ships *inside* the
wheel, so you need only an NVIDIA driver new enough for CUDA 12 (>= 525). No system
CUDA toolkit.

Ubuntu ships `python3` without the `venv` module, so:

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip git
```

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-gpu.txt
pip install -e .
```

`torch` is installed from the CPU index deliberately: it is only the reference
implementation for the correctness gate, never used in the benchmarks, and the default
build would pull ~3 GB of CUDA libraries duplicating what the jax wheel already ships.

`pip install -e .` is not optional — without it `python -m pinca_jax.*` cannot resolve
the `src/` layout.

Confirm before spending hours:

```bash
python -c "import jax; print(jax.__version__, jax.default_backend(), jax.devices())"
```

Expected: `0.11.1 gpu [CudaDevice(id=0)]`. If it says `cpu`, stop — the CUDA wheel did
not install, and the benchmark drivers will refuse to run anyway (see below).

## 3. Run it

```bash
bash run_gpu.sh
```

Over SSH, keep it alive and logged:

```bash
tmux new -s bench
```

```bash
bash run_gpu.sh 2>&1 | tee run_gpu.log
```

Detach with `Ctrl-b` then `d`; reattach with `tmux attach -t bench`. Watch the card from
a second shell with `watch -n2 nvidia-smi`.

### Scale presets

```bash
bash run_gpu.sh --profile smoke   # ~2 min, tiny scale, proves every stage wires up
bash run_gpu.sh --profile bench   # measurements + plots, no field figures
bash run_gpu.sh                   # full (default)
```

| Knob | smoke | bench / full |
|---|---|---|
| seeds | 1 | 3 |
| epochs (2-D) | 40 | 2000 |
| grid (2-D) | 16 | 64 |
| batch | 8 | 64 |
| grid (3-D) | 8³ | 32³ |
| epochs (3-D) | 20 | 800 |

Run `--profile smoke` once first and time it. The full run scales roughly with
`epochs × grid² × seeds`, so your own smoke time is a far better predictor than any
estimate here.

### Other flags

Anything you pass is forwarded to the runner:

```bash
bash run_gpu.sh --only bench2d,plots      # just these stages
bash run_gpu.sh --skip figures,baselines  # everything except these
bash run_gpu.sh --force                   # recompute cells already on disk
bash run_gpu.sh --no-gate                 # skip the test suite
DETERMINISTIC=1 bash run_gpu.sh           # deterministic XLA ops (slower)
```

---

## 4. What the run guarantees

### It will not silently give you CPU numbers

Every benchmark driver calls `env.require_gpu()` and **exits** on the CPU backend. Half
a matrix measured on CPU and half on GPU is worse than no matrix: throughput, latency
and achievable scale all differ. The escape hatch is `--allow-cpu`, which prints a
warning that the numbers are not comparable, and exists for development only.

### One model running out of memory will not end the run

Each `(phenomenon, architecture)` cell is retried at **half the batch**, repeatedly, down
to a floor. The batch actually used is recorded in the results as `_batch_used`, so you
can see which cells were reduced. If even the floor fails, the cell is recorded as
failed and the sweep continues.

The distinction matters: a genuine bug — a shape error, a NaN — is **not** retried. It
surfaces immediately instead of wasting five attempts.

### A crash costs one model, not a night

Results are written after **every cell**, atomically (temp file + rename, so a crash
mid-write cannot corrupt a resumable file). Re-running skips completed cells:

```bash
bash run_gpu.sh          # crashed after 6 hours? just run it again
```

It picks up where it stopped. Add `--force` only if you want to recompute.

### Memory is not preallocated

`XLA_PYTHON_CLIENT_PREALLOCATE=false` and the platform allocator are set by the wrapper
*and* by the Python entry point, so both paths behave. Without this XLA grabs ~90% of
VRAM up front, which makes every later allocation failure look like a hard OOM and
leaves no headroom for `nvidia-smi`.

JAX's compilation caches are cleared between cells, so peak memory tracks the largest
single model rather than the whole sweep.

---

## 5. The uniform matrix

Every architecture now runs on **every** phenomenon — the same list of competitors in
every table.

Previously the flux-form models hardcoded a 2-channel flux head, so they only worked on
single-channel fields. Multi-field phenomena (wave, Gray–Scott, shallow-water,
FitzHugh–Nagumo) were therefore measured with three models while scalar ones got five,
and the tables could not be compared row to row.

All models are now generic in the channel count: they emit one `(f_x, f_y)` pair per
field and apply a per-channel divergence, so each field's mass is conserved separately.
The bounded variants take the PDE's **measured physical range** rather than a hardcoded
`[-1, 1]` — clipping heat, whose amplitudes run 5–10, to `[-1, 1]` would have destroyed
the field.

At C = 1 the numerics are unchanged, so every previously published number still stands.
`tests/test_uniform_matrix.py` asserts this, and the PyTorch migration-correctness gate
still passes.

| Stage | Command | Writes | On failure |
|---|---|---|---|
| gate | `pytest tests/ -q` | — | fatal |
| bench2d | `pinca_jax.bench_all --group all` | `results/bench_<pde>_full.{md,json}` | fatal |
| bench3d | `pinca_jax.bench3d` | `results/bench3d_<pde>.{md,json}` | fatal |
| resolution | `pinca_jax.res_study` | `results/bench_resolution_<pde>.*` | continue |
| plots | `pinca_jax.plots` | `docs/figures/bench/*.png` | continue |
| baselines | `pinn_heat`, `deeponet_heat`, `darcy` | console + `results/` | continue |
| capture | `pinca_jax.capture` | `results/traj/*.npz` | continue |
| figures | `viz`, `viz3d`, `viz3d_volume` (all `--npz`) | `docs/figures/*.png` | continue |
| report | `arch_figs`, `report`, `md2pdf` | `docs/PI-NCA_Architectures_and_Results.{md,pdf}` | continue |

Only the measurement stages are fatal. Everything downstream is recorded and skipped.

---

## 6. Reading the output

```bash
cat results/bench_heat_full.md      # one phenomenon, every model, winner bolded
ls docs/figures/bench/              # 14 benchmark plots
cat results/run_manifest.json       # per-stage timings, failures, device stamp
```

The report regenerates itself from the results, so its tables cannot disagree with the
data:

```bash
python -m pinca_jax.report && python -m pinca_jax.md2pdf docs/PI-NCA_Architectures_and_Results.md
```

Every results file carries a `"device"` stamp (`jax` version, backend, device list, peak
memory), so a GPU run is self-identifying afterwards.

### Capture once, render forever

`pinca_jax.capture` trains one model per phenomenon and archives the raw solver/model
trajectories to `results/traj/*.npz` (2-D `(T+1,H,W)`, 3-D volumes `(T+1,D,H,W)`). Every
figure is rendered from those files, so figures can be rebuilt or restyled later with no
GPU and no retraining:

```bash
python -m pinca_jax.viz3d_volume --npz results/traj/heat_3d.npz
```

`np.load(path)["model"]` is the whole interface. `--max-mb` (default 64 per phenomenon)
strides the time axis to bound file size.

### Copy results off the box

```bash
cp -r ~/PI-NCA/docs/figures ~/PI-NCA/results /mnt/c/Users/<you>/Desktop/pinca_results
```

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NotOnGPU: refusing to benchmark on the 'cpu' backend` | working as intended | install `requirements-gpu.txt`; on Windows, move to WSL2 |
| backend is `cpu` however you install | native Windows has no JAX CUDA wheels | use WSL2 |
| `ModuleNotFoundError: pinca_jax` | src layout not installed | `pip install -e .` |
| a cell reports `FAILED (oom)` | model too large even at the batch floor | lower `--profile` scale, or rerun that cell alone with a smaller `--grid` |
| several cells OOM | grid/batch too large for the card | `bash run_gpu.sh --only bench2d --force` after lowering the profile |
| `nvidia-smi` empty inside WSL2 | Windows driver too old, or a driver was installed *inside* WSL | update the Windows driver; never install one in WSL |
| GPU sits near 0% util | small grids are launch-bound, not a bug | raise `--profile`; util is also low during pytest and plotting |
| run died overnight | — | just run `bash run_gpu.sh` again; finished cells are skipped |
| `orbax` path error on Windows install | MAX_PATH limit | see the note at the top of `requirements-jax.txt` |
