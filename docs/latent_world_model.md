# A JEPA-FNO latent world model in this benchmark: what fits, what does not

The proposal: encode the field, evolve the *latent* with an FNO, train by matching a
stop-gradient EMA encoding of the future field, with no field-space reconstruction in the
loss. This document records what was built, how it was made comparable with the rest of
the study, and — the part worth reading — which of the proposal's stated advantages this
project's own measurements already contradict.

Implementation: [`models/latent_fno.py`](../src/pinca_jax/models/latent_fno.py) (the
architecture), [`jepa.py`](../src/pinca_jax/jepa.py) (the objective, the probe protocol,
the collapse diagnostics, the study), `tests/test_latent_jepa.py` (19 tests).

---

## 1. The design does not fit the emulator interface, and the fix matters

Every accuracy number in this study is a relative $L^2$ **in field space** against the
reference solver, produced by a one-step map `s_t -> s_{t+1}` driven autoregressively.
A JEPA model as specified has **no decoder at all**: its output lives in a latent space
the model chose for itself, and its loss is a distance in that space.

That is not a minor interface mismatch. A latent MSE and a field relative $L^2$ are not
the same quantity, and a model can drive the former to zero by choosing a degenerate
space. Reporting one next to the other would be exactly the incommensurate-baseline error
this project spent considerable effort removing from the PINN comparison.

**Resolution.** A JEPA-pretrained encoder and predictor are never scored by their own
objective. They are frozen, a decoder is fitted on top by field-space rollout MSE
(`fit_decoder`), and the resulting model is evaluated through the same harness, the same
paired statistics, and against the same identity floor as everything else. That is a
linear-probe protocol, and it is what makes the comparison legitimate.

The study therefore compares three regimes at **identical architecture and parameter
count**, differing only in how they were trained:

| regime | what it is |
|---|---|
| `distill` | end-to-end field-space distillation — the protocol every other architecture in the benchmark uses. **The control.** |
| `jepa_probe` | JEPA pretraining; encoder and predictor frozen; decoder only fitted |
| `jepa_ft` | JEPA pretraining, then the same end-to-end distillation as `distill` |
| `constant_encoder` | encoder zeroed, decoder fitted alone — **the collapse floor** |

If `jepa_ft` does not beat `distill`, the pretraining bought nothing on this task. That is
a reportable result, not a failed experiment.

---

## 2. Two of the stated advantages are contradicted by measurements already in hand

### "Spectral global receptive field — unlike localized convolutions or NCAs"

The premise is that global spectral mixing is the strong choice. This project's 4090 run
measured that directly across ten phenomena, and it is not what the data says:

- The FNO **wins** heat, advection–diffusion, and Navier–Stokes. So far so good.
- On **Cahn–Hilliard the FNO is twelfth of fourteen** ($0.650$), beaten by a
  $5{,}520$-parameter bounded conservative cellular automaton ($0.357$). Global spectral
  mixing is close to the *worst* choice on stiff fourth-order dynamics.
- A physics-free **U-Net wins three of ten** phenomena (wave, Gray–Scott, Nagumo)
  outright.
- On shallow water a purely conservative flux NCA with $5{,}280$ parameters beats the
  $592{,}995$-parameter FNO.

So "spectral global receptive field" is a regime-dependent advantage, not a general one.
A latent FNO inherits that regime dependence. It should be expected to do well where the
full-resolution FNO does well and poorly where it does not — and the study runs on all
three regime representatives so that can be checked rather than assumed.

### "Resolution independence — trained coarse, scale to fine without modification"

Two problems, one measured and one structural.

**Measured.** Resolution transfer is one of the three axes the 4090 run did *not*
produce (the stage was skipped; see `docs/claims_audit.md`, claim C10
`NOT_YET_MEASURED`). So this project currently has no GPU-scale evidence about resolution
transfer for any architecture. What it does have is the rank-stability study, and on heat
the ranking is **not** stable in grid: Kendall's $\tau=0.73$ between grid $24^2$ and
$48^2$, with the winner changing from a multi-scale conservative NCA to the FNO. Whatever
resolution independence means here, it does not mean the ordering is preserved.

**Structural.** FNO mode truncation is fixed in *index* space. Retaining modes
$0\ldots m$ on a grid of $N$ keeps physical wavenumbers up to $2\pi m/N$, so doubling $N$
at fixed $m$ halves the retained physical bandwidth. Parameter sharing across
resolutions is not the same property as invariant accuracy across resolutions. In the
latent variant this compounds: the latent grid is $N/p$, so the retained bandwidth
depends on grid *and* patch size, and `SpectralConv2d` silently clamps `modes` to what
the latent grid can supply.

---

## 3. JEPA's usual motivation is weaker here than in video

The standard argument for a latent objective is that pixel reconstruction wastes capacity
on detail that is unpredictable in principle, so predicting in representation space
focuses the model on what is predictable.

That argument is much weaker in this setting. The target is a **deterministic solver
trajectory**: there is no aleatoric noise, nothing is unpredictable in principle, and the
task itself is *defined* in field space. Distillation already supplies dense, correct,
full-resolution supervision at every step.

So the honest hypothesis is not that latent training is more accurate. It is that it might
buy one of two other things:

1. **Cheaper rollout.** The predictor runs on a $p$-fold downsampled grid, so there are
   $p^2$ fewer spatial positions per step. This is measured directly (`rollout_cost`).
2. **A representation that transfers better off-distribution.** This is the more
   interesting possibility and is left to the existing out-of-distribution harness, which
   has its own held-out axes and a translation-equivariance negative control.

Neither is an accuracy claim, and the study does not present one.

---

## 4. Collapse is measured, not assumed away

EMA targets *help* against representation collapse; they do not prevent it, and the
published results that rely on them also rely on a predictor and on normalisation layers.
A collapsed encoder emits a near-constant latent, the target matches it trivially, and the
loss looks excellent.

`collapse_metrics` therefore reports two numbers on **held-out** states:

- `latent_std` — mean per-dimension standard deviation. Near zero is total collapse.
- `eff_rank` — $\exp$(entropy of the normalised covariance spectrum) divided by $d$, in
  $[1/d, 1]$. This is the number a low loss cannot fake.

The second catches a case the first misses entirely. A representation where every sample
lies on a single line through latent space has a perfectly healthy standard deviation and
an effective rank at the floor:

| representation | `latent_std` | `eff_rank` | `top1_var_frac` |
|---|---|---|---|
| isotropic random | 1.00 | 0.947 | 0.10 |
| constant | 0.00 | 0.0625 = 1/d | 0.00 |
| **rank one** | **0.58** | **0.0625 = 1/d** | **1.00** |

Two further guards:

- **A collapse floor.** `constant_encoder` zeroes the encoder and fits the decoder alone.
  This is the JEPA analogue of this project's identity emulator: it bounds what a
  degenerate representation scores, so a probe result can be read as evidence about the
  representation rather than about the decoder's capacity.
- **`eff_rank` is read against two references, not against 1.0.** A smooth diffusive
  field genuinely has low-dimensional structure, so a rank well below 1 is expected. The
  question is whether the latent objective drove the representation *below* what plain
  distillation produces — which would be collapse attributable to the objective rather
  than to the data. The generated table states both references.

An optional VICReg-style variance hinge (`--var-weight`) is available so that the
*necessity* of an explicit anti-collapse term can be measured rather than assumed.
Default zero, which tests whether EMA alone holds.

---

## 5. Latent rollout is a different map, not an optimisation

Two rollout modes exist and they are **not the same model applied two ways**:

- **Per-step decoding** (`__call__`, applied autoregressively): encode, predict, decode
  every step. Matches the teacher at every step. This is the mode that enters the
  benchmark matrix, because it is the one that satisfies the same interface as every other
  architecture.
- **Latent rollout** (`latent_rollout`): encode once, iterate the predictor $K$ times,
  decode once. Cheaper, but it only matches the teacher at the *end* — intermediate states
  do not exist in field space, and the encoder's reconstruction error is paid once instead
  of every step.

Because `decode(encode(x))` is not the identity, these produce different trajectories.
Both accuracy columns are reported and they are never pooled. A test asserts they
actually differ with a non-zero decoder, so the distinction cannot silently evaporate.

---

## 6. What was already learned from building it

Two things, before any GPU numbers exist.

**A real bug, caught by an assertion written for exactly this.** The decoder-only probe
initially used `optax.masked`. That applies the inner transform to the selected leaves
and passes the *remaining updates through untouched* — so the "frozen" encoder received
its raw gradient and moved by gradient ascent on every step. The probe would have been
fine-tuning, and every `jepa_probe` number would have been invalid while looking
perfectly reasonable. `fit_decoder` fingerprints the frozen subtree before and after and
asserts exact equality; that assertion failed on the first run. The fix is
`optax.multi_transform` routing everything else to `set_to_zero`, and a test now pins it.

**The architecture starts as the identity, like everything else.** The decoder head is
zero-initialised, so at epoch zero the model is $g(x)=x$. Without that, a comparison
against the other fourteen architectures at low epoch counts would be measuring
initialisation rather than learning.

---

## 7. How to run it

```bash
python -m pinca_jax.jepa --pde heat --grid 48 --seeds 3
```

Or as part of the pipeline, on all three regime representatives:

```bash
bash run_paper.sh --only jepa
```

Outputs `results/jepa_<pde>.{json,md}`. The two registry entries — `latent_fno`
($\approx 6\times10^5$ parameters, comparable to the FNO) and `latent_fno_iso`
($\approx 9\times10^3$, matched to the cellular-automaton budget) — are also in the main
benchmark matrix, trained by ordinary distillation, so the architecture is measured
against the other fourteen on every phenomenon independently of the JEPA question.

---

## 8. What this cannot tell you

- Nothing here is a GPU-scale result yet. The 4090 run predates this addition; the stage
  exists and is wired into the pipeline, but `results/jepa_*.json` at the time of writing
  comes from a wiring check whose numbers are meaningless by design.
- The probe protocol measures what a *fixed-capacity decoder* can read out of the
  representation. A better decoder might read out more. That is a property of the
  protocol and is why `jepa_ft` (full fine-tuning) is reported alongside it.
- Patch-based encoding breaks the exact translation equivariance that the convolutional
  and spectral architectures have: the encoder is equivariant only to shifts that are
  multiples of the patch size. The out-of-distribution harness has a translation negative
  control that will detect this, and it should be expected to show a non-zero gap for this
  architecture where it shows none for the others.
- No claim is made about conservation. The latent architecture has no conservation
  structure, and a flux-form constraint in latent space would not correspond to mass
  conservation in field space, because the decoder is a learned map. `conservation_err`
  is still reported for it, and should be expected to be large — as it is for the FNO and
  the U-Net.
