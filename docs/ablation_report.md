# Ablation Report

Controlled ablations isolating which components drive emulator performance.
Numbers are reduced-scale CPU, mean ± std over seeds.

## Ablation 1 — Output bounding on stiff dynamics (Cahn–Hilliard)
**Question.** The CH benchmark showed all emulators *diverge* (rel-L2 14–18 ≫ the
identity floor 0.93). Diagnosis: unbounded network outputs blow up over the long
eval horizon, while the solver state is clamped to [−1,1]. **Does clipping each
emulator step to [−1,1] (the field's physical range, as the solver does) fix it?**

**Setup.** Cahn–Hilliard, grid 24, train 12 / eval 48 steps, 200 epochs, 2 seeds.
Only change: `output_clip = None` vs `(-1,1)` (harness knob, applied in train+eval).

| arch | variant | rel-L2 | conservation err |
|---|---|---|---|
| plain_nca | unbounded | 12.93 ± 4.08 | 127.3 ± 73 |
| plain_nca | **clip[-1,1]** | **0.540 ± 0.008** | 8.75 ± 3.8 |
| pi_nca | unbounded | 16.51 ± 1.20 | **3.29e-5 ± 9.6e-6** |
| pi_nca | **clip[-1,1]** | **0.600 ± 0.020** | 7.56 ± 0.84 |
| _identity floor_ | — | _0.93_ | _0_ |

**Findings.**
1. **Bounding is decisive: 24–27× improvement**, taking both NCAs from catastrophic
   divergence (12.9 / 16.5) to **below the identity floor** (0.54 / 0.60). The CH failure
   was a *stability/blow-up* problem, **architecturally fixable**, not an intrinsic limit of
   local emulation. (Confirms the divergence diagnosis from the experimental report.)
2. **Stability ↔ conservation tension (new):** hard clipping **destroys** the flux-NCA's
   exact mass conservation (3.3e-5 → 7.6) — clip is a non-conservative projection that
   overrides the flux-divergence structure. So the two desirable properties (boundedness,
   exact conservation) *conflict* under naïve clipping.
3. **Component attribution:** once clipping dominates, plain_nca (0.54) and pi_nca (0.60)
   are statistically comparable — i.e. on stiff CH the **bounding** matters more than the
   **conservation** inductive bias for accuracy, reversing the heat-equation ranking where
   conservation was the differentiator. Different regimes reward different components.

**Implication for architecture design.** The ideal stiff-PDE emulator wants *both*
boundedness *and* conservation. Naïve clip gives boundedness at the cost of conservation;
a **conservation-preserving bounded update** (e.g. clip the flux/potential rather than the
state, or project back onto the mass-constraint after clipping) is a concrete next
architecture — flagged for the hybrid phase.

## Ablation 2 — Training-horizon curriculum on stiff Cahn–Hilliard (A3)
**Question.** CH eval is 48 steps but training used 12. Does training on longer rollouts
(train→eval horizon match) reduce the stiff-PDE error, independent of the bounding fix?

**Setup.** `bounded_cons_nca` (the CH winner), grid 24, eval 48 steps, 150 epochs, 2 seeds,
`rollout_steps ∈ {12, 24, 48}`.

| train_steps | rel-L2 | conservation err | train wall (s) |
|---|---|---|---|
| 12 | 0.633 ± 0.026 | 3.16e-4 | 45.6 |
| 24 | 0.619 ± 0.022 | 2.48e-4 | 88.6 |
| **48** (= eval) | **0.511 ± 0.023** | **6.84e-5** | 176 |

**Findings.**
1. **Monotonic improvement with horizon:** matching train to eval cuts rel-L2 0.633→**0.511**
   (~19% relative) and conservation error 3.2e-4→6.8e-5 — so the train/eval-horizon gap was a
   real, *separate* contributor to the stiff-PDE error, on top of the bounding fix (A1).
2. **Cost is proportional** (BPTT depth): 4× horizon ≈ 4× training time. A practical recipe is a
   curriculum (grow horizon during training) to get most of the gain at lower cost — queued.
3. **Combined recipe** for stiff PDEs: bounded + conserving (A1) **and** train≈eval horizon (A3)
   → CH rel-L2 0.51, well below the 0.93 identity floor, with near-exact conservation (7e-5).

## Ablation A2 — iso-parameter FNO on heat (3 seeds)
Shrinking FNO to NCA budget (8.4k params) degrades it 3.4× (0.035→0.119), losing to the
NCA-budget models — FNO's heat edge was largely parameter count (see experimental_report A2).

## Ablation A4 — conservation on/off at matched backbone width (2 seeds)
Same `AblationNCA` backbone (32/64, 3×3); only the head differs: `flux` (mass-conserving
divergence) vs `residual` (free). Isolates the conservation inductive bias.

| PDE | abl_flux (conserving) | abl_residual (free) | verdict |
|---|---|---|---|
| Heat (conservative) | **0.104** | 0.353 | conservation **helps** 3.4× |
| Nagumo (non-conservative) | 0.379 | **0.123** | conservation **hurts** 3.1× |

**A4 is the clean, matched-width proof of the central rule:** the flux/conservation head helps
when the PDE conserves mass and hurts when it does not — same backbone, opposite outcome.
(Detailed 20-metric tables: `results/bench_{heat,nagumo}_A4.md`.)

## Ablation A5 — perception / receptive-field size (2 seeds)
Same head (`flux`), same widths; vary perception: 3×3 single-scale vs 5×5 vs dilated
multi-scale (1,2,4).

| PDE | 3×3 (4.6k) | 5×5 (5.1k) | multi-scale (9.3k) | verdict |
|---|---|---|---|---|
| Heat (local) | 0.104 | 0.046 | **0.024** | wider perception monotonically better |
| Navier–Stokes (global) | 0.576 | 1.486 | **0.388** | multi-scale best; plain 5×5 **destabilises** |

**A5 findings.** Widening the receptive field helps both regimes, but *how* matters: on heat
all three improve with reach; on globally-coupled NS, a plain 5×5 stencil **destabilises**
(1.49, worse than 3×3) while the **dilated multi-scale** perception (reaching dilation-4
neighbours cheaply) is best (0.388). Multi-scale dilation is the robust way to widen reach —
it is what lets the multi-scale NCA partially cope with NS's non-locality.

## Ablation A6 — single-step vs multi-step (BPTT) training
<!-- A6:results -->
_(train_steps ∈ {1,4,12}, eval 48; appended when the run completes.)_

## Other planned ablations (infrastructure ready)
- **A4 Conservation on/off at fixed backbone** — flux-divergence head vs direct residual
  head, same perceive+MLP (isolated conservation contribution; partially covered by
  plain_nca vs pi_nca but with matched widths).
- **A5 Neighbourhood / perception size** — 3×3 vs dilated/multi-scale perception (locality
  vs receptive-field reach).
- **A6 Physics-loss weighting** — distillation-only vs + PDE-residual term.
Each reuses `harness.py` knobs + `bench.py`; results appended here with mean ± std.
