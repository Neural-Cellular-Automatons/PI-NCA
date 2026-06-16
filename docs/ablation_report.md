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

## Planned ablations (infrastructure ready)
- **A2 Iso-parameter** — match FNO params (~5.9e5) to NCA (~5e3) budget, or shrink FNO, to
  separate "spectral global mixing" from "more parameters" in the heat result.
- **A3 Training-horizon curriculum** — train_steps ∈ {6,12,24,48}; does train≈eval close
  the stiff-PDE gap independent of clipping?
- **A4 Conservation on/off at fixed backbone** — flux-divergence head vs direct residual
  head, same perceive+MLP (isolated conservation contribution; partially covered by
  plain_nca vs pi_nca but with matched widths).
- **A5 Neighbourhood / perception size** — 3×3 vs dilated/multi-scale perception (locality
  vs receptive-field reach).
- **A6 Physics-loss weighting** — distillation-only vs + PDE-residual term.
Each reuses `harness.py` knobs + `bench.py`; results appended here with mean ± std.
