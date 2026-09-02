# Architecture Diagrams — Hybrid Models

Data-flow diagrams for every architecture (Mermaid; renders on GitHub). State tensors are
NHWC `(B,H,W,C)`. "1×1 conv" = per-cell MLP; "perceive" = depthwise/learned spatial conv.
All update heads are **zero-initialised** so each model starts as the identity map (a warm start).

## Legend
- **Perceive**: spatial convolution (3×3 circular, optionally dilated/multi-scale) — the only
  non-local op in an NCA; defines the per-step receptive field.
- **Flux head → divergence**: predict a 2-channel flux `(f_x,f_y)` and apply its discrete
  divergence `Δx = (roll(f_x)−f_x)+(roll(f_y)−f_y)` ⇒ **mass conserved by construction**.
- **clip + mass re-project**: `clip(x,lo,hi)` then `conserve_energy` ⇒ bounded **and** conserving.

---

## 1. Conservative PI-NCA (DeepFluxNCA) — the base building block
```mermaid
flowchart LR
    X["state x<br/>(B,H,W,C)"] --> P["perceive<br/>3×3 circular conv, He-init"]
    P --> R1[ReLU]
    R1 --> M1["1×1 conv 64<br/>ReLU"]
    M1 --> M2["1×1 conv 32<br/>ReLU"]
    M2 --> FH["flux head<br/>1×1 conv → 2 ch<br/>(zero-init)"]
    FH --> DIV["discrete divergence<br/>Δx = ∂x f_x + ∂y f_y"]
    X --> ADD(("+"))
    DIV --> ADD
    ADD --> O["x_next<br/>(mass conserved)"]
```

## 2. MultiScaleFluxNCA — heat winner (dilated multi-scale perception)
```mermaid
flowchart LR
    X["state x"] --> P1["perceive d=1<br/>3×3"]
    X --> P2["perceive d=2<br/>3×3 dilated"]
    X --> P4["perceive d=4<br/>3×3 dilated"]
    P1 --> C["concat"]
    P2 --> C
    P4 --> C
    C --> R[ReLU]
    R --> M["1×1 conv 64 → ReLU"]
    M --> FH["flux head (zero-init)"]
    FH --> DIV["divergence"]
    X --> ADD(("+"))
    DIV --> ADD
    ADD --> CP{"bounds set?"}
    CP -->|yes| CLIP["clip + mass re-project"]
    CP -->|no| CE["mass re-project"]
    CLIP --> O["x_next"]
    CE --> O
```
*Dilations 1/2/4 widen the receptive field to ±4 cells per step (vs ±1 for a single 3×3),
closing the locality gap cheaply — why it beats FNO on heat at ~1% of the parameters.*

## 3. BoundedConsFluxNCA — Cahn–Hilliard winner (bounded AND conserving)
```mermaid
flowchart LR
    X["state x∈[-1,1]"] --> TM["record total mass"]
    X --> P["perceive 3×3 → ReLU"]
    P --> M["1×1 64 → ReLU → 1×1 32 → ReLU"]
    M --> FH["flux head (zero-init)"]
    FH --> DIV["divergence<br/>(conserves mass)"]
    X --> ADD(("+"))
    DIV --> ADD
    ADD --> CLIP["clip to [-1,1]<br/>(stabilise; breaks conservation)"]
    CLIP --> RP["conserve_energy<br/>(restore total mass)"]
    TM --> RP
    RP --> O["x_next<br/>bounded AND mass-conserving"]
```
*Resolves the stability↔conservation tension found in ablation A1: clipping alone fixed
divergence but destroyed conservation; the mass re-projection restores it.*

## 4. SpectralFluxNCA — two-stream (local conservation + global spectral)
```mermaid
flowchart LR
    X["state x"] --> LOC["LOCAL stream<br/>perceive 3×3 → ReLU → 1×1 → flux head"]
    LOC --> DIV["divergence → x_local<br/>(conserves mass)"]
    X --> GLB["GLOBAL stream<br/>lift 1×1 → [SpectralConv2d + 1×1] × depth → GeLU"]
    GLB --> PROJ["proj → g (zero-init)"]
    DIV --> SUM(("+"))
    PROJ --> SUM
    SUM --> CE["conserve_energy (optional)"]
    CE --> O["x_next"]
```
*Combines the NCA's local conservation with an FNO-style global spectral term. The global
stream truncates to the lowest Fourier modes (`F⁻¹(R⊙F[v])`) for O(1)-layer global reach.*

## 5. MultiChannelFluxNCA — SWE / FHN / Gray–Scott (per-field conservation)
```mermaid
flowchart LR
    X["state x<br/>(B,H,W,C), C=2..3"] --> P["perceive 3×3 → ReLU → 1×1 → ReLU"]
    P --> FH["flux head → 2C channels<br/>(f_x,f_y) per field (zero-init)"]
    FH --> RS["reshape (B,H,W,C,2)"]
    RS --> DIV["per-channel divergence"]
    X --> ADD(("+"))
    DIV --> ADD
    ADD --> O["x_next<br/>(each field's mass conserved)"]
```
*Correct prior for periodic conservation laws (shallow-water mass+momentum); a deliberately
wrong prior for source-term reaction systems (FitzHugh–Nagumo) — see the regime map.*

---

## Shared emulator training pipeline (all models)
```mermaid
flowchart TD
    IC["random IC<br/>ic.make_state(seed=42)"] --> PS["pre-seed:<br/>solver-evolve preseed_steps<br/>(developed state)"]
    PS --> TGT["teacher = solver.rollout(K)<br/>(differentiable)"]
    PS --> PRED["model rollout K<br/>(lax.scan, BPTT)"]
    TGT --> L["MSE loss"]
    PRED --> L
    L --> OPT["AdamW + LR warmup<br/>value_and_grad → update"]
    OPT --> PRED
    PRED --> EV["evaluate: long-horizon (eval_steps)<br/>20 metrics vs solver"]
```
*Single fixed seed (42), He-init, zero-init heads, LR warmup, and pre-seeding — the
"start from a better point" protocol (`docs/initialization_and_protocol.md`).*
