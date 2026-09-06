# Related work, and what is actually new here

Every arXiv identifier in this document is machine-verified against the arXiv API by
`python -m pinca_jax.bib`; titles and authors in [`bibliography.md`](bibliography.md) are
arXiv's own, not this repository's. This section replaces the broad survey in
[`literature_review.md`](literature_review.md), which remains as background reading. The
purpose here is narrower and less comfortable: to state precisely what prior work already
does, so that what remains is a claim a reviewer can check rather than a claim that
survives only because the neighbours were not named.

## The closest prior work, and how it narrows the contribution

**Saha and Wang, *Learning PDE Time-Stepping with Neural Cellular Automata*
[arXiv:2608.30328].** This is the nearest neighbour and it is very near. They propose an
NCA surrogate for long-horizon PDE time-stepping, motivated in the same way (a small,
local, homogeneous update rule applied identically at every cell, mirroring the locality
of differential operators), benchmarked against PDE-Net [arXiv:1710.09668], a modified
PINN, and an FNO, on five canonical PDEs (heat, advection, Burgers, Allen-Cahn,
Fisher-KPP), evaluated at twice the training time horizon.

The existence of that paper removes several things this repository might otherwise have
claimed. "NCAs are a viable autoregressive PDE surrogate" is not new. "NCAs versus PINNs
and FNOs on canonical PDEs" is not new. "Evaluation beyond the training horizon" is not
new.

**Richardson et al., *Learning spatio-temporal patterns with Neural Cellular Automata*
[arXiv:2310.14809]** trains NCAs on PDE trajectories to recover local rules behind
emergent behaviour, shows generalisation beyond the training data, and constrains NCAs to
respect given symmetries. So "NCAs learn PDE dynamics and can be structurally constrained"
is also not new.

**Praditia et al., *Finite Volume Neural Network* (FINN) [arXiv:2104.06010]**, and
Karlbauer et al. [arXiv:2111.11798], embed the finite-volume structure directly into a
network so that each quantity follows its own adaptable conservation law with explicit
fluxes between control volumes. That is the same structural idea as the flux-divergence
update used here. **A conservative flux-form neural PDE model is not new**, and this
repository should not and does not claim otherwise.

**Richter-Powell, Lipman and Chen, *Neural Conservation Laws* [arXiv:2210.01741]**
parameterise networks that satisfy the continuity equation exactly by construction, via
divergence-free vector fields and differential forms. Exact-by-construction conservation
in a learned model is therefore also established prior art, by a different mechanism.

## What is left

Three things, all narrow, all checkable.

**1. The flux-form conservation prior inside an NCA update rule, and a measurement of
when it helps.** FINN embeds finite-volume structure in a network with per-cell learnable
parameters; the NCA constraint is stronger and different -- *one* homogeneous rule, shared
across every cell, translation-equivariant and resolution-agnostic by construction. The
question this repository answers is not "can a network be made conservative" (yes, FINN,
Neural Conservation Laws) but "when the update rule is also constrained to be a single
local homogeneous CA rule, does the conservation prior still pay, and on which
phenomena". The A4 ablation holds the backbone fixed and switches only the head between a
conservative flux divergence and an unconstrained residual, which is the controlled form
of that question. Saha and Wang's NCA has no conservation structure, so this axis is not
answered there.

**2. The bounded-and-conserving tension, and a projection that resolves it.** Stiff
bounded fields (Cahn-Hilliard, Allen-Cahn) need the state clipped to its physical range
to remain stable; conservative models need total mass restored after the clip. The obvious
composition -- clip, then add a uniform offset to restore mass -- **re-violates the bound
it just enforced**, because a uniform offset moves clipped cells back outside the range.
Measured on Cahn-Hilliard with `pinca_jax.stability`, that is 4.4% of cells in exactly the
model family whose selling point is being simultaneously bounded and conserving.
`physics.conserve_energy_bounded` replaces the uniform offset with one proportional to
each cell's remaining headroom: mass is restored exactly, no cell can cross the bound, it
is a single vectorised differentiable pass, and when the target mass is genuinely
infeasible for the box it saturates and reports the residual instead of silently violating
the bound. Ablation A7 measures the cost of the naive choice rather than asserting it.

**3. A regime map rather than a winner, with the controls that make a null result
meaningful.** Prior comparisons ask whether the proposed model wins. The question here is
which structural prior matches which regime, over ten 2-D and six 3-D phenomena including
multi-field systems (shallow water, Gray-Scott, FitzHugh-Nagumo) that the five-PDE
comparisons do not cover. Three specific protocol commitments make the negative results
usable:

* a do-nothing identity floor in every comparison, so "learned worse than nothing" is a
  reportable outcome rather than an invisible one;
* physics-free CNN and U-Net controls at full and parameter-matched size, so a
  conservation result is not confounded with "an ordinary CNN was never tried";
* paired Wilcoxon tests with Holm-Bonferroni correction on the same initial conditions,
  so `tie` is a reported outcome rather than a rounding artefact.

That combination is the contribution. It is a benchmark-and-analysis contribution, not an
architecture-beats-architecture one, and the paper is framed accordingly.

## Positioning table

Locality: does the model's update read a bounded neighbourhood per step? Conservation:
does the architecture enforce it structurally (S), by projection (P), by penalty (L), or
not at all (-)? Supervision: what is the training signal?

| Work | Locality | Conservation | Supervision | Suite | Dims | What it adds relative to this repository |
|---|---|---|---|---|---|---|
| Growing NCA (Mordvintsev et al., *Distill* 2020) | local CA rule | - | target image | n/a | 2-D | Origin of the differentiable NCA; not a PDE surrogate |
| NCA PDE time-stepping [arXiv:2608.30328] | local CA rule | - | solver trajectories | 5 PDEs | 1-D/2-D | **Closest prior work.** Establishes NCA-as-PDE-surrogate vs PINN/FNO/PDE-Net |
| Spatio-temporal NCA [arXiv:2310.14809] | local CA rule | symmetry constraints | PDE trajectories, images | Turing systems | 2-D | Establishes NCA generalisation and structural constraints |
| NoiseNCA [arXiv:2404.06279] | local CA rule | - | texture targets | n/a | 2-D | Continuous space-time limit of NCA rules |
| FINN [arXiv:2104.06010], [arXiv:2111.11798] | finite-volume stencil | **S** (per quantity) | data + physics | transport | 1-D/2-D | Establishes conservative flux-form neural PDE models |
| Neural Conservation Laws [arXiv:2210.01741] | global (differential forms) | **S** (exact) | data | fluids | 2-D/3-D | Exact continuity by construction, different mechanism |
| PDE-Net [arXiv:1710.09668], 2.0 [arXiv:1812.04426] | local (learned filters) | - | trajectories | canonical | 2-D | Constrained-filter approach to learning PDE operators |
| FNO [arXiv:2010.08895] | **global** (spectral) | - | input-output pairs | canonical | 2-D/3-D | The global-operator baseline used here |
| PINO [arXiv:2111.03794] | global (spectral) | **L** (residual) | data + PDE residual | canonical | 2-D | Physics-informed operator; not yet measured here (see limits) |
| DeepONet [arXiv:1910.03193] | global (branch/trunk) | - | operator samples | canonical | 1-D/2-D | Operator-learning baseline, measured here on heat only |
| MP-PDE [arXiv:2202.03376] | local (message passing) | - | trajectories | canonical | 1-D/2-D | Generalises over geometry/resolution; regular grid here |
| MeshGraphNets [arXiv:2010.03409], GNS [arXiv:2002.09405] | local (mesh/graph) | - | trajectories | mesh systems | 2-D/3-D | Irregular geometry; out of scope on a periodic regular grid |
| Clifford layers [arXiv:2209.04934] | local | geometric structure | trajectories | fluid/EM | 2-D/3-D | Alternative structural prior (multivector fields) |
| PDE-Refiner [arXiv:2308.05732] | global | - | trajectories | canonical | 1-D/2-D | Long-rollout accuracy via a refinement objective |
| PINN [arXiv:1711.10561] | global (coordinate MLP) | **L** (residual) | PDE residual, no solver data | per-IVP | any | Different problem class; see the matched-comparison caveat |
| PDEBench [arXiv:2210.07182], APEBench [arXiv:2411.00180] | n/a | n/a | benchmark suites | broad | 1-3D | The benchmark-design standard this repository follows |
| **This repository** | local CA rule (and dilated / spectral hybrids) | **S** flux-form, **P** bounded headroom projection | solver distillation | 10 2-D + 6 3-D | 2-D/3-D | Conservation prior *inside* an NCA rule; bounded+conserving projection; regime map with identity floor, CNN controls and paired statistics |

## Deliberate scope limits

Stated rather than left for a reviewer to find:

* **PINO is not measured.** It is the most obviously missing baseline for the
  physics-informed axis. The PINN and DeepONet baselines here are heat-only and are
  reported as a separate, explicitly non-matched comparison (see
  [`matched_baselines.md`](matched_baselines.md)); they are not part of the head-to-head
  emulator claims.
* **Irregular geometry is out of scope.** Every phenomenon here is on a periodic regular
  grid, which is where FNO and convolutional NCAs are natural and where MeshGraphNets/GNS
  are not the right comparison. A claim about arbitrary meshes would be unsupported.
* **Burgers and Fisher-KPP, in Saha and Wang's suite, are not in this one**, and their
  1-D setting is not reproduced here. The suites overlap on heat, advection and
  Allen-Cahn only, so no number in this repository should be read as a direct replication
  of theirs.
* **No shared dataset.** This work distils its own solvers rather than training on
  PDEBench or APEBench data, which means the numbers here are not comparable to published
  numbers on those benchmarks. `pinca_jax.teacher_error` quantifies what the local solvers
  themselves get wrong so that limitation is bounded rather than merely acknowledged.
