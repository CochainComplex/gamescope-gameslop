# Frame generation quality roadmap

Updated 2026-10-19. Status: **engineering plan and experiment backlog**. Benefits,
thresholds and new interfaces below are hypotheses or proposed gates, not shipped
behavior or measured improvements. This document expands the five opportunities
in `.claude/reviews/framegen-quality-opportunities-2026-10-18.md` into work that
can be implemented, compared and rejected independently.

## 1. Product objective and current state

Build vendor-agnostic, frames-only generation with a good balance of image
integrity, motion continuity, throughput, control response and frame pacing.
The best-looking isolated frame is insufficient if it adds a frame of delay;
the fastest warp is insufficient if moving edges tear or shimmer. Keep a set of
measured tradeoffs rather than collapsing everything into FPS or one quality
score. Bidirectional interpolation is a useful quality reference and an opt-in
latency tradeoff, not a causal low-latency baseline with equivalent information.

The user's observations guide the failure corpus: bidir can look good but feel
laggy; warp can feel responsive but produce disturbing artifacts; Spatial is
promising in play. These are valuable reports, not controlled measurements of
the mechanisms responsible. Do not dismiss them or overgeneralize them.

Repository state checked while preparing this plan:

| Item | State and interpretation |
|---|---|
| Playable Spatial tree | `.claude/reviews/tools/wt-spatial`, branch `feat/framegen-spatial`, at `a2e7dd6c` during inspection. This includes the compositor-thread shutdown profile-flush guard. Recheck before editing. |
| Spatial experiment | Already implemented by `d3d144d4`, integrated with the menu at `96ad25e8`. Proposal 1 is partly built; it must be evaluated and refined rather than recreated. |
| Held-learning integration | Separate `.claude/reviews/tools/wt-spatial-learning`, branch `integration/spatial-learning`, at `c64baca08`, learning commit `e561e24cb`. Not adopted into the playable tree; does not yet contain the newer `a2e7dd6c` commit. |
| Integration evidence | Previously recorded release build and 79/79 tests pass at `c64baca08`. Earlier held-Checked GPU runs preserve legacy weights but precede final refinements. Final integration GPU validation remains open. These are historical results, not a new run for this roadmap. |
| Claude's campaign | `.claude/reviews/tools/fg-matrix`, recorded `snap/HEAD=6d695e79`. It owns the GPU. Record source status/patches and binary hash as well as HEAD; a HEAD label alone does not establish an exact tested source tree. |
| Main checkout | The old playable branch has other sessions' changes. Source descriptions here refer to the Spatial worktrees unless explicitly stated otherwise. Do not replace the old build or assume it has Spatial. |

See `.claude/reviews/spatial-learning-integration-2026-10-19.md` for the integration
handoff. Preserve newer target commits when integrating; never reset the target
to the earlier private UI snapshot `701cb200e`. This roadmap does not authorize
overlapping GPU campaigns, changing existing game profiles, or publishing code.

## 2. Shared architecture and evidence contracts

### 2.1 Separate the mechanisms being improved

| Mechanism | Existing implementation / navigation anchor | Responsibility |
|---|---|---|
| Hierarchical matching | `src/shaders/cs_framegen_motion_match.comp`, `cs_framegen_motion_match_refine.comp` | Find coarse-to-fine luma correspondences, regularize small motion, refine subpixel displacement. |
| Consistency | `cs_framegen_motion_fbcheck.comp` | Reverse correspondence and round-trip evidence; uncertainty is not a physical depth map. |
| Learned refinement | `cs_framegen_motion_net.comp`, `cs_framegen_motion_net_train.comp`, `cs_framegen_motion_net_opt.comp`, `framegen_evidence.h` | Refine flow/confidence with bounded updates and shared evidence gates; optimize and serve weights. |
| Spatial reconstruction | `src/shaders/framegen_reconstruct.h`, `cs_framegen_motion_warp.comp` | Select covering motion hypotheses using full-resolution color evidence instead of blending incompatible layer vectors. |
| Temporal prediction / richer reconstruction | `cs_framegen_motion_warp_accel.comp` | Predict/Guided mechanisms, including acceleration, reservoir evidence and bounded shading correction. |
| Host resource/dispatch logic | `src/rendervulkan.cpp`: `framegen_prepare_motion`, `framegen_warp_slot`, `framegen_create_luma_reservoir` | Allocate supported resources, select variants and record work with correct lifetime/synchronization. |
| Scheduling / cost evidence | `src/framegen/scheduling.hpp`, `src/framegen/device.cpp`; private `learning.hpp` | Decide what fits, retain recovery state and independent training timing. |
| Persistent model identity | `src/framegen/net_profile.hpp` | GSFP sets, encoding identity, Spatial role, embedded GSFR compatibility. |

Keep policies CPU-testable, shader operations explicit, and GPU lifetimes owned
by the renderer. Do not introduce a second independent scheduler, model store,
capture format or matcher without demonstrating that extending the existing one
would break a clear contract. Experimental names in this plan are not CLI flags.

### 2.2 Non-negotiable invariants

1. Causal modes use only information available at prediction time. A future
   label may supervise later updates; it must not become a prediction input.
2. Spatial keeps its own weight set **inside the same per-game GSFP file**, with
   its role bit `0x10000` combined with the encoding key. Do not implicitly serve
   another mode's trained set. Diagnostic manifests are not replacement weight
   stores. Deliberate warm-start experiments must be explicit and reversible.
3. Train and estimate motion from clean real frames. Generated output and the
   debug HUD do not become learning/history inputs. Retain the menu/overlay work.
4. History carries frame identity, timestamps, encoding, dimensions and lifetime
   provenance. Reject stale, incompatible or out-of-order evidence.
5. No new default buffering for proposals 1, 3 or 4. Proposal 2 starts offline;
   proposal 5 stores already-observed past evidence, without holding presentation
   to wait for future information.
6. Charge preparation, inference, reconstruction, copies, training and new
   synchronization to their actual pipeline shapes. Measure generation-only
   cost separately from background training.
7. CPU recording is followed by fresh-clock admission. Retain known GPU training
   cost and outstanding background query associations across time and scene
   cuts. A known slow batch must not revert to a cheap cold estimate periodically.
8. Frozen mode never trains; cancelled recording cannot advance optimizer or
   saved progress. Completion tokens pin resources and gate readbacks.

Additional queues and queue priorities do not provide portable deadline/QoS
guarantees. Explicit dependencies are required between queues; concurrent image
sharing is not a memory/execution dependency. See the [Vulkan queue rules](https://docs.vulkan.org/guide/latest/queues.html)
and [queue-priority specification](https://docs.vulkan.org/spec/latest/chapters/devsandqueues.html#devsandqueues-priority).

## 3. Q0 — Establish trustworthy comparison tools first

This is a prerequisite for promoting any quality change, not a request to stop
all useful shader prototyping. Complete the harness work on CPU while the active
campaign owns the hardware; schedule GPU parity and captures after handover.

### Existing tools and their limits

| Tool | Reuse it for | Do not infer |
|---|---|---|
| `scripts/framegen-field-gate.py` | Existing A/B → future C causal grayscale comparison, endpoint field diagnostics, changed/static error and lost/false edges; already accepts `--net` and `--net-key`. | Exact Spatial RGB reconstruction, arbitrary displayed phases or full-sequence smoothness. Its changed-pixel eligibility omits whole-static cases and currently couples predictive/bidir sample eligibility. |
| `scripts/framegen-temporal-gate.py` | Sequence error, flicker, edge flicker, lost/false edges and return to the next real frame. | Learned/Spatial evaluation without repair: `fields()` currently passes `None` to the field arm, with no model/key interface. Do not incorrectly attribute this defect to field-gate, whose model loading already works. |
| `scripts/framegen-net-eval.py` | Keyed-model parsing, low-resolution endpoint structure/residual diagnostics. | Displayed temporal stability: `temporal_sigma` is a standard deviation of endpoint residuals, not an order-sensitive generated-sequence test. |
| `GAMESCOPE_FRAMEGEN_RECORD_COLOR` / `scripts/framegen-color-eval.py` | Existing GSCF paired full-resolution color capture and held-out B comparison, preserving transfer/format metadata. | Causal future quality: the existing experiment uses A/C to reconstruct B. Captures and writes also perturb pacing. |
| `scripts/framegen-hindsight-eval.py` | Separate matching error from continuation/prediction error on consecutive captured samples. | That better matching alone can remove reversals or unseen future changes. |
| Analytical GPU guard / Spatial guard | Known transforms, phases, cuts, bounds, HDR/alpha and deterministic invariants through actual shaders. | Native gameplay preference, input response, sustained load or all drivers. |
| `fg-matrix` campaign | Frozen/online comparisons, source-cap regimes and learning curves from its recorded snapshot. | Validation of another binary or the private held-learning integration; native display latency from headless runs alone. |

### Q0 implementation units

- **Q0a — Identity and immutable evidence.** Every result records binary and
  profile hashes, source HEAD plus dirty patch, GPU/driver/backend, queue and
  format capabilities, dimensions, actual source timestamps, target phases,
  display/refresh policy, environment settings and a validity reason. A changed
  snapshot or profile invalidates resume reuse. Keep interrupted runs out of
  decisions; preserve their artifacts separately.
- **Q0b — Learned temporal replay.** Extend temporal-gate to load a frozen
  explicit profile/key, reuse `framegen-net-eval.py::load_blob` / `parse_key`
  following field-gate's existing delegation, and reject missing/incompatible
  models before replay. Add static cases separately; do not silently count an
  omitted scene as a success. Use the same eligible sample IDs for paired arms.
- **Q0c — Spatial and color parity.** Render actual production shader outputs
  for the replay cases, or verify the CPU replay against those outputs. Port the
  exact four-hypothesis RGB rule, confidence-scaled displacement and fallback.
  Establish format-aware tolerances before making a quality claim.
- **Q0d — Causal reference sequences.** Extend existing capture/replay with a
  distinct causal experiment kind and enough timestamped real color frames.
  Preserve GSCF's current interpolation meaning; do not relabel A/C → B as
  causal A/B → future. Any capture extension needs explicit versioning and
  decoder validation.
- **Q0e — Comparison report.** Report per-scene and per-regime paired deltas,
  valid counts, uncertainty intervals and representative worst cases. Where
  only window summaries exist, label a median of window p95 values honestly;
  it is not the global p95. Without logged denominators, label averaged quality
  ratios as window means rather than inventing a pooled pixel rate.

Q0 is complete when a deliberately broken sequence ordering, wrong profile key,
incorrect phase, static-drag regression and illegal future input all fail the
appropriate test. It must distinguish a sharper single frame from a smoother
displayed sequence without requiring a new production network.

## 4. Q1 — Make spatial edge reconstruction cheaper and better

**Hypothesis:** use high-resolution evidence selectively at motion boundaries to
avoid an average foreground/background vector, while retaining causal response.
This is the first runtime opportunity and is already partly implemented.

### What exists

Spatial calls the shared `reconstructField()` in `framegen_reconstruct.h` from
`cs_framegen_motion_warp.comp`. Four covering flow hypotheses are evaluated using
color correspondence, distance, confidence and a winner margin. Smooth flow
regions exit early. Spatial excludes the Predict/Guided temporal-history path
and keeps independent learned weights. Guided remains a richer comparison arm.

### Small implementation sequence

1. **Q1a: explain the existing gain.** Compare Checked, Learned, Spatial and
   Guided using the same real input sequence, source cadence and frozen weights
   where applicable. Separate matcher/preparation, inference and output-warp
   cost. Include boundary masks and smooth full-screen pans.
2. **Q1b: Checked + reconstruction ablation.** On a new experimental branch,
   invoke the existing spatial reconstruction on the raw Checked field, without
   learned inference, acceleration, shading or reservoir allocations. This asks
   whether a cheaper useful option exists and how much of Spatial's benefit
   requires the model. Start with a private specialization/arm; do not add menu
   choices until compatibility and benefit are demonstrated.
3. **Q1c: bounded selection work.** Evaluate the existing smooth-field exit and
   ambiguous-boundary gate. Change one gate or candidate-cost component at a
   time; retain a reference version without the optimization. Measure the
   worst case of full-screen fine detail, not just sparse edges.
4. **Q1d: stable choices without lag.** If winners alternate, diagnose whether
   the cause is matching ambiguity, tie-breaking, correspondence or motion
   changes. Try deterministic ties and stronger current evidence before any
   temporal smoothing. A smoother winner field that drags objects is a failure.

### Required experiment and decision

Use a thin foreground silhouette over textured background, fences, foliage,
particles, crossing objects, repeated patterns and rapid pan/stop/reversal.
Render identical stored inputs at several served phases, then judge sequences
at normal speed. Preserve alpha/HDR invariants and include ambiguous ties.

Promote a candidate only if boundary tearing/stretching decreases without more
ghost edges, foreground drag, shimmer or fallback holding, and its added cost
fits on supported hardware. If Checked + reconstruction wins its cost/quality
tradeoff, retain it as a distinct measured result; it does not prove learned
refinement is generally unnecessary. Stop tuning a threshold if improvements
depend on one scene or disappear under full-screen ambiguity.

**Artifact:** paired crops/clips, per-pass timings, boundary/temporal report,
supported/unsupported configurations and a decision to retain, revise or reject.

## 5. Q2 — Train for the motion actually displayed

**Hypothesis:** an unchanged small inference model can improve causal visual
motion if its training/calibration objective matches the served reconstruction
and the correction at the next real frame. This is not established by endpoint
loss reduction or additional optimizer steps.

### Existing mismatch and foundations

The training shader primarily optimizes endpoint luma reconstruction,
confidence and bounded corrections. The served warp ordinarily scales future
displacement by confidence; Spatial additionally reconstructs the field using
full-resolution RGB. Better endpoint alignment can coexist with hesitant
advancement, reversal overshoot or a visible correction at the next real frame.
The existing causal shading head is not a future-geometry training objective.

Reuse the shared confidence-raise evidence in `framegen_evidence.h`, the offline
trainer, field/temporal gates and full-color capture infrastructure. Do not
re-enable endpoint-trained bidir geometry corrections: bidir's established
learned authority is a confidence veto. Train/evaluate interpolation separately.

### Causal experiment definition

Start with high-cadence timestamped reference frames R0, R1, R2, R3, R4. Simulate
lower-rate visible real inputs R0, R2, R4. Predict hidden R3 using **only R0/R2**,
the model version and timing evidence available at R2. For this example:

```text
phase = (t3 - t2) / (t2 - t0)
displayed test sequence = real R2 -> generated prediction of R3 -> real R4
reference sequence     = real R2 -> hidden real R3             -> real R4
```

For irregular cadence, supply the model only the production-available planned
target time and preceding source interval. The future arrival time of R4 is a
label-alignment fact, not an oracle timing input. Check time units and provenance.
Never use a future-derived flow field or features from R3/R4 for the prediction.

### Q2 implementation sequence

1. **Q2a: replay only.** Implement Q0 parity and hidden-frame labels before
   changing any loss. Keep frozen neutral and existing trained profiles as
   controls. Score complete intervals, including the transition back to real.
2. **Q2b: confidence-only calibration offline.** Hold geometry/features fixed
   and test a bounded confidence-head update against the actual served output.
   Include uniform/static controls; a model that reduces displacement everywhere
   can appear safe while making motion visibly uneven.
3. **Q2c: sequence-aware objective ablation.** Compare endpoint-only training
   with displayed image/edge error, reference-relative temporal-change error and
   real-frame correction terms. One useful diagnostic is the change in error
   between consecutive displayed samples, including generated-to-real; do not
   equate it with physical input latency or universal perceptual quality.
   Publish loss scales and weights before inspecting the held-out test set.
4. **Q2d: geometry changes only if needed.** Determine whether the existing
   inputs/outputs can express a useful improvement. Phase conditioning or new
   history features would change the model contract and require separate
   version/profile work; unchanged inference size is a hypothesis, not a pledge.
5. **Q2e: optional delayed online supervision.** Only after offline success,
   design a bounded record of causal features, timestamps and model version.
   Once a later real label arrives, update future predictions only. Specify
   ownership, input pins, memory cap, staleness/drop policy and deadline budget
   before adding GPU dispatches. Online real frames do not reveal every hidden
   intermediate phase; do not pretend they provide dense future ground truth.

### Safeguards and acceptance

Keep cuts, disocclusions, exposure changes and unlearnable effects separately
classified rather than treating all large errors as training opportunities.
Keep current finite-weight validation, bounded corrections, prior regularization
and serving guards. Freeze the weight snapshot used by each evaluated prediction;
later labels must not retroactively improve its reported score.

Accept only paired improvements in causal sequence quality and real-frame edge
correction without increased ghosting, blur, static drag, confidence collapse or
missed deadlines. Hold/blur controls must not win merely by hiding edge energy.
Report abrupt onset, stop, reversal, irregular cadence and long-horizon prediction
separately. Endpoint PSNR or training loss alone is not an acceptance criterion.

**Artifact:** versioned causal dataset manifest, leakage tests, parity report,
frozen candidate model, objective ablations and unseen-scene results. Start with
a copied experimental Spatial profile, preserving the user's original weights.

## 6. Q3 — Improve matching under lighting changes and repetition

**Hypothesis:** more useful correspondence evidence and a real ambiguity measure
can preserve coherent motion with fewer artifacts than simply rejecting pixels.

### What exists

The matcher uses 3x3 luma SAD, a small-motion prior and coarse-to-fine search.
Refinement includes a vector-median seed, local search, zero-motion anchor and
guarded subpixel estimation. Forward/backward evidence exists. Confidence
compares the winning motion with zero motion, not a distinct competing moving
solution. Repeated textures can therefore have several plausible motion basins.

### Separate experiments

| Unit | Change | Specific question |
|---|---|---|
| Q3a | Instrument best and best spatially distinct alternative on CPU replay. | Can an ambiguity gap predict wrong-layer selection or winner flicker beyond the existing zero-motion comparison? |
| Q3b | Mean-removed/bounded normalized luma matching. | Does it resist additive/exposure change without losing motion in low-contrast regions? |
| Q3c | Gradient-assisted or small census-style descriptor. | Does robust local structure help real illumination changes/repetition at acceptable cost? |
| Q3d | Spend a bounded extra candidate/refinement budget only where evidence warrants it. | Can ambiguous regions improve without raising field resolution or cost everywhere? |

Do not count adjacent samples of one subpixel minimum as independent hypotheses.
Define separation in documented field/full-resolution units; retain raw matching
error separately from regularization. Rescale/recalibrate confidence for a new
descriptor rather than applying SAD thresholds to an unrelated cost distribution.
Check whether the existing quadratic subpixel fit remains valid for discrete
census costs; disable or replace it explicitly if necessary.

First extend analytical/replay cases with known translation, brightness offsets,
gain/exposure ramps, dark noise, repeated fences, scrolling text, specular motion,
translucency and conflicting layers. Then use held-out game captures. Gradient
or census evidence is not a general solution for reflections, particles or
content that does not obey a single correspondence.

Accept more correct motion and fewer temporal boundary failures, not merely
higher confidence or lower changed-pixel coverage. Report static false motion,
reverse consistency, wrong-pattern jumps, learned-feature distribution changes
and all added preparation/texture cost. A new descriptor can require retraining
or profile isolation even if its tensor dimensions are unchanged.

Start without additional GPU images. If another RGBA16F /8 field is justified,
3840x2160 requires approximately 0.99 MiB per field before alignment and other
resources. Specify formats/capabilities and fallback behavior; measure actual
traffic, occupancy and worst-case divergence rather than inferring cost from
allocation size.

**Artifact:** descriptor/candidate ablation table, confidence calibration plots,
analytic error cases, unchanged-cost control and a decision on selective GPU work.

## 7. Q4 — Make training updates and priors more useful

**Hypothesis:** improve the information in each fixed training budget and the
initial weights before increasing inference size or training frequency.

### What exists

Online training currently dispatches 16 tiles per update, split between forward
and reverse directions; tile origins use deterministic uniform hashing in
`cs_framegen_motion_net_train.comp`. The offline trainer also selects frames and
crops randomly. Its last-tenth validation split is not a held-out-game protocol.
Adam, regularization toward the prior, bounded state and served-weight EMA
already exist. External pretrained-weight loading already exists too.

### Q4a — Stratified tile sampling

Keep the same tile count, optimizer settings and both directions for the first
experiment. Compare uniform selection with a deterministic mixture of uniform
coverage and informative regions from existing luma/motion evidence. An initial
test mixture can reserve half the budget for uniform samples and divide the
remainder between motion boundaries and trustworthy moving/ambiguous regions.
These proportions are experimental parameters, not new defaults.

Score a small bounded candidate set; avoid a full-resolution CPU scan, GPU→CPU
readback, unbounded retry loop or extra inference pass. Deduplicate tiles and
normalize weights/counts so a different sampled area does not accidentally
change the effective learning rate. Preserve static examples and both
correspondence directions. Compare equal tile/update and equal GPU-time budgets;
equal optimizer counts alone can hide selection overhead.

Do not mine only the largest residuals: cuts, reveals, flashing effects and
noise may be large precisely because they are not learnable. Freeze sampling
seeds for paired comparisons and record selection frequencies, duplicate rate,
informative/static coverage and accepted/skipped updates in test artifacts.

Accept improved boundary behavior and time-to-useful adaptation without static
drift, confidence collapse, forgetting or more gameplay deadline interference.
Reject a strategy that wins only on its own sampled loss or requires greater
total training time to reach equivalent quality.

### Q4b — Better starting priors

Existing activation precedence is the exact matching per-game set, then an
explicit external blob, then neutral initialization. A better packaged prior
should reuse this path instead of creating a second loading mechanism.

1. Split train/validation/test by game and scene before crop generation. Keep
   near-duplicate frames in one split; reserve unseen content and effects.
2. Train bounded candidate priors offline, with explicit encoding and feature
   applicability. Do not assume one blob is appropriate across gamma/PQ/scRGB.
   Existing explicit blobs may seed any missing key, so applicability is a
   deployment requirement, not something the current loader guarantees.
3. Compare neutral, prior-frozen and prior-online arms. Measure first-use quality,
   adaptation curves, steady quality, robustness and forgetting at equal budgets.
4. Record dataset/model hash, feature contract, optimizer/loss settings and role
   applicability in an experiment manifest. Keep actual mode-specific learned
   sets inside GSFP. Explicit warm-start copies must not alter the donor set.
5. Test save/reload, short-session exit, mode/encoding isolation and failed-load
   preservation. Current profile capacity is eight sets; account for role ×
   encoding combinations before inventing more model identities.

Sampling alone need not change tensor layout. New input normalization, feature
meaning or output semantics requires explicit compatibility decisions, and new
tensor layouts require GSFR/schema validation updates. Preserve incompatible
originals; do not silently overwrite profiles or copy the wrong role as a
convenience. Any format change is its own reviewed task.

**Artifact:** fixed-budget learning curves, unseen-content report, frozen prior
blobs/manifests and a profile compatibility/preservation test matrix.

## 8. Q5 — Explicit visibility, then bounded background color history

**Hypothesis:** distinguishing missing visibility from bad correspondence can
improve fallback decisions. Actual recovery of a newly visible background needs
usable past color evidence; luma validation alone cannot supply missing RGB.
This is the highest lifetime/bandwidth/ghosting risk and should come last.

### What exists and what it cannot do

FB checking already records round-trip error and tests multiple reverse
hypotheses. Guided's `reservoirCandidate()` uses two-interval-old luma plus local
occlusion and three-frame correspondence evidence, then samples **current-frame
color**. The shading head applies a bounded residual. Neither is a general
historical background-color cache, and Spatial intentionally excludes that
temporal-history path.

### Q5a — Visibility reasons without RGB history

Separate out-of-bounds projection, inconsistent round-trip, competing matches
and plausible reveal/collision evidence instead of treating them all as one low
confidence value. These are evidence categories, not oracle visibility/depth.
Initially derive them from existing fields in the current pass or a supported
bounded representation. Use them to restrict reconstruction eligibility and
choose fallback, then test whether that alone improves visible outcomes.

Reject this stage if classification merely suppresses more motion, produces
unstable masks at silhouettes or adds global cost for little recoverable area.
Only proceed when the error corpus demonstrates a substantial class of reveals
for which observed history could actually provide useful color.

### Q5b — One bounded older-color source

Before implementation, specify:

- Which clean **real** color image is owned, at what resolution/format, age and
  timestamp; whether it is copied or safely retained; memory/traffic bounds.
- How the historical point is transported to the target and how same-layer
  correspondence is verified. Round-trip agreement alone is not proof that a
  surface remained visible or unchanged.
- Validity masks for bounds, age, motion uncertainty, collisions, cuts, alpha,
  transfer/color changes and exposure. Never fill from generated history or
  recursively reuse the previous synthetic fill.
- Conservative blending/selection limits, invalidation and fallback. A small
  unfilled region can be preferable to a confident stale object trail.
- Resource transitions, queue dependencies, completion pins, ring ownership,
  reset/resize/shutdown behavior and supported-format fallback. Do not retain
  presentation images indefinitely or introduce a wait for history.

At 4K, one uncompressed full-resolution RGBA16F image contains about **63.3 MiB**
of texel payload, excluding alignment, masks and extra buffered copies. One
uncached full-image read plus write at 120 Hz represents roughly 15.9 GB/s of
payload traffic; this is illustrative accounting, not measured DRAM bandwidth
after caching/compression. Prefer a justified resolution/format/coverage budget
and measure it. One extra history slot is a substantial change on an iGPU.

### Required evidence

Use controlled foreground reveals, crossing objects, camera pan/reversal,
animated backgrounds, changing lighting, particles, transparencies and cuts.
Include the counterexample of content never observed in a usable real frame:
the algorithm must fall back rather than claim recovery. Measure trails, stale
color, edge popping, valid recovered area and bandwidth separately from endpoint
residual. Do not judge a cache solely by PSNR on repeatable static backgrounds.

**Artifact:** visibility confusion/failure analysis, bounded-history design,
lifetime diagram, worst-case memory/traffic report and color/temporal ablations.
No default enablement without broad native gameplay evidence.

## 9. Shared evaluation: how to preserve a balanced result

### 9.1 Corpus and comparison arms

Cover smooth pan, fine moving silhouettes, repeated textures, foliage, particles,
specular/translucent surfaces, exposure changes, onset/stops/reversals, disocclusion,
scene cuts, HUD/overlay composition and static/noise controls. Include the user's
low-source-rate and 40–60 fps regimes; record observed cadence rather than only
the requested cap. Avoid pairing unrelated portions of different playthroughs.

At a 120 Hz target, controlled source-cap regimes of 60/40/30 fps exercise x2/x3/x4
respectively. Also test irregular cadence and source intervals that cannot fill
all refresh opportunities. Higher configured multipliers do not guarantee that
generation admission or prediction horizon will use every slot. Report requested
and effective mode, real/generated/repeat outcomes and validity separately.

| Axis | Minimum useful comparison |
|---|---|
| Reconstruction | Hold/off, Checked, Learned, Spatial, Q1 Checked + reconstruction; Guided and bidir as separately labeled quality/cost references. |
| Learning | Fixed neutral / fixed trained profile for quality isolation; separate online adaptation and frozen-prior arms. |
| Hardware | At least available AMD and NVIDIA paths; Intel and other Mesa drivers remain explicit coverage gaps until tested. No vendor-name-dependent algorithm correctness. |
| Queues/backend | Dedicated and supported shared-queue fallbacks; nested Wayland and native DRM results kept separate; multi-GPU where applicable. |
| Content/format | SDR and actual supported HDR/float paths, relevant resolutions including bandwidth-heavy cases, alpha and bounds. |
| Timing | Steady and irregular source cadence, short warmup and long steady state, cold and trained states reported separately. |

### 9.2 Quality and human perception

Grade both images and trajectories: changed-region error, lost/false edges,
boundary position, static drag, winner switching, ghost trails, flicker,
generated-to-real correction and reference-relative change over time. A sharp
but oscillating edge and a blurred but stable edge are different failures; keep
both visible in the report. Verify motion-mask eligibility does not exclude the
very static or difficult cases being claimed as safe.

Use normal-speed randomized paired clips and actual controlled gameplay, not
only paused crops or slow motion. Ask separate questions about edge integrity,
motion continuity, perceived control response and overall preference. Keep
content, cadence, display mode and profile state matched. Include easy tracking,
fixed gaze and rapid camera changes; record display settings, viewing distance
and whether pursuit/eye tracking was measured rather than assumed. Small
informal tests establish feedback, not population-wide psychophysical proof.

FovVideoVDP models spatial/temporal/peripheral aspects and display conditions;
interpolation-specific metric work also cautions against relying only on image
scores. Such tools complement the corpus and gameplay tests, not replace them
or measure input-to-photon latency. [FovVideoVDP](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/),
[VFI perceptual metric research](https://arxiv.org/abs/2210.01879).

### 9.3 Throughput, response, jitter and uncertainty

Measure per-pass GPU time, CPU recording/submit time, copies/readbacks, actual
source cadence, generated presentation rate, deadline misses, repeats and
source-to-presentation age. Keep absolute latency, variability and systematic
one-refresh lateness separate. Native feedback must distinguish compositor
decisions from confirmed physical presentation. A headless or captured-quality
run cannot settle the nested one-host-frame issue or prove native low latency.

Control response needs an explicit input-to-visible-response measurement where
possible; source-frame age alone does not establish it. Record workload, power
and thermal conditions. Run interleaved/randomized repeated baseline/candidate
comparisons, characterize baseline noise, and show intervals rather than calling
every tiny delta a regression or every noisy win an improvement.

Proposed initial experiment gates, to agree **before** collecting candidate
results and refine from baseline noise:

- Zero new correctness, preservation, bounds, alpha/HDR or lifetime failures;
  zero added intentional causal-frame buffering.
- A declared incremental shader budget per device, initially target at most
  5% of one refresh interval at p95 for a cheap reconstruction experiment
  (about 0.42 ms at 120 Hz). This is a planning budget, not measured headroom;
  total batch cost and deadline results still decide whether the arm is viable.
- A predeclared meaningful quality benefit in the targeted class, for example
  a paired 5% reduction in a boundary/sequence error with uncertainty reported,
  with no material static/ghosting regression. This illustrative percentage
  must not substitute for visual agreement or be tuned on the held-out set.
- No new systematic source-to-display frame of delay or recurring timing spikes;
  tail/miss-rate non-inferiority margins are set from repeated baseline noise
  per regime. Never average away a bad device, source rate or content class.

Failure to meet a draft gate means revise/reject or retain an explicitly costly
opt-in arm; do not change default timing behavior to disguise its cost.

## 10. Driver-facing engineering review

Before retaining each runtime experiment:

- Query storage/sample/transfer support and memory types; do not assume R8
  storage or host-coherent memory is universal. Honor mapped layout offsets and
  row pitch. Existing memory-type/layout/profile-preservation findings remain
  separate tracked correctness work, not implied fixes in this roadmap.
- Preserve descriptor/command lifetime, scratch-image hazards, cross-queue
  ownership/visibility and completion-gated readbacks. Prefer existing passes
  and descriptors when they express the operation cleanly; avoid speculative
  full-resolution copies or steady-state device/queue waits.
- Inspect register pressure, occupancy, memory access, divergence and workgroup
  behavior on actual drivers. Do not hardcode subgroup width or assume equal
  performance from one shader shape. [Vulkan subgroup guidance](https://docs.vulkan.org/guide/latest/subgroups.html).
- Keep costs keyed to actual batch shape and count; isolate background timing.
  A dedicated queue is not proof of independence from rendering contention.
- Make unsupported paths explicit and conservative; preserve the established
  fallback and shared-queue behavior. Add tests for rejection, not only success.
- Keep profiles, prior compatibility and sampling/feature semantics in the
  review. A stable tensor shape does not by itself preserve model meaning.

## 11. Execution order and reviewable change units

| Order | Work item | Entry condition | Exit artifact |
|---|---|---|---|
| 0 | Finish held-learning adoption and shutdown fix reconciliation | Claude campaign/handover coordinated; preserve `a2e7dd6c` and newer commits. | Final integration functional GPU cases, pacing comparison, reviewed merge/build; originals preserved. |
| 1 | Q0a/Q0b evidence + learned temporal replay | CPU work can proceed now; no campaign mutation. | Model/key validation, order-sensitive sequence tests, static controls, immutable result manifest. |
| 2 | Q0c/Q0d production Spatial parity + causal color reference | GPU handover; bounded capture plan. | Replay/capture fixtures with exact causal inputs, timestamps and shader parity. |
| 3 | Q1a/Q1b reconstruction ablation | Frozen comparable inputs and profiles. | Existing Spatial characterization plus Checked + reconstruction decision. |
| 4a | Q3a/Q3b matcher evidence | Q0 parity and labeled failures. | Distinct-candidate and robust-cost ablations; one runtime candidate if justified. |
| 4b | Q4a fixed-budget sampling | Reliable training and sequence evaluation; independent branch/profile copy. | Equal-budget curves and unseen-content report. |
| 5 | Q2a–Q2c output-aware offline learning; Q4b priors | Causal reference labels and validated served-output replay. | Objective/prior ablations and explicit compatibility/provenance. |
| 6 | Q5a visibility; Q5b color only if justified | Remaining reveal errors quantified; simpler arms evaluated. | Visibility analysis before a separate owned-color-history design. |

Q3/Q4 CPU/offline work can proceed independently. Do not combine matcher,
sampling and loss changes in the first comparison; their interaction is a later
factorial/combination experiment. Do not schedule GPU work in parallel merely
because implementation branches are independent.

Each change unit needs a problem statement, exact control/candidate, source and
profile identity, cost budget, tests, known failure cases, independent reviewer
and retain/reject decision. Shader changes require appropriate actual-GPU guards
in addition to CPU policy tests. Build/test the exact final tree after merging.
Reset to a baseline by reverting the experiment or selecting its existing mode,
not by replacing user profiles or moving another session's branch.

### Next session's first concrete task

Read `.wolf/FRAMEGEN_NEXT.md` and current status, recheck campaign ownership and
all branch heads. On a separate branch from the reconciled Spatial stack, start
**Q0b**: add frozen `--net` / explicit `--net-key` support to temporal-gate by
reusing the existing parser, then add order-sensitive sequence and static cases.
Do not call that Spatial evaluation until Q0c parity exists. No renderer feature
or live menu change is needed for this first unit.

## 12. Research map and earlier work

These sources motivate questions, not claims of numerical gains in this fork:

- [Joint Bilateral Upsampling](https://graphics.uni-konstanz.de/publikationen/Kopf2007JointBilateralUpsampling/index.html):
  high-resolution guidance for reconstructing a coarse solution; relevant
  context for Q1, not a description of this exact hypothesis selector.
- [Brox et al., optical flow with brightness/gradient constraints](https://lmb.informatik.uni-freiburg.de/Publications/2004/Bro04a/):
  a reference for robust correspondence terms in Q3, not a proposed full solver
  replacement.
- [UnFlow](https://ojs.aaai.org/index.php/AAAI/article/view/12276):
  occlusion-aware bidirectional census training illustrates robust local
  evidence. Its learned pipeline and benchmark results do not transfer directly
  to this small matcher or causal future prediction.
- [Dense Inverse Search](https://arxiv.org/abs/1603.03590):
  a speed/accuracy reference for correspondence search and aggregation, not
  evidence that replacing this GPU hierarchy is beneficial.
- [Softmax Splatting](https://sniklaus.com/softsplat):
  a forward-warp collision-resolution reference for Q5. It is an interpolation
  method; importing its entire model or assuming future evidence is unsuitable
  for the current causal contract.
- [FovVideoVDP](https://www.cl.cam.ac.uk/research/rainbow/projects/fovvideovdp/)
  and [VFI perceptual metric research](https://arxiv.org/abs/2210.01879):
  motivation for spatial/temporal evaluation and human checks. Neither is a
  latency measurement or a universal predictor of this game's preference.

Reuse [proposal 07](framegen-proposals/07-frames-only-sota-alignment.md),
[research notes](research-framegen.md), [architecture](framegen-architecture.md),
[testing](framegen-testing.md) and [maintenance](framegen-maintenance.md).
Prior review: `.claude/reviews/framegen-five-pov-review-2026-10-18.md`.
Do not blindly reintroduce the removed global camera prior, unrestricted bidir
geometry corrections, blind temporal smoothing, unlimited history, or simply a
larger model. Each previously traded one failure for another or lacks a bounded
failure argument. New evidence can justify a new design; names alone cannot.
