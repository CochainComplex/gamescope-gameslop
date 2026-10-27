# Frame Generation Maintenance and Publication Guide

The contracts a frame-generation change must preserve, the source map, and the
validation and publication checks. It complements the algorithm description in
[`framegen-architecture.md`](framegen-architecture.md); the numbered proposal
documents are design history, not a substitute for checking the current symbols.

Audience: maintainers and reviewers of the frame-generation code.
Related: [Documentation index](README.md) · [User guide](framegen-howto.md) · [Configuration reference](framegen-reference.md) · [Architecture](framegen-architecture.md) · [Testing](framegen-testing.md) · [Proposals](framegen-proposals/README.md)

## I changed X: what do I run?

Start here, then read the matching class in
[Validation by change class](#validation-by-change-class) for the commands.

| You changed | Minimum to run | What it proves |
|---|---|---|
| Docs, launcher scripts, comments | The shell and Python checks from CI (`bash -n`, `shellcheck -x`, the hybrid fixture test, `py_compile`) | Scripts parse and lint; no GPU needed |
| A policy header in `src/framegen` | The standalone `c++` contract test, then `meson test` for `framegen_deadline` | The pure logic still meets its contract; no Vulkan or meson needed for the first step |
| Scheduler, queue, resources or shaders | Full `meson test`, the nested smoke harness ([testing](framegen-testing.md)), and a native DRM/KMS run | Execution and import (nested); real vblank smoothness (native only) |
| Shader math, confidence or ML | All of the above, plus the offline evaluators and matched visual A/Bs | Quality and timing, not only that it runs |
| A flag, variable, default, HUD row or metrics field | Update [`framegen-reference.md`](framegen-reference.md) in the same commit | The reference stays complete |

## Contents

- [I changed X: what do I run?](#i-changed-x-what-do-i-run)
- [Source map](#source-map)
- [Non-negotiable runtime contracts](#non-negotiable-runtime-contracts)
- [Temporal and algorithm contracts](#temporal-and-algorithm-contracts)
- [C++ and shader ABI](#c-and-shader-abi)
- [Resource and cache identity](#resource-and-cache-identity)
- [In-situ learning](#in-situ-learning)
- [Validation by change class](#validation-by-change-class)
- [Publication checklist](#publication-checklist)

## Source map

| Area | Primary source | Ownership |
|---|---|---|
| CLI and environment | `src/main.cpp` | parsing, validation, debug controls |
| Framegen types and pipeline policy | `src/framegen/types.hpp`, `src/framegen/policy.hpp` | mode/pipeline vocabulary and pure degradation-ladder resolution |
| Temporal, scheduling, and dispatch policy | `src/framegen/temporal.hpp`, `src/framegen/scheduling.hpp`, `src/framegen/dispatch_policy.hpp` | pure phase math, cadence/deadline planning, and capability-to-strategy selection |
| Deadline timeline and bidir delay controller | `src/framegen/deadline.hpp` | absolute-deadline admission, the display-chain timing state and the minimum viable present lead (`update_minimum_viable_present_lead`), and the bidir delay D, ring bound and queue planning |
| HUD, metrics and GPU timing | `src/framegen/hud.hpp`, `src/framegen/metrics.hpp`, `src/framegen/query_ring.hpp` | HUD text/rate state, metrics-window accounting, and timestamp-query ring bookkeeping |
| Self-supervised adaptation policy | `src/framegen/adaptation.hpp` | B4 counter decoding, EMA state, and next-batch threshold derivation |
| Numeric and setting contracts | `src/framegen/numeric.hpp`, `src/framegen/settings.hpp` | fast-math-safe fp32 classification and strict scalar/path parsing |
| Shader and ML contracts | `src/framegen/push_constants.hpp`, `src/framegen/net_layout.hpp`, `src/framegen/net_profile.hpp` | CPU/GLSL ABI, tensor layout, and GSFR validation/migration |
| Atomic capture/profile output | `src/framegen/atomic_file.hpp`, `src/framegen/atomic_file.cpp` | unique same-directory staging, checked buffered close, cleanup, and atomic rename |
| Held-out colour capture policy | `src/framegen/color_probe.hpp` | pure reference/endpoint offset selection and measured phase admission; the renderer retains continuity checks, settings, textures, completion and submission |
| Vulkan scheduling and algorithms | `src/rendervulkan.cpp`, `src/rendervulkan.hpp` | history, resources, dispatch recording, and ML execution |
| Queue and timestamp execution | `src/framegen/device.cpp`, `src/rendervulkan.hpp` | framegen submission, completion, command-buffer retirement, and GPU-time accounting |
| Presentation choice | `src/steamcompmgr.cpp` | real/generated/repeat arbitration and timed flips |
| Backend present | `src/Backends/DRMBackend.cpp`, `src/Backends/WaylandBackend.cpp` | final generated-frame substitution |
| Algorithms | `src/shaders/cs_framegen_*.comp` | extrapolation, motion estimation, validation, warp, ML |
| Offline ML tools | `scripts/framegen-net-*.py`, `scripts/framegen-color-eval.py` | GSFD parsing, training, GSFR evaluation, GSCF full-colour evaluation |
| Validation harness | `scripts/framegen-nested-smoke.py`, `scripts/framegen-showcase.sh`, `scripts/framegen-showcase-report.py`, `scripts/framegen-hindsight-eval.py`, `scripts/framegen-headless-host.py` | nested smoke cases, showcase/baseline compare, offline evaluation; see [framegen-testing.md](framegen-testing.md) |
| Contract tests | `tests/test_framegen.cpp`, `tests/framegen_deadline.cpp` | degradation, temporal, cadence/deadline admission, adaptation, dispatch, ABI encoding, net layout, GSFR compatibility, and atomic-output contracts |

Use symbol names in documentation and reviews. Numeric source-line references
become wrong whenever the hot path is reorganized.

The `src/framegen` policy and contract headers are deliberately stateless and
header-only: they add no link boundary to command recording. In particular,
`scheduling.hpp` owns deterministic cadence confidence, the bounded online
source-cadence filter, fixed-slot deadline admission, gap/slot planning,
keep-up thresholds, and deadline-ladder transitions; it does not own clocks,
timestamp provenance, history, or the decision to submit. `device.cpp` is a
narrow exception for `CVulkanDevice` queue/timestamp methods; the class remains
the sole owner of that state. `adaptation.hpp` similarly owns only deterministic
counter decoding and threshold math. `rendervulkan.cpp` retains the mapped-memory
completion gate, readback sequence ownership, scene lifetime, logging, and the
dispatches that consume those thresholds. Keep mutable algorithm resources,
history, and dispatch ordering together until they can move behind one explicit
owner with a proven lifetime. Splitting tightly coupled dispatch helpers merely
to reduce line count is not an architectural improvement and can hide ordering
or ownership bugs.

Repaint request policy in `deadline.hpp` has two temporal stages within
`step_repaint_requests`: fold the previous iteration's requests and carriers,
then plan from the current inputs. Fold even when no paint ended: deferred
work can leave carriers outstanding. Carrier identities come from the previous
iteration's log, while yield age advances at
most once per vblank target. Retire an acknowledged force before comparing the
current counter snapshot; a force re-raised to the same value must get its own
commit. Keep GPU completion and presentation ownership in the consumers.

## Non-negotiable runtime contracts

1. **Real content has priority.** In causal modes, a newly latched game frame
   discards every stale generated candidate before presentation. Bidirectional
   mode is the explicit exception: its generated entries and real endpoints form
   one ordered, bounded presentation timeline.
2. **Normal presentation never waits for framegen.** Readiness uses
   `hasCompletedFramegen`; an incomplete candidate becomes a hardware repeat or
   is discarded. `waitFramegen` is reserved for exceptional reset/teardown paths
   that are about to release referenced resources.
3. **Generated frames are compositor-owned.** They must not create a client
   commit, `frame_done`, presentation feedback, or an anti-lag/Reflex event.
4. **Queue ordering is explicit.** Dedicated-queue work waits for the real
   composite's scratch-timeline value and signals the framegen timeline. The
   shared-queue fallback relies on in-order execution and is capped to one
   generated result per batch to bound head-of-line work.
5. **At most one generation batch is in flight.** This is the ownership rule
   behind the descriptor ring, timestamp query ring, motion intermediates, and
   `genReadA`/`genReadB` pins. Supporting concurrent batches requires explicit
   per-submission ownership for all four; increasing a ring size alone is not a
   correctness fix.
6. **Output images are acquired by ownership, not guessed age.** Real composite,
   generated, and late-composite targets are writable only when
   `CVulkanTexture::IsInUse()` is false. Completed cross-queue read pins are
   released before real-target acquisition. Real pressure first runs one
   non-blocking backend event-progress pass, then invalidates speculative
   history and retries; persistent pressure skips safely. This extra poll is
   exceptional-path only. Generated pool pressure shortens or skips a batch.
   No path rewrites a pending or scanned-out image.
7. **Every framegen submission is covered by reset lifetime.** Update
   `lastFramegenWorkSeqNo` for work that retains framegen textures, including
   descriptor-free copies. Reset must retire that sequence before releasing the
   resources.
8. **Classic fixed-slot mode suppresses VRR and tearing.** VRR hybrid is an
   explicit, dedicated-queue-only timing regime. Do not let a shader-quality
   change implicitly select a present mode.
9. **Deadline-scheduler prediction is admission-only** (called "JIT" in
   [proposal 06](framegen-proposals/06-jit-phase-display-clock.md); the
   `GAMESCOPE_FRAMEGEN_JIT` variable is a legacy no-op, the deadline scheduler
   is the default). It may spend idle presentation-GPU work
   on a disposable backup or skip that work. It must not delay a real frame,
   wait for a generated frame, or override real/generated/repeat arbitration.
10. **Storage-image formats match their views.** Generic output shaders use
   formatless write-only storage images and require the enabled Vulkan
   `shaderStorageImageWriteWithoutFormat` feature. This is what permits the same
   output path to bind RGBA/BGRA 8-bit, RGB10A2, 16-bit UNORM, and floating-point
   views without lying in SPIR-V. Explicitly formatted intermediate shaders must
   continue to match their concrete image views. R16F luma additionally requires
   enabled `shaderStorageImageExtendedFormats`; devices without it use the
   existing RGBA16F luma fallback.

## Temporal and algorithm contracts

- A motion-field vector is **displacement over one observed real-frame
  interval**, in field texels. It is not pixels per second and not a normalized
  velocity.
- Predict/Guided acceleration compares consecutive displacement fields only when
  frame IDs, pipeline, mode, and timestamps are consecutive. The preceding field
  is normalized by `currentDt / historyDt`; the irregular-sample quadratic term
  uses `currentDt / (currentDt + historyDt)`. Missing timestamps or an excessive
  interval ratio disables acceleration instead of guessing.
- `src/framegen/temporal.hpp` owns only those arithmetic transformations. The
  renderer owns timestamp provenance, validity gates, scene history, and the
  decision to submit. Do not move a guard into pure math unless every caller
  preserves the same early-exit and fallback behavior.
- `src/framegen/scheduling.hpp` has the same boundary for cadence admission and
  deadline degradation. The renderer supplies monotonic timestamps, measured
  GPU cost/sample identity, queue capability, and mutable ladder state. A helper
  result must not acquire resources, record commands, wait, or select a flip.
- Acquire-ready cadence and composite-time fallback are distinct timestamp
  provenances. Reset the predictor when provenance changes, timestamps stop
  increasing, or the scene gap expires. Never mix display-quantized and
  pre-vblank intervals in one learned state.
- Causal slot `phase` lies after the newest real frame. The shader coefficient is
  derived from that phase and `--framegen-strength`, then bounded by the forward
  cap. Bidirectional `phase` lies in `[0, 1]` between checked real endpoints and
  does not use the causal strength scale.
- Forward/backward consistency and full-resolution source agreement reduce
  confidence. They do not rewrite a surviving motion vector into an unrelated
  fallback trajectory.
- Conservative bidirectional ML cannot change checked geometry, raise
  confidence, or emit causal shading focus. Its authority is a confidence veto.
- Cheaper pipelines must stay independently usable. Features added to Predict or
  Guided must not allocate, dispatch, or change results in Warp through Learned
  unless that cross-pipeline change is deliberate and separately validated.
- A scene cut, focus change, layer/EOTF change, long gap, resource shape change,
  or base/output-space transition invalidates incompatible history. The first
  real frame after invalidation primes history and does not generate.

## C++ and shader ABI

Framegen push constants bypass the shared upload arena so dedicated-queue work
cannot race its host allocator. The C++ structures from `FramegenPushData_t`
through `FramegenMotionNetOptPush_t` are copied verbatim into GLSL blocks.
`src/framegen/push_constants.hpp` has compile-time per-block size, semantic-member offset,
standard-layout, and trivially-copyable assertions for the blocks it covers;
the cap on block size (`k_uFramegenPushConstantSize`) is asserted in `src/rendervulkan.cpp`.

When changing a push block:

1. Change the C++ structure and every matching GLSL variant in the same commit.
2. Preserve scalar types and byte offsets; do not substitute C++ `bool` for a
   shader integer or float flag.
3. Keep the block at or below `k_uFramegenPushConstantSize` (64, defined in
   `src/rendervulkan.cpp` with its `static_assert`).
4. Build all shaders, including paired, fp16, bidirectional, training, and
   optimizer variants. A successful C++ compile alone does not prove the GLSL
   member order matches.

Other duplicated ABI constants require the same treatment:

- Net shape `12 -> 16 -> 16 -> 4`, parameter count `4644`, and the 2048-wide
  served-weight layout are shared by C++, inference/training/optimizer shaders,
  and the Python trainer/evaluator.
- GSFR magic/version/layer metadata are shared by runtime profile loading and
  the Python tools. `src/framegen/net_profile.hpp` is the runtime source of truth
  for metadata construction, strict validation, finite-weight checks, and the
  v1/v2 shading-head migration. A format change requires an explicit
  compatibility policy and matching tool updates. Serialized finite-weight
  checks use `src/framegen/numeric.hpp` to inspect IEEE-754 bits because the
  production C++ build uses `-ffast-math`; do not replace them with
  `std::isfinite`. The same rule applies to GPU readbacks, measured capture
  phases, CLI input, and environment floats.
- Framegen scalar environment settings use the complete-string parsers in
  `src/framegen/settings.hpp`. Keep range policy at each caller: some settings
  clamp a valid finite number while others fall back when it is out of range.
  Malformed, overflowing, NaN, and infinite values must never reach dispatch
  math.
- GSFD and GSCF capture versions are public analysis-file formats. Reject unknown
  versions rather than partially interpreting them.
- B4's 96-counter layout is shared by the stats, apply, training, and CPU
  readback paths. `src/framegen/adaptation.hpp` is the CPU source of truth for
  the image width, counter interpretation, EMA, and derived thresholds. A shader
  layout change requires matching decoder and boundary-test changes.
- `k_uFramegenDescriptorSets` (`src/rendervulkan.hpp`) must cover the maximum dispatch count of one legal
  batch. The current worst case is 25 (`k_uFramegenMaxDispatchesPerBatch`, also in
  `src/rendervulkan.hpp`) and the ring has 30 entries. Recording
  counts framegen dispatches and fails before a descriptor can wrap within one
  batch; this capacity does not permit a second batch in flight.

`src/framegen/dispatch_policy.hpp` maps observed capability booleans and PCI
vendor ID to an immutable strategy. Vulkan format probing, `ShaderType` mapping,
format-keyed caching, and debug reporting stay in `framegen_dispatch_for_format`.
Unknown vendors use the capability-derived fp16/LDS path. A new hardware
override requires matched microbenchmark evidence and must keep output math
equivalent to the generic variant.

## Resource and cache identity

`previousReal` and `currentReal` retain output-ring images by reference. Real
target selection must continue to use `CVulkanTexture::IsInUse()` so logical
history, unfinished generation reads, and backend scanout ownership share one
correctness check. `genReadA/B` must be released as soon as `genReadSeqNo`
completes, but never before. The generated output pool is disjoint from that
ring. Both pool capacities are sized from the requested multiplier, not the
current degraded multiplier: generated output uses `2·multiplier`, while the
real composite ring uses 8/10/12 slots at x2/x3/x4 to absorb the deeper host
queue caused by higher presentation cadence. These are capacity estimates;
`IsInUse()` is still the safety condition if a backend retains more.

On nested Wayland, `wl_buffer.release` is an aggregate transition to reusable
storage. A repeated attach while the same buffer is still compositor-owned must
not add another lifetime reference: one acquire transition is paired with the
one release transition. Do not replace this with per-commit counting unless a
different protocol supplies explicit per-use completion tokens.

The finalized motion field is reusable only when all identity keys match:

- current real-frame ID;
- motion pipeline;
- causal versus bidirectional mode;
- required forward and reverse field availability.

A reused field runs only per-slot warps. It must not repeat estimation, the B4
probe, learned inference, capture, or online training. Clear the identity on
history invalidation, resource rebuild, held-out color probing, or any operation
that overwrites the working field. Acceleration history additionally requires a
consecutive preceding pair and matching interval metadata.

The timestamp query pool has device/process lifetime. Each batch selects one
unowned pair from its depth-4 ring; a slot is not reset until its previous
query-to-sequence association has been consumed. A temporarily full association
ring skips that optional timing sample instead of overwriting unread data. Scene
invalidation clears associations and costs so a late pre-cut result cannot seed
the new scene; the one-batch completion gate still orders any discarded old query
before a new reset. Resize or format changes do not recreate the pool.

## In-situ learning

The per-pair order is intentional:

1. consume completed readbacks/profile work from the preceding batch;
2. estimate and consistency-check raw fields;
3. capture raw training inputs when requested;
4. run inference with the currently served weights;
5. run B4 cut/trust analysis;
6. record training, with scene-cut gating, and publish optimizer output for the
   next batch;
7. warp generated slots from the finalized field.

Training never consumes its own refined output as ground truth. Same-pair JIT or
idle refills do not train again. A cut zeros the affected gradient work before
Adam consumes it. Non-finite served weights are rejected and the optimizer is
reinitialized from the finite prior.

Profile I/O stays off the compositor hot path. Checkpoints use the owned writer
thread and atomic replace; reset/exit joins it before the final flush. The
renderer owns profile state, logging, and worker synchronization;
`net_profile.hpp` owns the stateless serialized-format checks;
`net_profile_io.cpp` owns cold synchronous GSFR/GSFP input-file reads, returning
status and validated data to the renderer's existing logging/activation path.
Bare GSFR retains legacy migration and trailing-byte acceptance; GSFP retains
its bounded read and exact current-format checks. This input seam does not own
the store, neutral initialization, checkpoint worker or GPU readbacks.
`AtomicOutputFile` owns only the output transaction. It creates a unique staging
file beside the destination, checks every write and the buffered close, then
renames. Destruction before commit removes the staging file and preserves the
old destination. Concurrent instances can race over which complete snapshot is
newest, but cannot share or corrupt a partial staging file. GSFD/GSCF capture
uses the same primitive and remains a measurement-only synchronous path.
GSFD/GSCF payload serialization lives in `src/framegen/capture_file.cpp`. It
accepts borrowed CPU plane views only; the renderer retires GPU writes before
constructing them and owns transaction commit, capture counters, logging,
readback release and post-capture history invalidation. The serializer neither
retains views nor submits work. One packed-row writer serves both formats;
byte-level file contract tests protect metadata, plane order and padding.
Keep the three disk/readback consumers explicitly out of line: they are polled from
`framegen_submit_planned`, and GCC otherwise folds cold string and file machinery
into the Vulkan command recorder even when capture is disabled.

The per-step served-weight readback uses two reusable vectors. GPU data is copied
into the scratch vector, every value is checked with the fast-math-safe finite
test, and only then are scratch and live snapshots swapped. Do not overwrite the
live vector in place: the last healthy snapshot is both the reset warm start and
the only data eligible for persistence after optimizer divergence. During tests,
point `GAMESCOPE_FRAMEGEN_NET_PROFILE` at a disposable copy. Never use a known-good
profile as the writable test destination.

## Offline numerical reference

`scripts/framegen_field_reference.py` owns the shared CPU field/reconstruction
operators, empirical `Numerics` model and bilinear sampler with analytic
gradients. It depends only on NumPy and the standard library. The trainer
re-exports the same sampler; the field gate retains its existing operator
aliases for temporal analysis and other consumers. File-based script imports
bootstrap the resolved sibling directory and use the canonical module name.
Compare different source trees in separate processes to avoid Python's module
cache mixing their implementations.

Keep CLI parsing, scoring, parity thresholds and applicability reporting with
their existing tools; GSFD decoding has the separate reader boundary below. Moving an operator must preserve expression
order, dtypes and half-image boundaries, including both sampler gradients used
in training. The selected model remains empirical: unchanged numerical output
does not qualify another GPU or turn a failed parity gate into a pass.

`scripts/framegen_net_formats.py` owns the offline GSFR/GSFP layout constants,
weight decoding and export. Trainer/evaluator aliases preserve existing callers;
GSFD dataset parsing stays in its separate reader; training, scoring and CLI
argument handling remain in the tools. This format leaf imports neither a driver
nor the numerical reference.
Preserve Python's existing admission behavior independently of the C++ runtime:
bare shape-compatible versions warn without legacy shading migration, bare
trailing bytes are rejected, and profile sets require the current version.
Format refactors need exact bytes, array layout and error/stream comparisons.

`scripts/framegen_net.py` owns the shared NumPy convolution, forward/backward
passes and initialization for the fixed offline refiner. It imports the existing
layout constants from the format leaf; trainer aliases preserve analysis callers.
Keep loss, optimizer and CLI policy in the trainer; cropped features/evidence
have the separate boundary below. Preserve expression
order, ReLU masks, cache ownership and RNG progression (the zero head and biases
consume no draws). Verify nonzero-head gradients as well as neutral output;
neutral-only comparisons can hide changes in the earlier layers.

`scripts/framegen_color_capture.py` owns GSCF v1/v2/v3 admission, metadata
and packed RGB plane decoding. The color evaluator re-exports the existing
names and retains optional LPIPS and CLI policy; NumPy scoring has the separate
boundary below.
Preserve plane/candidate identity, validation order, errors and extra flag-bit
acceptance. Decode stored code values without guessing a transfer function or
clipping HDR range; alpha is ignored and nonfinite pixels reach the existing
scoring policy. Parser changes need exact arrays, metadata and malformed-input
comparisons, including each supported pixel format and capture version.

`scripts/framegen_color_metrics.py` owns full-color code-value box means, SSIM,
luminance, per-frame grading and temporal residual-change measurements, with
evaluator aliases for analysis consumers. It imports only NumPy and `math`.
Preserve reference-based HDR scaling, reduction order and metric thresholds;
do not unify these with field-resolution scoring or infer an EOTF transform.
Per-frame grading sanitizes nonfinite components and reports affected pixels;
temporal scoring deliberately propagates nonfinite residuals. Sequence ordering
and homogeneous-set checks remain in the evaluator. Verify exact metrics and
CLI output, continuity breaks, candidate identity and block partitions. These
image-quality measurements do not qualify live pacing or input latency.

`scripts/framegen_field_capture.py` owns the existing GSFD v1 sample/dataset
readers, with trainer aliases for analysis consumers. It imports no trainer,
numerical reference or model code. Preserve native float16 field views and
contiguous float32 lumas, including RGBA16F red-channel extraction. Keep existing
permissive admission and dataset error/stream behavior: trailing/nonfinite data
is not newly rejected, `load_dataset` skips `ValueError` only, and empty/all-bad
or truncated-header inputs retain their distinct outcomes. Hindsight's broader
exception handling remains with that consumer. Reader extraction is not policy
hardening or a parity improvement.

`scripts/framegen_net_features.py` owns the trainer's cropped twelve-channel
inputs and evidence gate, retaining trainer aliases for analysis consumers.
It uses the canonical bilinear sampler and imports no trainer, model, capture
reader or scoring policy. Preserve casts before crop arithmetic, channel order,
mode and evidence thresholds/expression order. The evaluator's whole-frame
builder retains its separate shape, fixed mode and dtype timing; do not unify
it with the crop builder or merge the distinct SSIM implementations as a source
refactor. Verify channels, sampler coordinates, gate boundaries and actual
training/evaluator outputs; exact CPU extraction does not qualify GPU parity.

## Validation by change class

**Documentation, launchers, comments:**

```sh
bash -n scripts/framegen-doctor.sh   # repeat per file; bash -n parses only its first operand
shellcheck -x scripts/*.sh
bash tests/gamescope_hybrid_test.sh
python3 -m py_compile scripts/*.py
```

Proves: the scripts parse, lint clean, the hybrid launcher fixture passes, and
the Python tools compile. The authority is the "Check framegen maintenance
tooling" step in `.github/workflows/main.yml`, which lists the exact files; do
not maintain a second copy here. The globs above are a quick local shortcut
and may include scripts CI does not check.

**Policy headers only (standalone, no Vulkan or meson):**

```sh
c++ -std=c++20 -Wall -Wextra -Werror -Isrc \
  tests/test_framegen.cpp src/framegen/atomic_file.cpp \
  -o /tmp/gamescope-framegen-contracts
/tmp/gamescope-framegen-contracts
c++ -std=c++20 -O3 -ffast-math -Wall -Wextra -Werror -Isrc \
  tests/test_framegen.cpp src/framegen/atomic_file.cpp \
  -o /tmp/gamescope-framegen-contracts-fastmath
/tmp/gamescope-framegen-contracts-fastmath
git diff --check
```

Proves: the stateless contract tests pass with and without `-ffast-math`
(the second run catches NaN/inf checks that fast math would optimise away).
The standalone `c++` lines build only `tests/test_framegen.cpp`. The second
contract test, `tests/framegen_deadline.cpp`, needs Catch2 and runs only through
meson: `meson test -C <builddir> framegen framegen_deadline` is the fast path.

**C++, shader, resource, queue, or scheduling changes:**

Proves: it builds, the contract tests pass, and the shader benchmark still
runs. It does not prove pacing; the nested and native runs below do that.

```sh
GAMESCOPE_BUILD_DIR=build-perf ./scripts/env-gamescope-local.sh \
  ninja -C build-perf
GAMESCOPE_BUILD_DIR=build-perf ./scripts/env-gamescope-local.sh \
  meson test -C build-perf --print-errorlogs
GAMESCOPE_BUILD_DIR=build-perf PIPELINE=guided \
  ./scripts/test-framegen.sh bench <gpu>
```

`<gpu>` is `auto`, `nvidia`, `amd`, `intel`, or a `vendor:device` id taken from
`./scripts/test-framegen.sh gpus`.

`build-perf` is just a second meson build directory used for benchmarks. Any
name works, and `GAMESCOPE_BUILD_DIR` selects it. Create it once:

```sh
GAMESCOPE_BUILD_DIR=build-perf ./scripts/env-gamescope-local.sh \
  meson setup build-perf -Dbuildtype=release -Denable_tests=true
```

Use the same tree for every script in one comparison. Mixing trees makes
benchmark and cadence comparisons meaningless.

Also run a nested functional test and a native DRM/KMS pacing test. Check debug
logs for real/generated/repeat cadence, degradation changes, scene cuts, output
pool pressure, late drops, and routing. Nested success proves execution and
import; it does not prove native vblank smoothness.

**Shader math, phase, confidence, reconstruction, or ML changes:** perform all of
the above, then compare held-out GSCF/GSFD metrics and run matched visual A/Bs at
x2, x3, and x4 (in forward mode the multiplier changes only buffer sizes, in bidir it changes how many frames are made). Include fast camera motion, disocclusion, translucency, HUD/text,
low source rate, a scene cut, and presenter overload. Report quality and timing;
an image-quality gain that introduces cadence misses is not a net improvement.

### Repowise synchronization

Run repository analysis outside timing and A/B campaigns, using their shared
campaign lock. Check the actual checkout and preserve unrelated edits first.
When incremental updates leave stale pages or incomplete analysis, reconcile the
configured repository and persist a whole-repository health run:

```sh
repowise update --no-workspace --full --yes --docs --no-agents --no-cost-tracking
repowise health --no-workspace --recompute
repowise doctor --no-workspace --quick --format json
repowise status --format json
repowise doc-drift --no-workspace --check --format json
```

In Repowise 0.59, the unfiltered default-table health recomputation persists its
results; JSON or file-filtered recomputation only reports them. Verify stored
health through MCP as well as CLI output, including the analyzed revision and
available coverage. Keep genuine findings; freshness does not mean a clean score.

Regenerate coverage through the relevant tests before ingesting a fresh report
with `repowise coverage add <report> --strict`. Its measurement revision follows
the report's modification time, rather than ingestion time. Preserve original
reports and their timestamps. After committing, verify synchronization against
the new `HEAD`, including pending work and stale-page checks. Keep generated
databases, pages, coverage reports and local receipts out of commits.

## Publication checklist

- No absolute home paths, private benchmark locations, serials, credentials, or
  machine-specific GPU IDs are defaults in tracked files.
- Build trees, wrap downloads, captures, learned profiles, logs, and benchmark
  binaries are not committed. Meson wrap directories stay ignored; wrap files
  and submodule declarations are the reproducible dependency boundary.
- New files follow the repository default license in [`LICENSE`](../LICENSE)
  unless they carry their own SPDX header. Do not redistribute third-party benchmark
  assets with this repository.
- Any new or changed flag, environment variable, default, HUD row or metrics
  field updates [`framegen-reference.md`](framegen-reference.md) in the same
  commit. Verify against the getopt table in `src/main.cpp` and
  `grep -rn GAMESCOPE_FRAMEGEN_ src/`.
- `git diff --check`, the local build, and the full test suite pass. Review the
  final diff specifically for push-layout, queue wait/signal, cache-key, history
  invalidation, and flip-priority changes.
- Documentation distinguishes measured results from hypotheses and nested tests
  from native DRM/KMS validation.

The CI secret scan is defence-in-depth for tracked files only. The realistic
leak channel is runtime logs: umu and other game launchers can dump the entire
environment, tokens included, and those logs are never committed, so the scan
never sees them. Scrub logs before sharing them anywhere.
