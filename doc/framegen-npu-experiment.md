# Optional NPU assistance: experimental design and spike review

Status: 2026-10-19. Offline companion prototype implemented in
[`tools/framegen-npu`](../tools/framegen-npu/README.md); no renderer integration.
Measured campaign results, data lessons and the fast-path roadmap (how the NPU could free iGPU
time): [framegen-npu-findings.md](framegen-npu-findings.md).
Source report: `.claude/reviews/npu-framegen-spike-2026-10-17.md`. Its scratch
artifacts no longer exist, so the report is historical evidence, not a
reproducible current benchmark. Preserve Claude's running GPU campaign.

## What the spike establishes

The local XDNA2 stack executed synthetic random-weight networks with reported
2.55/4.12/7.76ms p50 and approximately8–9ms p99 for small/medium/large networks.
It demonstrates a possible inference backend. It does not establish better
motion fields, successful training, a usable UI mask, or gameplay improvement.

Several conclusions need qualification:

- **Fixed cost is a hypothesis.** Timing three different architectures cannot
  isolate a universal2.5ms ORT overhead. Host copies, conversions, dispatch,
  compilation state, shape, memory traffic and NPU compute need separate traces.
- **Large-net GPU victory is unproven.** The large iGPU6–8ms number is an
  estimate; NPU7.76ms is measured. There is no measured like-for-like GPU speedup.
  Earlier production timings also cover different resolutions/work and do not
  establish the present Spatial inference/training cost.
- **Separate compute does not mean no contention.** CPU preparation, DRAM,
  package power and thermals can slow the iGPU even without GPU dispatches.
- **A previous pair is not a current pair.** Feeding old flow into new matching
  requires explicit coordinate/age semantics. It can bias search incorrectly
  after acceleration, reversals, cuts, camera motion or animation.
- **Late results can disturb a sequence.** Switching field interpretation on
  later generated phases of the same real pair can produce a correction pop.
  Do not implement this first. Use one immutable decision for a pair/sequence.
- **Baked UI is not necessarily static.** Menus, subtitles, minimaps and camera
  HUDs animate, fade and move. Old masks require conservative expiry and scene
  invalidation, not a guarantee that a frame-late mask is safe.
- **Numeric error is not image quality.** Relative max error on random weights
  does not evaluate motion, disocclusion, confidence calibration or HDR behavior.
- **No demonstrated NPU backward pass.** CPU training can coexist with NPU
  inference; that is different from moving GPU gradients onto an NPU.

The local XRT precedence workaround is useful, but not proof that every earlier
dtype/INT8 failure had the same cause: the default fp32 and INT8 cases were not
rerun with it. Treat them as unresolved. Do not install or replace packages in
the SDK venv, especially its ORT build, to make a generic tool pass.

## Division of responsibility

| Component | Owns | Must not own |
|---|---|---|
| Vulkan renderer | Current fields, generation, presentation, live model buffers and GSFP saves | SDK imports, NPU session setup, blocking helper IO |
| Generic coordinator (future live module) | Capability state, bounded transport, generations and eligibility | Vendor API types, frame-critical scheduling waits |
| NPU adapter helper | Compiled model, inference, cache, SDK environment, execution evidence | Direct scene/presentation changes or profile writes |
| CPU learning worker | Explicit bounded training batches and candidate model snapshots | Concurrent mutation of GPU optimizer or user GSFP |
| Evaluation tools | Frozen comparisons, artifacts, provenance, acceptance gates | Treating completed steps as visible improvement |

The initial prototype implements only isolated offline stages: CPU training or
exact-set load → ONNX/fixture preparation → vendor helper inference → numeric
parity report. It does **not** yet implement the future coordinator or a teacher
loss. That separation keeps experimental dependencies out of normal builds.

A supported NPU means a requested device/runtime can execute the actual model
with verified placement and acceptable numerical behavior. PCI vendor or a
provider name is discovery, not support. No support means baseline GPU behavior;
optional explicit offline CPU training remains available without an NPU.

AMD adapter uses VitisAI ONNX Runtime; Intel uses native OpenVINO `NPU` (the
built-in ORT OpenVINO provider is deprecated). Other vendors can implement the
same narrow interface. There is no vendor enum in profile identity. Runtime
packages/SDKs are installed by the user, not bundled or altered by gamescope.

## Experiments worth pursuing

1. **CPU learning with a frozen NPU teacher.** On recorded real pairs, compare
   a pretrained teacher against current checked/refined fields. CPU student
   learning may mix real endpoint evidence with bounded teacher guidance only
   where the teacher passes evidence tests. Keep the GPU inference/student shape
   unchanged. First prove the teacher beats the baseline and the CPU objective
   preserves static/weak-evidence cases. No usable teacher exists in this spike;
   random weights are not supervision. Delayed CPU real-label learning is a
   separate control and may prove useful without any NPU.
2. **Lagged search proposals.** After an offline win, provide a bounded optional
   search seed, retaining the zero-motion/local GPU candidates and independent
   cost test. Old-flow transport, real-pair gaps and expiry must be modelled.
   Compare incorrect-seed escape cost too. Never overwrite trusted current flow
   merely because a larger network produced a vector.
3. **UI masks or scene classification.** Use only advisory bounded outputs after
   class-specific accuracy and temporal tests. Avoid toggling quality/multiplier
   on every classifier update; policy still needs hysteresis and measured cost.
4. **NPU backward kernels.** Defer. Neither ORT inference nor exporting an ONNX
   forward graph supplies the existing optimizer/gradient semantics. Custom AIE
   kernels would create a significant vendor maintenance obligation.

Follow [Q0–Q5 of the quality roadmap](framegen-quality-roadmap.md). Endpoint luma
is useful execution/training evidence but cannot certify causal Spatial RGB or
displayed sequence quality. Exact model/key and input-generation identity are
required in every offline comparison.

## Future live module invariants

These are design requirements, **not implemented guarantees**:

1. Explicit opt-in; state machine Disabled→Probing→Preparing→Ready, with Failed
   returning to baseline. Asynchronous startup/cache compilation; no frame or
   game launch waits for readiness. Bound retries and require explicit recovery
   after persistent failures.
2. Start with bounded host staging, not dma-buf complexity. One input job in
   flight, one replaceable queued request and one latest result. Submit/poll
   must be nonblocking; full queue means drop. Snapshot immutable data only
   after normal GPU completion; never synchronize a frame to obtain NPU input.
3. Each job/result carries protocol version, model hash, feature semantics,
   game/profile role+encoding key, reset generation, scene generation, real-pair
   IDs, capture time, extent/format and expiry. Arrival order is not validity.
4. Result must be finite, shape/size-bounded, current-generation, matching-role
   and fresh enough. Invalidate on cut, resize, encoding change, mode change,
   profile switch and helper restart. Never reuse generated output as truth.
5. CPU learning uses a private parameter/optimizer copy and bounded work/data
   budget. Frozen UI settings remain frozen. Publish only complete candidates;
   choose one optimizer authority per active role, or explicitly reconcile
   versions. Concurrent CPU/GPU updates must not lose or double-apply learning.
6. Renderer adopts a validated candidate at an existing safe model-use boundary;
   only its owner publishes GPU buffers and saves GSFP. Spatial keeps its own
   set **inside the same profile**. Existing sets/donor files are preserved.
   Incompatible teacher/student feature semantics need explicit model versioning,
   not silent reuse of the same tensor dimensions. Preserve the eight-set cap.
7. Admission includes measured CPU work, readback/copy traffic, NPU runtime,
   memory bandwidth and package-power impact. A dedicated NPU is not hard QoS.
   Disable assistance if real/generated deadline tails or jitter regress.
8. Crash/hang/IPC error discards optional output and all stale generations.
   Teardown has no compositor wait on SDK shutdown; helper lifetime is independent
   and bounded. Avoid automatic repeated ten-minute compiles during gameplay.

## Validation and acceptance before live adoption

Keep stages independent, with a frozen model/profile and recorded identity:

- **Execution:** actual NPU partition/device, cache miss/hit, numerical drift,
  quantization behavior, two directions, supported and unsupported models.
- **Failure:** no device, missing permissions/SDK, CPU fallback, compile-only
  runtime, interrupted cache, helper crash/timeout, out-of-date/NaN/malformed
  results, duplicate completions, scene/profile resets and shutdown.
- **Quality:** paired held-out game/scene classes, actual future labels, static
  controls, boundaries/repeated textures/transparency, reversals and cuts;
  rendered RGB, flicker and real-frame correction plus blinded normal-speed play.
- **Contention:** NPU idle/busy versus equivalent GPU and CPU work under the
  actual game load. Report real/gen throughput, p95/p99 source age, missed slots,
  presentation jitter, memory/CPU usage and package energy/thermals. Separate
  native and nested modes; decision timestamps are not physical presentation.
- **Portability:** AMD890M/XDNA laptop and a real Intel NPU machine, plus no-NPU
  desktop. Mark unsupported/unmeasured paths explicitly. Generic architecture
  does not establish cross-vendor hardware certification.

Choose an experiment only when its paired quality/response benefit justifies
its compute and maintenance cost. Do not require a global mean win that hides
bad classes; either fix those classes or define a tested conservative eligibility
rule. Do not promote this module simply because raw inference/parity passes.

## Local validation on 2026-10-19

All 11 companion tests pass using the existing Ryzen AI venv, including actual
CPU training, ONNX export and ONNX Runtime CPU output parity. System Python
passes10 tests and skips the export test because ONNX is not installed there.
Failure tests exercise wrong/non-finite/stale-equivalent outputs, CPU execution
evidence rejection, failed-cache quarantine, helper failure and timeout. These
are isolated helper contracts, not compositor integration tests.

A discovery-only probe found `/dev/accel/accel0`, PCI 1022:17f0, `amdxdna`
2.25.260102.56.release_20260630 and ONNX Runtime 1.27.0 advertising VitisAI. No
NPU model compilation or NPU inference was run for this prototype, and Intel
hardware was not available. The existing SDK/prefixes, user profiles, renderer
and playable builds were not changed. GPU campaigns were not started or stopped.

One initial smoke-test premise was corrected: a one-step CPU training request
has zero learning rate at its only cosine-schedule step, so it exports a neutral
head. The test now uses two steps; the actual trainer was preserved.

## Campaign

The module's `experiment.py campaign --spec /absolute/campaign.json` absorbs
the fg-matrix NPU-stage orchestration: neutral and exact-key GPU-profile
held-out evaluation, CPU training and candidate evaluation, preparation bundles,
then sequential selected NPU runs. See the [spec and command](../tools/framegen-npu/README.md#campaign).
Captures come from `GAMESCOPE_FRAMEGEN_RECORD`, using the fg-matrix harness
`run.py record` phase or any motion-mode run; use frozen, separate train and
held-out directories. Evaluation/preparation stay in the gamescope Python
(NumPy plus ONNX for export); NPU execution stays in the configured SDK Python.

Hash receipts permit complete, consistent steps to resume. Stale or incomplete
directories stop without deletion. Per-step artifacts and `campaign.json` /
`campaign.md` preserve metrics, source-minus-neutral deltas, CPU training time,
NPU status/load/p50/p99 and raw-head parity. The single verified VAIML BF16
configuration is `tools/framegen-npu/amd-bf16.example.json`; reference its absolute
path (or a private copy) in both `config_file` and `config_files`.
These remain field-resolution offline metrics, not served RGB; temporal sigma
is residual variation, and NPU timing is the host call including copies.
`quality_validated` remains false. The orchestration has CPU-only contract
tests; adding it does not constitute a new hardware or quality result.

## Primary runtime references

- [AMD model compilation/deployment and cache rules](https://github.com/amd/ryzen-ai-documentation/blob/main/docs/modelrun.rst)
- [ORT CPU-fallback configuration](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)
- [ORT Python session/fallback/profiling API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [OpenVINO NPU device](https://docs.openvino.ai/2025/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- [ORT OpenVINO provider deprecation](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)
