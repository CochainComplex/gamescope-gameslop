# Native presenter compute proof of concept

This is the first checkpoint toward a replaceable generation engine on the
presenter GPU. It is a standalone, offscreen Vulkan 1.2 conformance program.
It changes no gamescope presentation, shader selection or default.

The two implementations are phase-correct blend and analytic translation with
known displacement. They test the execution contract; neither is a neural model
or a claim of improved frame-generation quality. Vulkan provides the portable
generation implementation. The optional training plugin below adds HIP execution
of the current network's learning operations. The separate
[DLSS 5 neural-rendering experiment](nr/README.md) provides an offscreen AMD
runtime probe; it is not a live generation backend or native CUDA provider.
Its [upstream credits and dependency references](nr/README.md#upstream-projects-and-attribution)
identify the external runtime and the source portions adapted by the probe.

## Placement and ownership

The existing compositor already executes motion estimation, refinement and
generation on its selected `CVulkanDevice`. Its extra generation queue belongs
to that physical device. Selecting the second card as gamescope's presenter
therefore already puts this work there; another queue is not another GPU.

The PoC requires an exact nonzero Vulkan device UUID. It never picks ordinal 0
or interprets ordinal 1 as the presenter. PCI address is reported when available.
`--render-uuid` optionally checks a distinct, visible comparison device. This is
a declared topology, not evidence of which GPU an actual game uses. Inventory
can include software devices, but execution rejects them.

The small boundary consists of:

- `src/framegen/compute_job.hpp`: deterministic pair/phase validation, the
  float32 RGBA ABI and batch ownership. Causal generation is explicitly rejected.
- `vulkan_engine.*`: one device/context owner, allocations, descriptors,
  pipelines, submission and completion. No presentation scheduler.
- `generate.comp`: interchangeable generation methods using the same resources
  and parameter ABI, including arbitrary representable interpolation phases.
- `main.cpp`: independent synthetic expected values and adversarial checks.

`submit()` stages host inputs and records one batch. `poll()` only queries the
timeline; it never waits. GPU completion does not release ownership until
collection or retirement. Discard marks output unusable while retaining storage
until actual completion. Reset is rejected while any ticket owns resources and
advances the epoch afterwards. Tickets from old epochs/serials cannot affect a
new batch. The API is single-owner and is not thread-safe.

Mapped memory is flushed/invalidated where noncoherent; host/compute barriers
cover device reads and host readback. Standalone `shutdown()` drains and destroys
resources, then checks validation errors before reporting success. The destructor
provides fallback draining. A polling timeout does not cancel GPU access or
guarantee bounded teardown after a device hang; supervise probes as processes.

The fixed image contract is tightly packed float32 RGBA. Signed and greater-than-
one values and alpha are retained, with no transfer, gamut or tone conversion.
This checks arithmetic transport, not the compositor's HDR/display pipeline.
The translation method uses a known previous-to-current displacement in full
pixels and clamped bilinear sampling. It estimates neither motion nor depth.

## Common backend contract

`src/framegen/backend_contract.def` declares the SDK-free capability profiles
shared by C/C++ and `scripts/framegen_backend.py`. Negotiate once at session
creation, then keep the existing typed execution path:

| Profile | Execution and payload | Model state |
| --- | --- | --- |
| `generation` | Asynchronous float32 RGBA pairs | No learned state |
| `snapshot_training` | Asynchronous host training tensors, frozen native v1 ABI | Caller commits candidate outputs |
| `resident_training` | Asynchronous host training tensors, resident optimizer | Backend health-checks and publishes or rolls back the complete state |
| `npu_raw_head` | Blocking host raw-head inference in the isolated NPU worker | Inference only |

The contract states operation, execution mode, typed payload/model semantics,
state ownership, transport, optional diagnostics, exact-device support and the
one-job limit. Capabilities are promises by an adapter, not runtime placement
evidence. NPU execution still requires its existing provider/kernel evidence.
The NPU adapters do not promise training or exact-device selection.

`backend_bindings.hpp` returns the original concrete Vulkan owner or validated
native callback table by reference. The Python binding retains its provider and
stores original callables; cache `execute` outside an inference loop. Binding
adds no per-job wrapper, tensor conversion, queue, synchronization or scheduler.
Native references are borrowed: retain the owner or table/module/context through
actual retirement. Python bindings retain their provider; a binding is not a
second resource owner. Python wire validation accepts the same named uint32
tuple schema across the tools' intentionally isolated script imports.
Existing backend copies, polling and completion/retirement rules still apply.
Native HIP v1 binary layouts remain unchanged; an optional
`gsc_compute_get_contract(abi, size, payload)` exports the common descriptor.
Consumers accept a legacy module only after validating its exact frozen v1 table.

Use the typed payload that fits the operation. These profiles do not make NPU
inference interchangeable with online training, or caller-owned snapshots with
backend-owned publication. A future provider implements the matching payload
and lifetime rules; adding a runtime alone does not establish compatibility.
There is no CUDA provider or universal GPU/NPU image-sharing implementation.

The training receipt records `backend_contract` and `backend_contract_source`.
The resident replay binds the common contract by default; `--direct-interface`
skips cold binding for matched measurements and records `direct_interface` on
stderr. The measured job loop and 80-byte diagnostic report are identical.
This is an offline experiment; live presentation has no new control or wait.

## Build and check

From the repository root, with Meson, a C++20 compiler, Vulkan development files
and `glslangValidator` installed:

```sh
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock meson setup /tmp/gamescope-compute-build tools/framegen-compute'
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock meson compile -C /tmp/gamescope-compute-build'
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock meson test -C /tmp/gamescope-compute-build --print-errorlogs'
```

The normal Meson test runs CPU ownership/temporal contracts only. GPU execution
is explicit. Keep builds/tests separate from timing/A-B campaigns and retain the
shared campaign lock. Enumerate, then substitute an exact UUID from that output:

```sh
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock /tmp/gamescope-compute-build/framegen-compute-poc --list'
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock timeout --signal=TERM --kill-after=5s 30s /tmp/gamescope-compute-build/framegen-compute-poc --self-test --device-uuid PRESENTER_UUID --render-uuid RENDERER_UUID --validation'
```

Validation is optional but fails explicitly if its layer is unavailable. Preserve
the complete stdout/stderr through process exit; warnings and errors are evidence.
Tests include distinct channel/alpha signals, negative/HDR-range values, quarter
phases, partial workgroups, invalid inputs, stale tickets, discard, reset recovery,
buffer resizing and extreme-displacement border clamping. Synthetic oracles do
not establish natural-image quality. Coherent memory can leave the noncoherent
branch covered by source review only.

## Optional training plugins

`training_plugin.h` is a small, SDK-free native C ABI for one job: training the
current GSFR v3 network. It defines device identity, immutable tensor/state inputs,
candidate output and completion/retirement. The portable Python consumer `train.py`
loads an explicitly named shared library. It has no HIP import, automatic provider
selection, plugin registry or separate scheduler. A CUDA or other implementation
can implement this same ABI; none is supplied as a placeholder.

The HIP shared library is disabled by default. Enabling it needs an installed
HIP compiler/runtime and an explicit offload architecture. Building does not probe
or select a GPU. Example for a `gfx1150` device:

```sh
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock meson setup /tmp/gamescope-training-build tools/framegen-compute -Dhip_training=enabled -Dhip_arch=gfx1150'
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock meson compile -C /tmp/gamescope-training-build'
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock meson test -C /tmp/gamescope-training-build --print-errorlogs'
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock python3 tools/framegen-compute/train.py --plugin /tmp/gamescope-training-build/libframegen-training-hip.so --list-devices'
```

Execution matches the explicitly selected PCI address **and** the plugin's UUID.
It never substitutes another device or silently falls back to the CPU. Compare
the PCI address with the intended presenter's Vulkan inventory; device ordinals
and UUID encodings need not match between APIs.

The implementation follows the current online GLSL operations:

- The same 4,644-parameter `12→16→16→4` network and weight order.
- Sixteen deterministic 4×4 loss tiles, eight per direction, using raw checked
  fields and fast weights. Conditioning is zero, as in the current recorder.
- The same loss/evidence gradients, deterministic slice reduction, component
  clipping, Adam moments/correction, prior relaxation, fast-weight bounds and
  served EMA. This differs from the offline trainer's TV/global-norm policy.
- Immutable input snapshots; separate candidate fast/m/v/EMA and served arrays.
  No accepted optimizer state or prior is changed by a discarded result.

The context owns a nonblocking stream, pinned staging and device allocations.
Completion includes input copies, both kernels and output copies. Only a complete,
finite result can be collected. The caller commits it; discard does not free an
unfinished job. Destruction drains before releasing allocations, and a failed
drain retains the context/module. This host-staged offline path makes no latency
or zero-copy claim.

The first replay modes are causal **without shading history** and conservative
bidir. GSFD v1 has no third-frame history, scene-cut statistics, input-encoding key
or training-mode provenance. Shading requests are rejected. The ABI can explicitly
test a cut guard: it zeroes gradients while Adam momentum/decay still advances.
Bidir suppresses nonconfidence **gradients**; it does not mask optimizer updates.
Retained causal moments can therefore move nonconfidence weights. Start a fresh
state when changing replay mode; the CLI fixes the mode for each run.

To train recorded tensors, supply an exact bare GSFR v3 prior and ordered GSFD
files. Replace the example paths, PCI address and declared key with your inputs:

```sh
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock python3 tools/framegen-compute/train.py --plugin /tmp/gamescope-training-build/libframegen-training-hip.so --device-pci 0000:c7:00.0 --net /path/prior.gsfr --samples /path/fg_000001.bin /path/fg_000002.bin --input-key 0xf0 --mode causal --output /path/new-trained.gsfr'
```

The output must be a new path. The consumer reuses existing GSFD and GSFR codecs,
checks exact payloads/finites and exports **served EMA**, never fast weights.
Its receipt records consumed source/plugin hashes, the declared key, device,
optimizer controls and actual changed-weight counts. Each artifact is published
without overwrite; the model/receipt pair is not one atomic filesystem object.
GSFR has no Adam moments: reloading is a warm start, not exact continuation.

The key is a user declaration, not recovered provenance. Do not mix captures with
unknown encodings into a deployment profile or infer a quality improvement from
changed weights. A generated bare GSFR can use the existing
`GAMESCOPE_FRAMEGEN_NET` loading path after appropriate quality validation. This
tool never changes a live profile or enables HIP inside the compositor.

### Training validation

`check_training.py` holds the shared campaign lock and compares HIP with the
original online GLSL arithmetic, embedded verbatim by `make_training_reference.py`
under storage-buffer views. Both directions, every gradient slice, all optimizer
rows and served output are compared; test inputs include odd dimensions, zero and
nonzero heads, multi-step updates, cuts, retained momentum, component clipping,
weight bounds and copied-input ownership. The Vulkan reference uses the existing
standalone context and retirement owner, not another image pipeline.

```sh
repowise distill 'python3 tools/framegen-compute/check_training.py --plugin /tmp/gamescope-training-build/libframegen-training-hip.so --reference /tmp/gamescope-training-build/framegen-training-reference --device-pci 0000:c7:00.0 --vulkan-device-uuid VULKAN_UUID --out /path/new-parity-results'
```

The checker owns the lock; do not acquire it again around this command. Its
reference process is offscreen, supervised and separate from the HIP process.
This proves arithmetic/transport parity for the supported inputs. It does not
prove production Vulkan image/HIP imports, shading-history training, real-frame
priority or native presentation. Those remain separate acceptance work.

A live adapter must preserve renderer ownership of the activation key, immutable
prior, RNG, progress and checkpoints. It needs an explicit **served-only** upload;
reusing the activation upload would also overwrite the decay prior. No such live
adapter is included here.

## Cross-check of the earlier DLSS architecture

The earlier report is retained unchanged as historical evidence in
`.claude/reviews/tools/dlss-architecture-20261102/architecture.txt`. Review against
live gamescope and RedDukeDev revision
`89b590957a48250baad39ef5f7408824f95cb801` established these corrections:

1. Distinct adjacent raw images do not prove midpoint timing. The existing luma
   capture has no source timestamp sidecar; dropping duplicates destroys timing.
2. The proposed ingest completion token did not participate in the existing
   `framegen_work_completed()` busy gate. All shared-resource writers must be
   covered by one owner before a live adapter is safe.
3. Deadline expiry does not retire helper accesses or free its shared slot.
4. Upstream loads at most 16 captured input sets but can silently wrap them over
   more evaluations. It resets only evaluation zero; reset windows need separate
   runs. Stale output files cannot establish fresh execution.
5. Under Wine, host-copy defaults coexist with attempted texture sharing. Actual
   sharing and CPU synchronization modes need evidence from the selected path.
6. Production's first checked motion field is current-anchored previous-to-current
   displacement, so the audited DLSS convention requires `MV = -scale * field`.
7. Blanket GPU-support and licence conclusions exceeded the cited evidence.
   Runtime deployment targets and distribution terms require separate checking.

The audited DLSS-G path remains D3D12/NVAPI/HIP and reduces HDR colour to RGBA8.
Its count/index interface does not supply our arbitrary continuous phases. These
constraints argue against using that stack as the native core. A future DLSS
adapter remains a separate experiment, dependent on its runtime and model.
These are DLSS-G constraints. The requested DLSS 5 neural-rendering visual
upgrade uses FP16 colour/residuals and a different asynchronous composition API;
see [the AMD-presenter NR PoC](nr/README.md). Its strict offscreen probe leaves
the existing compositor and generation methods intact.

## Resident Vulkan training and compact diagnostics

`ResidentTrainer` is an experimental, offscreen Vulkan owner. Its optimizer
state and served model remain in the selected backend between jobs; normal
collection copies only an **80-byte** `TrainingReportWords` record. It reuses
`VulkanEngine` and the original generated training/Adam shader bodies. It does
not replace the compositor's online trainer or change its persistence/defaults.

The three storage buffers contain input tensors/prior, gradient slices, and two
five-row banks plus the report. Training writes the candidate bank. A portable
64-thread reduction checks every logical parameter in fast weights, moments,
EMA and served weights using binary32 exponent bits. A subsequent publication
pass copies the whole healthy candidate into the accepted bank, or restores the
whole candidate from the preceding accepted bank. The completion token covers
both validation and publication/rollback. The accepted served row is at `9P`;
the candidate served row at `4P` must never be bound for inference.

There are no new norm/range rejection thresholds. The report contains coverage,
non-finite state/served counts, changed served representations, maximum absolute
served weight/delta, model epoch, input key, optimizer step, attempt nonce and
accepted step. Delta overflow saturates the diagnostic at the largest finite
float; diagnostics cannot reject an otherwise finite candidate. Every report
word is overwritten, including reserved words. CPU decoding requires exact
identity, complete coverage and a consistent verdict. A malformed report
quarantines the context. Rejection restores all moments as well as weights, so
the next attempt reuses the unaccepted Adam step; the attempt nonce still moves.

Contexts have a fixed shape/key and a unique nonzero model epoch. One job may be
outstanding. Methods serialize ownership; `checkpoint()` refuses outstanding
work and holds ownership through invalidation/copying. There is no cancellation
API that pretends queued GPU publication can be undone. Shutdown is explicit;
the destructor is a standalone fallback drain, never a live presentation hook.

`checkpoint()` explicitly downloads the accepted `[fast,m,v,EMA,served]` bank
(92,880 bytes). `checkpoint(true)` also downloads the 16 gradient slices.
Gradient diagnostics may contain non-finite values which Adam sanitized; the
accepted model must remain finite. Full diagnostics plus the normal report read
390,176 logical bytes per job. Partial reads still invalidate an allocation-wide
range when its mapped memory is noncoherent. Backend residency here means no
CPU optimizer-state round trip, not guaranteed device-local VRAM: this standalone
owner uses host-visible buffers and still stages the captured input tensors.

The SDK-free wire layout lives in `src/framegen/training_report.h`, with CPU
admission in `training_report.hpp`. HIP/CUDA can implement that report later;
the existing HIP v1 ABI remains unchanged. There is no runtime registry or
unverified cross-API image sharing.

Build using the standalone commands above. CPU report contracts are part of
`meson test`; GPU checks remain explicitly invoked on a selected hardware UUID:

```sh
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock /tmp/gamescope-compute-build/framegen-resident-health --device-uuid UUID --validation'
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock /tmp/gamescope-compute-build/framegen-resident-owner --device-uuid UUID --validation'
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock /tmp/gamescope-compute-build/framegen-resident-training --device-uuid UUID \
  --input STEP1.input --input-key 240 --output-dir NEW_DIRECTORY \
  --steps 32 --warmup 0 --check-reference --validation'
```

The replay consumes the same GSCT `.input` arithmetic fixtures emitted by
`check_training.py`. It requires a step-1 initial state: fast/EMA equal the
prior, zero moments. `--check-reference` compares the evolving accepted bank and
all gradients bit-for-bit with the original Vulkan arithmetic after every job;
it is conformance, not timing. `--full-diagnostics` includes an explicit full
snapshot in each measured job. `--warmup` evolves the same optimizer before the
`--steps` measured jobs, so both comparison arms need identical counts.

CSV records acceptance, changed/non-finite counts, maxima, submit/wait/collect/
snapshot/total milliseconds and exact epoch/key/attempt/accepted-step identity.
Setup and teardown are separate stderr metadata. Checkpoints at steps
1/7/1023/1024/1025 when reached, plus the final checkpoint, are diagnostic raw
five-row float files with identity JSON sidecars, written only after checked GPU
shutdown. They are not atomic GSFR/GSFP profiles. For a normal served-model
export, pass the last row through the existing codec/publication path; GSFR
reload remains a warm start rather than optimizer-state continuation.

Live integration still requires preserving GPU-free emergency-exit snapshots,
outgoing-key ownership and short-session checkpoint/reset behavior. Compact
diagnostics cannot substitute for those existing CPU-cache guarantees. Offscreen
health tests and readback timings do not establish native pacing or visual quality.

## Integration and optional CUDA/HIP work

For a future compositor adapter, borrow the existing presenter device, local
history images, output acquisition and `submitFramegen` retirement owner. Do not
instantiate this standalone device or host-staging pipeline in the compositor.
Keep admission, client commits and presentation feedback with their current owner.
Real presentation must never wait for another runtime.

CUDA/HIP modules may own a matched context/stream and imported resources. The
boundary should permit another generation implementation; it should not yet grow
a generic job registry or scheduler for unspecified future workloads. Before adding
an adapter, prove exact physical-device matching, allocation/format/stride
compatibility, memory visibility and supported cross-API semaphores. An API return
is not proof that GPU reads have retired. Another queue or runtime does not imply
freedom from memory, bandwidth or shader contention.

Next checkpoints are native image interoperability on the presenter, an actual
replacement algorithm/model, and independent synchronization review. Then follow
instrument parity, held-out offline quality, full metrics, matched blind comparisons
including framegen OFF, and real-game confirmation. Latency, cadence, continuity,
integrity and sharpness remain separate acceptance factors. Offscreen conformance
does not qualify Wayland arbitration, native DRM/VRR/HDR, physical input latency,
or actual render/present topology.
