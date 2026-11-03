# AMD-presenter DLSS 5 neural-rendering PoC

This opt-in, offscreen experiment runs the actual DLSS 5 neural-rendering
network on an explicitly identified AMD presenter GPU. It enhances images;
it is separate from DLSS-G interpolation and gamescope's GSFR training network.
It changes no compositor code, presentation defaults, live profiles or Steam
configuration. Native NVIDIA is a later provider, not a working fallback here.

The parent uses standard Python only. GPU identity and result semantics stay
independent of the compute vendor; the AMD adapter contains Wine/D3D12, ZLUDA,
HIP and NVAPI compatibility details. It does not advertise an unimplemented
universal per-job NR API: upstream's latest-result composition API supplies no
external-image retirement token, and cannot stand in for Pair/RawHead/training.

## Upstream projects and attribution

The runtime, NGX host, gate, Linux HIP bridge and optimized kernels come from
[RedDukeDev's DLSS 5 NR AMD Custom](https://github.com/RedDukeDev/dlss5-nr-amd-custom).
This adapter targets [release 1.3.2](https://github.com/RedDukeDev/dlss5-nr-amd-custom/releases/tag/1.3.2),
at [revision b5744cf03208f1442d63eeacc48eaf77a6bb4765](https://github.com/RedDukeDev/dlss5-nr-amd-custom/commit/b5744cf03208f1442d63eeacc48eaf77a6bb4765).
`prepare.py` adds gamescope-specific device and correctness guards to a private
copy of that project's runtime/harness; it does not implement the NR algorithm.
The external `nvngx_dlssnr.dll` network is NVIDIA's
[DLSS neural-rendering technology](https://www.nvidia.com/en-us/geforce/technologies/dlss/),
supplied separately under its own terms.

| Project | Role in this experiment |
| --- | --- |
| [RedDukeDev/ZLUDA](https://github.com/RedDukeDev/ZLUDA), forked from [vosen/ZLUDA](https://github.com/vosen/ZLUDA) | The upstream package's CUDA compatibility provider for AMD. Stock ZLUDA is not a qualified drop-in replacement. |
| [AMD ROCm/HIP](https://github.com/ROCm/HIP) | HIP 7 runtime used by the Linux bridge and AMD kernels. |
| [GE-Proton11-7](https://github.com/GloriousEggroll/proton-ge-custom/releases/tag/GE-Proton11-7) | Qualified Wine host and the source of the probe's DXGI/D3D12 DLLs. |
| [vkd3d-proton](https://github.com/HansKristian-Work/vkd3d-proton) | Proton's D3D12-to-Vulkan implementation. |
| [DXVK](https://github.com/doitsujin/dxvk), [DXGI interop declarations](https://github.com/doitsujin/dxvk/blob/master/src/dxgi/dxgi_interfaces.h) | Vulkan identity-query interface adapted in `presenter_identity.hpp`, under zlib/libpng. Upstream's optional DXVK 3.1.1 bundle is separate from the Proton DLLs used here. |
| [Khronos Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | SDK declarations copied, with their source headers and hashes, into the private build include tree. |

The upstream game injector is a [RedDukeDev fork](https://github.com/RedDukeDev/OptiScaler/tree/dlss5-nr)
of [OptiScaler](https://github.com/optiscaler/OptiScaler); this offscreen probe
uses the runtime harness instead. Upstream's
[third-party credits](https://github.com/RedDukeDev/dlss5-nr-amd-custom/blob/b5744cf03208f1442d63eeacc48eaf77a6bb4765/THIRD_PARTY.md)
also acknowledge clshortfuse's RenoDX colour-composition design, Bjorn Ottosson's
OkLab matrices, and countervolts' d4r super-resolution/command-list components.
Those credits do not establish that the separate upscaler components execute
in this NR probe.

[Third-party notices](THIRD_PARTY.md) retain the RedDukeDev and DXVK notices for
the adapted source portions. Private prepared copies preserve upstream's
`LICENSE` and `THIRD_PARTY.md`. External models, binaries and their dependencies
are not included in tracked gamescope source; these references do not change
their licences or grant redistribution rights for a runtime package.

## What is implemented

`prepare.py` requires clean upstream tag 1.3.2, revision
`b5744cf03208f1442d63eeacc48eaf77a6bb4765`, and creates a new private copy.
Original upstream and gamescope sources remain intact. The adapter:

- Selects the DXGI adapter by its actual Vulkan UUID **and** PCI address,
  through DXVK's documented interop declarations. Requires PCI property support.
- Enumerates the compute provider and requires exactly one matching PCI
  address. Vulkan and CUDA/HIP UUIDs are not treated as interchangeable.
- Requires FP16 output diagnostics before PNG conversion. NaN/Inf and malformed
  coverage/range reports fail. RGB changes are numerical, excluding signed zero
  and alpha-only changes. Disabled transport checks every uploaded half word.
- Explicitly pairs the linear FP16 upload with `DLSS5NR_ENCODING_LINEAR`.
- Requires fresh composed NR results and independently compares their raw RGB
  against a passed, input/device/source-matched disabled baseline.

`run.py` creates an isolated output directory and Wine prefix. It forbids runtime
CPU image-copy fallback with `DLSS5NR_COPY=0`, checks successful external-memory
imports and records input/model/helper/kernel/transport DLL/source hashes.
It explicitly requests upstream's Blackwell *model compatibility identity*
(`DLSS5NR_ARCH_NEURAL=0x1b0`); the physical AMD device is separately verified.
The package's default Ada identity can load the DLL but fails NR feature creation.

Process groups, the private prefix's Wine server, termination signals, timeout
and cleanup failure all have explicit handling. Existing offline signal handling
is reused. There is no automatic installation, translation-cache generation,
GPU selection by ordinal, game injection or compositor launch. Supplied source
and output trees cannot overlap; existing output directories are preserved.

Initial fixture upload and final FP16/PNG readback are **offline diagnostic I/O**.
They do not establish a readback-free live compositor path. Shared-image imports
prove this D3D12/Vulkan-to-HIP path on the tested configuration, not arbitrary
gamescope dma-buf formats/modifiers or external semaphore compatibility.

## Prepare and build

Prerequisites are a private, clean upstream checkout, Vulkan development headers,
MinGW-w64, CMake/Ninja, the verified 1.3.2 runtime/model/kernel/cache package,
GE-Proton11-7 and ROCm 7. The private Wine bridge uses a Proton-specific server
request; other Wine builds require their own qualification. Downloaded external
binaries stay outside tracked source and retain their own terms.

```sh
repowise distill 'python3 tools/framegen-compute/nr/prepare.py --source /path/upstream-1.3.2 --out /path/new-adapted-source'
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock env -u CPATH -u CPLUS_INCLUDE_PATH -u C_INCLUDE_PATH -u OBJC_INCLUDE_PATH CXXFLAGS=-I/path/new-adapted-source/vulkan-headers cmake -G Ninja -S /path/new-adapted-source/runtime -B /path/new-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=mingw.cmake'
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock env -u CPATH -u CPLUS_INCLUDE_PATH -u C_INCLUDE_PATH -u OBJC_INCLUDE_PATH cmake --build /path/new-build --target dlss5nr_harness gate ngx -j2'
```

The MinGW build must not inherit host Linux include paths. The preparation step
copies only required Vulkan SDK headers into a private include root and records
their hashes. CMake copies `nr-poc-source.json` beside the built helper.

Run a disabled control first, then the enabled correctness or asynchronous arm:

```sh
repowise distill 'python3 tools/framegen-compute/nr/run.py --proton /path/GE-Proton11-7-x86_64 --runtime /path/package/OptiScaler/dlss5nr --linux /path/package/linux --build /path/new-build/bin --hip-library /opt/rocm/lib/libamdhip64.so.7 --image /path/control.png --presenter-pci 0000:c7:00.0 --presenter-uuid 00000000c70000000000000000000000 --out /path/new-off --disabled'
repowise distill 'python3 tools/framegen-compute/nr/run.py --proton /path/GE-Proton11-7-x86_64 --runtime /path/package/OptiScaler/dlss5nr --linux /path/package/linux --build /path/new-build/bin --hip-library /opt/rocm/lib/libamdhip64.so.7 --image /path/control.png --presenter-pci 0000:c7:00.0 --presenter-uuid 00000000c70000000000000000000000 --out /path/new-on --baseline /path/new-off'
```

Substitute the actual presenter identity. The runner owns the existing shared
GPU campaign lock; do not acquire it again around the runner. `--async` omits the
NR network wait. The offscreen harness still waits for its ordinary D3D12 queue
submissions and sleeps between frames: its numbers are not live latency,
throughput or jitter measurements. The default `--wait` is correctness-only.
`--timeout` is 180 seconds by default (allowed 1–600).

Results are `result.json`, `stdout.log`, `stderr.log`, `cleanup.log`,
`output.png` and `output.png.fp16`. A passed result requires zero process and
prefix-cleanup errors. A RUNNING state, a changed PNG or an intended-sharing log
alone cannot establish success. The raw output is tightly packed RGBA16F linear
SDR. Match separate gray/channel controls before assessing an enabled image.

```sh
repowise distill 'flock -n .claude/reviews/tools/spatial-learning/gpu-owner.lock python3 tools/framegen-compute/nr/test_nr.py --source /path/upstream-1.3.2'
```

The normal CPU test needs no Wine/GPU/SDK runtime. Supplying `--source` also
checks anchors in the actual pinned source. Child cleanup tests create and
retire only their own Python process groups.

## Live integration boundary — proposal, not shipped behavior

Reuse `CVulkanTexture` references, dma-buf export metadata, existing GPU image
copy/barrier commands and shared timeline-semaphore ownership. Copy on the GPU
into a retained NR slot when holding a presentation-ring image through inference
would restrict ring reuse. Keep a single evaluation active and at most the newest
pending capture. Reuse framegen's deadline/admission decisions; skip NR under
pressure. Do not add a presentation scheduler or a growing work queue.

Two reuse constraints were verified against live gamescope:

1. `bFramegenShared` covers Vulkan queue families, not external HIP ownership.
   Ordinary exportable images may have optimal tiling with an unspecified DRM
   modifier. Require a proved format/layout/modifier/import combination and
   explicit external release/acquire; preserve every plane offset and stride.
2. `submitInternal()` consumes external `AddDependency`/`AddSignal` semaphores;
   dedicated `submitFramegenCommands()` currently does not. Its completion/reset
   gates also do not include external NR accesses. Use a submission path with
   the required external synchronization actually wired, and prove ordering
   using deliberately delayed producers/consumers before attaching live images.

An image, imported view, FD, semaphore and runtime must outlive every user.
Expiry/discard only removes usefulness. Upstream `process()` return, result age
and status counters are not retirement. Add an explicit provider retirement
mapping before handoff; until then retain to drained context shutdown. Native
NVIDIA should implement the same image/identity/lifetime contract using its
native runtime, with vendor SDK details confined to that adapter. Native GPU
execution and native Linux hosting are separate qualifications.

NR inference adds GPU load even when the CPU never waits. Admit measured work
within available presenter capacity and measure interference on p95 real-frame
latency and cadence. Keep compilation, model loading, allocation and optional
full diagnostics outside presentation. Use compact completion/age/error reports
for routine diagnosis; explicit diagnostic readback must remain opt-in.

Quality integration additionally needs trustworthy motion direction/scale,
capture timestamps, reset/cut identity, reprojection and disocclusion handling.
Current framegen estimates are not engine depth. Missing or unqualified tracking
cannot justify showing old neural images during movement. Preserve the current
frame and original HDR precision; qualify linear proxy/exposure/residual
composition separately from this static SDR test. Follow the five-POV review,
offline gates, runtime metrics, matched blind NR-off/on and real-game evidence
ladder before any default change.
