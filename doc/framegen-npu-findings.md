# NPU findings and fast-path roadmap

Status: 2026-10-19. Results of the first NPU campaign with the offline module
[`tools/framegen-npu`](../tools/framegen-npu/README.md), plus a source-level study of how the NPU
could take work off the iGPU. The design rules, module boundaries and acceptance gates are in
[framegen-npu-experiment.md](framegen-npu-experiment.md); this page holds what was measured and
what to try next. There is still **no renderer integration and no demonstrated quality or pacing
benefit**.

Labels: **[measured]** taken on this machine and reproducible from the files named;
**[cited]** from an external source or source code; **[est]** arithmetic, not measured.

## 1. Summary

- **The NPU path works end to end.** A refiner trained on the GPU (or on the CPU) is exported to
  ONNX, compiled for the NPU in BF16 and executed there. All four campaign models ran fully on the
  Vitis AI provider (404 profiled kernel events per run, no CPU fallback) [measured].
- **Today it is slower than the GPU.** One refiner call takes p50 ~1.5 ms / p99 1.8–2.6 ms on the
  NPU, against ~0.6 ms for the same net as a shader on the 890M and 0.79 ms on 8 CPU threads
  [measured]. The NPU frees the GPU but adds latency.
- **The cost is the software path, not the hardware.** ORT plus host copies dominate; AMD documents
  an 84 µs dispatch round trip [cited], so a native path could reach ~0.2–0.6 ms [est].
- **The goal is GPU headroom, not CPU offload.** Games leave few free CPU cores, so the NPU is the
  only spare engine. The realistic first win is moving the refiner's inference off the iGPU:
  ~0.9 ms per real frame at 1440p, ~1.8 ms at 4K [est].
- **Online learning is the hard conflict.** The NPU compiler bakes weights in as constants and a
  recompile takes ~6 min, so `GAMESCOPE_FRAMEGEN_NET_ONLINE` cannot run on the NPU as built
  today. A frozen NPU trunk with a GPU-trained last layer keeps it (section 7, idea 2).
- **No quality verdict yet.** The first held-out comparison was invalid (section 4); the captures
  were re-recorded and the comparison is being rerun.

## 2. Hardware and software stack

| Part | Version on the test laptop |
|---|---|
| NPU | AMD XDNA2 (Strix, aie2p 6x8), `/dev/accel/accel0` |
| Driver / runtime | amdxdna 2.25.260102.56 (DKMS), XRT 2.25.37, NPU firmware 1.1.2.64 |
| Kernel | 6.17 OEM |
| SDK | Ryzen AI SW 1.8.0 (venv), ONNX Runtime 1.27 with Vitis AI EP, VAIML BF16 config |
| iGPU | Radeon 890M, RADV (Vulkan 1.4) — presents and generates |
| Render GPU | NVIDIA (game) |
| Power mode | `default` (`xrt-smi examine -r platform`); changing it is root-only |

The runtime only executes when `/opt/xilinx/xrt/lib` precedes the SDK's bundled XRT on
`LD_LIBRARY_PATH` (the bundled 2.19 `libxrt_coreutil` otherwise hides XRT 2.25 and the runtime
silently compiles without running). The module's AMD config template and the single BF16 pass
config are `tools/framegen-npu/amd.example.json` and `amd-bf16.example.json`.

## 3. Measured results (campaign-1c)

Workload: the production refiner net (3 convolutions 12→16→16→4, 3×3 kernels, ~4.6k parameters)
at field resolution 248×152 (output 2048×1280), one field direction per call, 400 calls per model
(100 per direction × 4 fixture directions), ORT profiling on. Source:
`.claude/reviews/tools/fg-matrix/npu/campaign-1c/campaign.json` [measured].

| Model | p50 | p99 | Max abs deviation NPU vs fp32 | Compile + load (first time) |
|---|---|---|---|---|
| GravityMark, GPU-trained learned | 1.48 ms | 1.77 ms | 0.027 | 366 s |
| GravityMark, CPU-trained (20k steps) | 1.49 ms | 2.48 ms | 0.075 | 362 s |
| Xonotic, GPU-trained learned | 1.55 ms | 2.57 ms | 0.046 | 362 s |
| Xonotic, CPU-trained (20k steps) | 1.48 ms | 2.49 ms | 0.107 | 362 s |

Same net, same input size, other engines [measured]:

| Engine | p50 | p99 | What the time covers |
|---|---|---|---|
| NPU (BF16, ORT Vitis AI) | ~1.5 ms | 1.8–2.6 ms | host call incl. copies |
| CPU, ORT 8 threads (fp32) | 0.79 ms | 0.92 ms | host call |
| CPU, ORT 1 thread (fp32) | 3.27 ms | 5.49 ms | host call |
| 890M, production shader | ~0.6 ms (interpolated) | – | GPU timestamp of the pass only |

The 890M shader costs 0.545 ms at 1080p, 1.057 ms at 1440p and 2.140 ms at 4K
(`GAMESCOPE_FRAMEGEN_BENCHMARK=1`, `motion net refine (per real)`). NPU/CPU times include host
overhead and GPU times do not, so the comparison is indicative only.

Other campaign facts:
- **Parity tolerance:** the module's default (atol 0.001) is not reachable in BF16. Runs passed at
  atol 0.15 / rtol 0.1. The raw-head error of 0.03–0.11 corresponds to up to ~0.15 texel after the
  bounded residual [est]; whether that is visible must come from the offline gate, not from parity.
- **CPU-trained weights drift more under BF16** (2.3–2.8× the GPU-trained error).
- **CPU training cost:** 20,000 NumPy steps take ~70 s on one core per benchmark.
- **The original module quarantined a compiled cache after a parity miss.** Future runs preserve
  compile success separately: `ready` is never reset to `building`, and a failure records its
  reason plus whether the compile is reusable. Parity failures include `max_abs_error` in
  `result.json`; a successful compile can be measured again with a new run directory.
  Results and campaign tables label `load_ms` as cold or warm (`cache_state_before`).
- **Fixture coverage:** only 3 of 4 fixture directions were distinct per bundle; the duplicate check
  only rejects four identical directions. Future bundles select the highest mean-motion pair and
  a distinct pair with the highest mean confidence, each in both directions, rather than the
  first/last capture frames. This improves coverage without establishing image quality.

Module follow-up: see the README for
[cache states and subprocess lifecycle](../tools/framegen-npu/README.md#explicit-npu-execution)
and [campaign timeout budgets](../tools/framegen-npu/README.md#campaign).

## 4. Data lessons (why the first quality comparison was invalid)

Found by the module review (`.claude/reviews/npu-module-review-2026-10-19.md`):

1. **Frozen captures.** The first Xonotic captures held 850/1000 static samples: 1000 samples at
   ~60/s cover only start-up and map loading. 703 of 800 neutral held-out frames had zero
   confidence.
2. **Held-out not independent.** Training and held-out captures were separate runs of the same
   deterministic scene, so held-out frames matched training frames.
3. **`conf-resid` flattered the baseline.** Frames whose confidence sums to zero counted as 0
   error, which favours the untrained field.

Fixes now in place:
- The harness records **one 120 s run per benchmark** (`run.py record`), drops start-up, takes
  training (20–60 s) and held-out (80–110 s) from **separate time windows**, removes static
  samples and refuses a capture with too few moving samples.
- Windows use each sample's publish time, not its index. `GAMESCOPE_FRAMEGEN_RECORD` samples
  on every new motion preparation, including cheaper motion rungs; extrapolation is excluded.
  It is not restricted to rung 0. The observed load-dependent ~7–17 samples/s does not by itself
  identify the bottleneck: the single readback slot and synchronous disk writes need counters
  for eligible/prepared/copied/flushed samples and skipped reasons before changing the ring.
  Its header sequence number is a GPU submission counter, not a frame counter.
- `scripts/framegen-net-eval.py` computes `conf-resid` only over frames with confidence and
  reports how many were skipped; if no frame has confidence the value is NaN (refused by the
  campaign) instead of 0.

## 5. Where the iGPU time goes, and what can move

Per real frame, 890M, 1440p, learned x2 [measured unless marked]:

| Work | GPU time | Frequency | Move to the NPU? |
|---|---|---|---|
| Online training (backprop + Adam) | part of 1.2–3.1 ms total net cost (~0.15–2 ms by subtraction [est]) | per real frame | Not as built (no gradient path). Shrink it with idea 2. |
| Refiner inference | 1.06 ms (4K: 2.14) | per real frame | **Yes**, the first target (idea 1). |
| Motion setup (pyramid, block matching, FB check) | 0.75 ms | per real frame | No: SAD search is not MAC work, and a ≥0.3 ms round trip makes it a loss. |
| History copies | ~0.4 ms | per real frame | No (memory transfer). |
| Stats / quality probes | ~0.05 ms | per real frame | Not worth it. |
| Warp | 0.53 ms each | per generated frame | No: full resolution, latency-critical, HDR path. |
| Composite, HUD, present | – | per output frame | No. |

Removing the ~1 ms refiner dispatch also removes a non-preemptible compute pass from the queue that
composites and flips, which helps frame-time consistency beyond the average.

Keep pass cost separate from added queue latency: the internal Xonotic A/B report
shows Spatial-frozen adding **4.25/4.5 ms** over framegen off: `src_flip_ms_p50`
5/6 → 9.25/10.5 ms (ref/tip). These are medians of two valid 40 s repetitions on a
headless 2048×1280@120 host, source 60 fps, x2. The internal run log
identifies the AMD 890M compositor and RTX PRO 500 game renderer. Noise is material: Learned-frozen is
6.5/7.25 ms, Checked changes from 7.5 to 5 ms. Source-ready queue residence is not input
latency; offloading the refiner alone does not establish that this delay disappears.

## 6. Why the NPU is slow today, and the fast-path enablers

### 6.1 Overhead
The measured ~1.5 ms is an ORT plus host-copy number. Estimated split [est]: ~1.2 ms ORT and
copies, the rest dispatch and compute. Floor estimate for this net: dispatch ~84 µs [cited] +
compute ≥0.05 ms + I/O ~0.05 ms ⇒ **~0.2–0.3 ms**; nothing in the repo measures below 1.47 ms yet.

### 6.2 Power and clocks
- Boot DPM is the lowest level; `turbo` disables clock gating; raising DPM needs ~3.5 s to settle
  (`aie2_pm.c`) [cited]. `DRM_AMDXDNA_SET_STATE` is root-only, so gamescope cannot set it.
- Calls arrive every 8–17 ms in a game, so **time at game cadence**, never in a back-to-back loop
  (clock-gating wake-up is hidden there). Turn ORT profiling off for latency runs
  (`backends.py` enables it).

### 6.3 Skip ORT
The SDK ships the FlexMLRT client (`flexmlrt/include/FlexMLClient.h`, `libflexmlrt.so`) that the
Vitis AI EP drives. It can load the existing compiled cache without ORT, accepts tensors backed by
`xrt::bo` (`setBo`, `skipPrePostProcessing`, `skipBoSync`) and has blocked layouts
(`HCWNC8/HCWNC4`) to match when a shader writes the input directly. Cheaper stop-gap: ORT IO
binding with pre-allocated buffers.

### 6.4 Zero-copy transport (APU: one physical RAM)
1. **4 KiB-aligned host allocation imported everywhere** — Vulkan `VK_EXT_external_memory_host`
   (`minImportedHostPointerAlignment` 0x1000 on the 890M [measured]), XRT `xrt::bo` user pointer,
   OpenVINO NPU CPU-VA import. One allocation for AMD, Intel and Vulkan.
2. **dma-heap / udmabuf + dma-buf** — Vulkan `VK_EXT_external_memory_dma_buf`; amdxdna imports
   foreign dma-bufs (mainline since Linux 6.16 [cited]); public entry `xrt::bo(device, fd)`.
3. Export from the NPU side is limited (user-pointer objects are no longer exportable).
Imported buffers must not be cache-synced by the driver (newer amdxdna returns -EOPNOTSUPP); x86
DMA is coherent. **Untested:** RADV and amdxdna importing the same memory together.

### 6.5 Synchronization
- **GPU → NPU:** amdxdna has no input fences or implicit sync (only `EXEC_BUF` is implemented).
  Use one sleeping waiter thread: wait on the framegen timeline semaphore, then start the NPU
  run. It costs a wake-up, not a core.
- **NPU → GPU:** the kernel signals a per-context timeline syncobj, and a drm_syncobj fd can be
  imported as a Vulkan timeline semaphore. XRT 2.25 does not expose that kernel fence publicly,
  so completion still passes through an XRT thread unless the ioctls are issued directly (later).
- A helper process (SDK isolation, as the design doc requires) passes buffers and fences by fd
  (SCM_RIGHTS); cost ~50–100 µs per extra hop [est].

### 6.6 Batching and contexts
Run both field directions in one dispatch (batch 2). Other NPU clients time-slice the array, so
benchmark only on a quiet NPU; the context QoS struct carries priority and latency hints.

### 6.7 Memory bandwidth
Not a constraint for these nets: ~1–3 MB per call, <1 GB/s at 60 real fps, against ~96 GB/s
measured iGPU bandwidth and ~50 GB/s effective NPU bandwidth [cited]. Power and thermals (shared
SoC budget) and NPU preemption are the real contention risks.

## 7. Ideas, ranked by iGPU time freed

| # | Idea | GPU ms freed / real frame | Latency added | Feasibility | Main risk |
|---|---|---|---|---|---|
| 1 | **Frozen refiner inference on the NPU** — native stack, zero-copy, both directions in one dispatch, GPU fallback | ~0.9 (1440p), ~1.8 (4K) [est] | +0.3–0.6 ms; fits the 4–6 ms slack at 120 Hz, not at 240 Hz | now / soon | weights frozen; sync via waiter thread |
| 2 | **Split net** — frozen trunk on the NPU, last layer trained online on the GPU | as #1 plus part of training (0.1–1.5) [est] | as #1 | soon | adaptation capacity drops from 4.6k to ~0.6k parameters |
| 3 | **Large flow/confidence net, one pair late** — search seed or reverse-field provider | 0–0.3; mainly a quality lever | none (late by design) | soon (offline), research (live) | stale-seed bias |
| 4 | **NPU gradients** for online training | the rest of training (0.1–2) [est] | none (async) | research | custom AIE kernels, maintenance |
| 5 | **Advisory outputs** — scene cut, scene class, HUD mask | ~0 | none | soon | no quality evidence |

Notes:
- **Idea 1** uses `GAMESCOPE_FRAMEGEN_NET_PROFILE` or a `_NET` blob; it cannot serve
  `NET_ONLINE`. Bidir is the easiest first target: its confidence-only net and delay `D` give extra
  slack.
- **Idea 2** keeps online learning. The last layer is 12.5% of the inference MACs and is where
  adaptation matters most; bidir training already freezes the trunk. The NPU returns the
  16-channel activation (1.2 MB in BF16) and the GPU runs and trains only the last layer.
  Alternative: make weights graph inputs (fixed im2col conv + `MatMul` with dynamic weights) if
  VAIML accepts it; one probe compile decides.
- **Idea 3** is where the NPU has headroom the iGPU cannot afford (a 174k-parameter
  encoder/decoder ran at 7.76 ms on the NPU in the spike). Same-pair use would cost 3–8 ms [est]
  and exceed the slack; one pair late it costs no latency.
- **Idea 4:** the SDK is inference-only; mlir-aie has GEMM and int8 1×1 conv examples but no 3×3
  bf16 conv or backward kernels; the only published NPU training port offloads GEMMs and pays
  per-call overhead (arXiv 2504.03083). Defer until ideas 1–2 prove the transport.

## 8. Pipeline design for idea 1

```
real frame N lands
  GPU cmdbuf A: composite as today, then pyramid / match / FB check, pack into the NPU input buffer
                --signal T(N)-->
  waiter thread: wait T(N) -> NPU run (forward + reverse) -> signal U(N)
  GPU cmdbuf B (first slot): wait U(N) or deadline -> unpack, stats, train, warp
  deadline passed -> run the existing GPU net pass for this pair (same weights), count npu_miss
```

Rules:
- One NPU job in flight and one queued, latest wins.
- A pair uses either the NPU or the GPU field, never both; one immutable decision per pair.
- The framegen queue never blocks inside a command buffer (timeline wait at submit time).
- Pair id, model hash and reset generation travel with every job.
- Shared-queue devices pay a second submit; evaluate dedicated-queue and shared-queue setups.
- Slack at 60 real fps → 120 Hz output: one output period (8.33 ms) minus setup (~0.75 ms), warp
  (~0.53 ms) and the commit lead ⇒ ~4–6 ms [est]. At 240 Hz output only ~1.5–2 ms: stay on the GPU.

## 9. Intel parity

OpenVINO's NPU plugin imports dma-buf fds and 4 KiB-aligned CPU pointers but documents no
synchronization primitive, so Intel is CPU-mediated like AMD. Level Zero's mutable command lists
may lower dispatch latency. No Lunar Lake latency figures for small nets were found; the 0.3–0.6 ms
target is unverified there. The shared host allocation (6.4 option 1) is the common transport.
Desktops without an NPU (7900 XT / 5700 XT) keep the GPU path; the feature stays optional.

## 10. Experiment plan

Standalone under `tools/framegen-npu/native/`, no gamescope changes, quiet NPU only.

| Step | What | Success | Effort |
|---|---|---|---|
| **E-A** | ORT baseline at game cadence: 2000 calls with 8.3 / 16.7 ms between calls, profiling off, IO binding on; power modes default / performance / turbo; split copy vs `session.run` | learn which of copy, ORT, dispatch or clock-gating wake dominates | ~30 min |
| **E-B** | Native FlexMLRT call against the existing compiled cache (no recompile): host buffers first (parity vs ORT, max abs ≤ 0.075), then `xrt::bo` user-pointer buffers with `skipPrePostProcessing`/`skipBoSync` | p50 ≤ 0.5 ms and p99 ≤ 1.0 ms at 8.3 ms cadence; no-go if p50 > 1.0 ms | ~1 day |
| **E-C** | One 4 KiB-aligned buffer imported into Vulkan and the NPU (then the dma-heap variant); GPU writes, waiter thread runs the NPU, GPU reads; plus GPU baselines: current shader and a `VK_KHR_cooperative_matrix` version | both imports work; submit-to-release p99 ≤ 1.5 ms with zero copies | ~1 day + short GPU window |

**Go/no-go:** continue only if E-B and E-C pass **and** the best GPU baseline stays ≥ 0.7 ms (a
cooperative-matrix refiner at ≤ 0.4 ms would shrink the saving to ~0.3 ms and not justify the
complexity). After a go: bidir confidence net on the NPU behind an env flag, through the
learned-path ladder (parity → offline gate → METRICS → blind A/B → real games), then idea 2.

## 11. What not to do

- Never put NPU work on the current slot without a deadline and an identical GPU fallback, and
  never block the compositor thread on the NPU, XRT or SDK shutdown.
- Never time with back-to-back loops, with profiling on, or while another NPU client runs.
- Do not rely on `NET_ONLINE` with a VAIML-compiled net, and do not recompile per weight update.
- Do not quantize the refiner to int8 without the offline gate; flow residuals are sub-texel.
- Do not start with hand-written IRON conv kernels (no bf16 3×3 or backward examples; unstable API).
- Do not move block matching, warp, composite or any full-resolution pixel work to the NPU (HDR
  rule, bandwidth, latency).
- Do not assume implicit sync or cross-process fences work without a test.
- Do not cache-sync imported buffers; do not assume gamescope can set the NPU power mode.
- Do not link ORT, Vitis AI or OpenVINO into gamescope or ship the SDK; keep the optional helper
  boundary from the design doc.
- Do not claim a benefit before the learned-path ladder; this page holds no quality evidence.

## 12. Open questions

- Real split of the 1.5 ms (E-A) and the native floor (E-B); `xrt-smi validate --run latency` in a
  quiet window gives the local dispatch floor.
- RADV + amdxdna sharing one buffer; an XRT fence imported as a Vulkan timeline semaphore.
- Whether VAIML accepts runtime weights (dynamic `MatMul`), which would remove the frozen-weights
  limit.
- CPU-trained vs GPU-trained quality on independent held-out data (rerun with the new captures).
- Separate GPU cost of online training vs inference in a real run (per-pass timestamps).

## 13. Sources

- Campaign results: `.claude/reviews/npu-campaign-1-report.md`,
  `.claude/reviews/tools/fg-matrix/npu/campaign-1c/campaign.{json,md}`.
- Module review: `.claude/reviews/npu-module-review-2026-10-19.md`.
- Fast-path study with full citations and source line references:
  `.claude/reviews/npu-fastpath-ideas-2026-10-19.md`.
- Earlier spike: `.claude/reviews/npu-framegen-spike-2026-10-17.md`.
- AMD Ryzen AI `xrt-smi` (latency test, 84 µs sample): https://ryzenai.docs.amd.com/en/latest/xrt_smi.html
- amdxdna driver: https://github.com/amd/xdna-driver ; mainline
  `drivers/accel/amdxdna/amdxdna_gem.c`; Linux 6.16 dma-buf import/export:
  https://www.phoronix.com/news/Linux-6.16-DRM-Misc-Next-1
- OpenVINO NPU remote tensors: https://github.com/openvinotoolkit/openvino/blob/master/docs/articles_en/openvino-workflow/running-inference/inference-devices-and-modes/npu-device/remote-tensor-api-npu-plugin.rst
- Intel NPU driver overview: https://github.com/intel/linux-npu-driver/blob/main/docs/overview.md
- mlir-aie: https://github.com/Xilinx/mlir-aie
- NPU training port (GPT-2, IRON): https://arxiv.org/html/2504.03083v1
- XDNA2 GEMM performance and NPU bandwidth: https://arxiv.org/html/2512.13282v1
- 890M bandwidth: https://chipsandcheese.com/p/amds-radeon-890m-strix-points-bigger-igpu
