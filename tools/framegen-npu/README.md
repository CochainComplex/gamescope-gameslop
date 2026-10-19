# Experimental NPU companion

Offline prototype, disabled by default. This directory is independent of the
renderer and build dependencies. Gamescope never imports it, launches it, links
an SDK, or waits for it. Installing an NPU SDK does not change gameplay.

Read [the design and review](../../doc/framegen-npu-experiment.md) before extending
this into runtime code. There is **no demonstrated quality or pacing benefit**.

Implemented:

- CPU training through the existing NumPy trainer, or explicit selection of an
  existing GSFP role/encoding set; exports to a **new artifact directory**.
- ONNX export of that refiner's raw head, with real GSFD feature fixtures for two
  pairs in both directions and NumPy reference outputs. Opset17, embedded weights,
  static shapes, 3-texel replicated input halo, three valid convolutions. This
  follows the offline evaluator; production shader/full-color parity is separate.
- AMD ONNX Runtime/Vitis AI and Intel **native OpenVINO NPU** adapters. Native
  OpenVINO avoids depending on the deprecated built-in ORT OpenVINO provider.
  Intel is an implemented adapter, **not hardware-validated support**.
- Separate SDK interpreter/environment, process-group timeout, local failure,
  strict NPU execution checks, model/runtime/driver/config cache identity and
  failed-cache quarantine. No automatic SDK installation or library replacement.
- Raw-head numeric comparison on every measured call, explicit tolerances,
  timing and provenance JSON. These are execution/parity tests, not quality gates.

Not implemented: NPU gradients, NPU teacher distillation, automatic CPU learning
in games, live weight publication, motion priors, masks, classifier policy,
Vulkan/dma-buf transport, quantization or model hot swapping. CPU learning and NPU
inference are separate stages. This does not offload today's GPU online updates.

## CPU contracts

```bash
python3 -m unittest discover -s tools/framegen-npu -p 'test_*.py' -v
```

NumPy required. The CPU train/export test additionally requires `onnx` and runs
CPU ONNX parity when `onnxruntime` is available. It never requests an accelerator.

## Configure and probe

Copy `amd.example.json` or `intel.example.json` into a private config and replace
the placeholders with **absolute paths**. Commands below run from the repo root.
Probe imports the selected SDK in a child and lists candidates, without creating
a model session or compiling anything:

```bash
python3 tools/framegen-npu/experiment.py probe \
  --config /absolute/private-npu.json --out /tmp/npu-probe-unique
```

An advertised provider or accelerator node does not prove support for a model.
The `available_is_not_execution` marker is deliberate. A missing SDK, inaccessible
device, crash or timeout fails locally; the compositor continues unchanged.

AMD: set `XILINX_XRT` and the child's `LD_LIBRARY_PATH` in the config **before**
launch. On this machine `/opt/xilinx/xrt/lib` must precede the SDK's bundled XRT.
The single verified VAIML BF16 pass configuration is
[`amd-bf16.example.json`](amd-bf16.example.json). Point both
`provider_options.config_file` and `config_files` at that file's **absolute path**
(or a private copy), as shown in `amd.example.json`, so its contents participate
in cache identity. Do not export these variables into your
game, desktop, playable launch script, or dependency prefix. SDKs are user-owned;
this tool does not redistribute them.

`runtime_tag` must identify SDK, XRT/plugin, driver and firmware versions. Update
it whenever any changes. Runtime version and available Linux accelerator driver
identity are collected too; they cannot discover every indirectly loaded SDK
library or firmware change. List any additional config files in `config_files`.

## Prepare recorded data on the CPU

Use copies of captures. To evaluate existing Spatial weights, select its exact
set; output metadata must match that key. The source GSFP is only read:

```bash
/absolute/onnx-venv/bin/python tools/framegen-npu/experiment.py prepare \
  --data /absolute/recorded-capture --net /absolute/copied-game-profile.bin \
  --net-key 0x100f0 --profile-key 0x100f0 --out /tmp/spatial-npu-bundle-unique
```

For a fresh CPU-learned experiment instead:

```bash
/absolute/onnx-venv/bin/python tools/framegen-npu/experiment.py prepare \
  --data /absolute/recorded-capture --train-steps 200 --crop 32 \
  --profile-key 0x100f0 --out /tmp/cpu-npu-bundle-unique
```

The preparation interpreter needs NumPy and ONNX. CPU training uses one BLAS/OMP
thread and one crop batch, with a separate process and `--training-timeout`.
This remains real CPU work and shared-memory traffic; do it outside pacing
measurements. The existing offline trainer starts neutral and does not resume
optimizer/online progress; `--net` and `--train-steps` are mutually exclusive.
Use at least two steps: the trainer's final cosine-schedule step has zero LR.

Capture directory needs at least two GSFD files, same resolution, large enough
for `crop + 6`. The CPU trainer uses its existing final-tenth validation split;
that is not held-out-game validation. The two fixture pairs are execution tests,
not a representative quality corpus. Current mode0 endpoint features do not
provide a causal future-motion teacher.

`candidate.gsfr` is an experimental standalone blob plus `profile_key` metadata,
**not a modified production GSFP**. Exact-set input is required for GSFP even if
it has only one set. Do not inject the candidate into a live profile or borrow
weights from Learned for Spatial. The renderer retains sole ownership of live
GSFP persistence. A neutral zero head is refused because it is a weak execution
test. Partial/failed preparations remain in their new directory for diagnosis;
they have no complete manifest and cannot be run.
The run also refuses fixture sets whose reference outputs are indistinguishable
from zero or a repeated result at the declared tolerance. Use informative pairs
and justified tighter tolerances, rather than weakening this check.

## Campaign

`campaign` replaces the fg-matrix `npu-stage.sh` orchestration with one pipeline
inside this module. Produce GSFD captures with `GAMESCOPE_FRAMEGEN_RECORD` via
the fg-matrix harness `run.py record` phase or any motion-mode run. Freeze/copy
separate train and held-out captures before starting; do not record into them.

```bash
python3 tools/framegen-npu/experiment.py campaign --spec /absolute/campaign.json
```

The gamescope Python running this command needs NumPy and ONNX: it evaluates
through the existing evaluator functions and prepares bundles in process. Only
NPU execution uses the SDK Python from `config`; all CPU work finishes first,
then selected NPU sources run sequentially through the existing supervisor.
Setting every `npu` to `false` permits CPU-only work (including ONNX export).

The spec has exactly these fields; paths must be absolute. Keys are JSON uint32
integers (`240` = `0xf0`, `65776` = `0x100f0`). `cpu` is optional; profile lists
may be empty when a CPU source is present. Source names `neutral`/`cpu` are
reserved, and benchmark/source names use letters, digits, underscore or hyphen.

```json
{
  "config": "/absolute/private-npu.json",
  "cache": "/absolute/npu-cache",
  "out": "/absolute/new-campaign",
  "runs": 100,
  "atol": 0.001,
  "rtol": 0.05,
  "timeout": 7200,
  "benches": [{
    "name": "gravitymark",
    "train": "/absolute/captures/gravitymark-train",
    "heldout": "/absolute/captures/gravitymark-heldout",
    "profiles": [{
      "name": "gpu-spatial",
      "net": "/absolute/copied-spatial.prof",
      "key": 65776,
      "npu": true
    }],
    "cpu": {"steps": 20000, "crop": 32, "profile_key": 65776, "npu": true}
  }]
}
```

`timeout` bounds each CPU training subprocess and each SDK run, in seconds.
Every source gets a preparation bundle; GPU profiles use their explicit set key,
and the CPU candidate's `candidate.gsfr` is evaluated without a profile key.
`out/<bench>/` contains `eval-neutral`, `eval-<source>`, `bundle-<source>` and
selected `run-<source>` directories. Each successful step has a hash receipt.
Reusing the same spec skips only complete steps whose input and artifact hashes
match. Inputs, code, config contents and referenced SDK config files participate
in identity. Stale/partial/failed directories stop the campaign and remain for
diagnosis; use a new output root after diagnosis. No automatic deletion or
overwrite of step directories occurs; output and cache must be separate and
outside the input files. Runtime changes also require a new `runtime_tag`.

`campaign.json` and `campaign.md` are refreshed after steps and on failure. They
compare neutral and all sources side by side, including CPU training duration,
NPU load/timing/parity and source-minus-neutral deltas for every metric. Lower
residual/bad/edge/sigma and higher SSIM are preferable within this offline scope.
These are **field-resolution offline metrics, not served RGB**;
`temporal_sigma` measures residual variation, not displayed flicker.
`quality_validated` remains false. **NPU timing = host call incl. copies**;
numeric parity does not establish visible quality or a gameplay benefit.

## Explicit NPU execution

Only after the GPU campaign owner releases the machine for contention work:

```bash
python3 tools/framegen-npu/experiment.py run \
  --config /absolute/private-npu.json --bundle /tmp/spatial-npu-bundle-unique \
  --cache /absolute/private-cache --out /tmp/npu-run-unique \
  --timeout 1800 --runs 30 --atol 0.001 --rtol 0.05
```

The first compile can take 10–15 minutes. It is explicit offline work, never a
startup prerequisite. The default timeout is 60 s; increasing it opts into that
work. Results are `result.json` and `worker.log`; generated signatures stay in
the private run directory. For a new run, supply a new output directory. Cache
root may be reused. A per-key lock prevents concurrent compilation; an interrupted
or failed key stays quarantined. Use a **new cache root** after diagnosis, rather
than blindly deleting caches another process may use.

AMD disables CPU graph-assignment and Python runtime fallback, requires the
requested provider to remain active, and inspects ORT kernel events. CPU kernels
or missing execution evidence fail the experiment, including hybrid graphs that
would otherwise legitimately use CPU shape/output operations. Conservative
rejection is intentional. Internal provider implementation details still need
vendor profiling to establish actual hardware placement.

Intel compiles specifically for `NPU`, never AUTO/MULTI/HETERO, and checks
`EXECUTION_DEVICES`. Missing NPU support fails; there is no CPU substitute labelled
as NPU success. Adding another vendor means adding an adapter with `probe`,
`load`, `run`, `evidence`, and failure tests, not modifying render code.

Timings cover one direction's host call including copies, with one warmup per
direction and ORT profiling enabled on AMD. p99 from 30 calls is weak evidence;
use larger repetitions and separate unprofiled timings for performance claims.
Calls are sequential, not a measured bidirectional batch or GPU/NPU concurrent
run. BF16/INT8 drift can pass numeric tolerances and still damage visible quality.

The GPU fallback is structural at this stage: no experiment output is connected
to the renderer. There is no claimed live fallback implementation to validate.
