#!/usr/bin/env python3
"""Isolated AMD DLSS5 NR correctness probe; never launched by gamescope.

Initial upload/final diagnostic download are fixture I/O. Runtime interop must
share GPU images. This tool does not measure live latency or presentation.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(pci, uuid):
    if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', pci):
        raise ValueError('presenter PCI address must be canonical lowercase domain:bus:device.function')
    if int(pci[8:10], 16) > 31:
        raise ValueError('PCI device number exceeds 31')
    if not re.fullmatch(r'[0-9a-f]{32}', uuid) or int(uuid, 16) == 0:
        raise ValueError('presenter Vulkan UUID must be 32 nonzero lowercase hexadecimal digits')
    return pci, uuid


def png_size(path):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError('input must be a PNG fixture')
    w, h = struct.unpack('>II', header[16:24])
    if not 16 <= w <= 4096 or not 16 <= h <= 4096:
        raise ValueError('fixture dimensions must be between 16 and 4096')
    return w, h


def validate_evidence(stdout, stderr, raw, size, pci, uuid, disabled):
    if f'NR_POC_PRESENTER uuid={uuid} pci={pci} ' not in stdout:
        raise ValueError('D3D12/Vulkan presenter identity was not established')
    if 'NR_POC_ENCODING linear=1 encoding=1 hdr=0' not in stdout:
        raise ValueError('linear SDR input and network proxy encoding were not established')
    record = re.search(r'NR_POC_RAW width=(\d+) height=(\d+) invalid=(\d+) changed=(\d+) max_abs=(\S+) disabled=(\d+) changed_any=(\d+)', stdout)
    if not record or tuple(map(int, record.group(1, 2))) != size or int(record[3]) != 0 or int(record[6]) != int(disabled):
        raise ValueError('missing or invalid raw FP16 diagnostic report')
    if len(raw) != size[0] * size[1] * 8:
        raise ValueError('FP16 diagnostic extent or finiteness failed')
    values = [x[0] for x in struct.iter_unpack('<e', raw)]
    max_abs = float(record[5])
    if (any(not math.isfinite(x) for x in values) or not math.isfinite(max_abs) or max_abs < 0
            or not math.isclose(max_abs, max(map(abs, values)), rel_tol=1e-7)):
        raise ValueError('FP16 diagnostic finiteness/range disagrees with the report')
    changed, changed_any = int(record[4]), int(record[7])
    if not 0 <= changed <= 3 * size[0] * size[1] or not changed <= changed_any <= 4 * size[0] * size[1]:
        raise ValueError('invalid changed-channel coverage')
    if disabled:
        if changed_any:
            raise ValueError('disabled transport did not preserve the uploaded image')
        return {'disabled_transport_bit_exact_harness_asserted': True, 'raw_sha256': hashlib.sha256(raw).hexdigest()}
    if not re.search(r'NR_POC_COMPUTE pci=' + re.escape(pci) + r' matches=1(?:\s|$)', stdout):
        raise ValueError('compute runtime did not establish the exact presenter PCI identity')
    imports = re.findall(r'hipImportExternalMemory returned (-?\d+)', stderr)
    if len(imports) < 2 or any(int(value) != 0 for value in imports):
        raise ValueError('successful shared GPU image imports were not established')
    results = re.search(r'frames (\d+), results composed (\d+), network (\d+)x(\d+), last evaluation ([\d.]+) ms', stdout)
    if not results or int(results[2]) < 3 or min(int(results[3]), int(results[4])) <= 0 or changed <= 0:
        raise ValueError('no fresh changed NR output; refusing to count passthrough as inference')
    return {'results_composed': int(results[2]), 'network_size': list(map(int, results.group(3, 4))),
            'evaluation_ms': float(results[5]), 'successful_external_memory_imports': len(imports),
            'changed_rgb_channels': changed, 'raw_sha256': hashlib.sha256(raw).hexdigest()}


def compare_baseline(raw, baseline, size, changed):
    if len(raw) != len(baseline) or len(raw) != size[0] * size[1] * 8:
        raise ValueError('baseline FP16 extent mismatch')
    numeric_changes, max_delta = 0, 0.0
    for i, (a, b) in enumerate(zip(struct.iter_unpack('<e', raw), struct.iter_unpack('<e', baseline))):
        if not math.isfinite(a[0]) or not math.isfinite(b[0]):
            raise ValueError('nonfinite raw baseline comparison')
        if i % 4 != 3:
            numeric_changes += a[0] != b[0]
            max_delta = max(max_delta, abs(a[0] - b[0]))
    if numeric_changes <= 0 or numeric_changes != changed:
        raise ValueError('reported RGB changes disagree with the matched disabled baseline')
    return {'independent_changed_rgb_channels': numeric_changes, 'max_rgb_delta': max_delta}


def kill_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def command(command, env, out, timeout):
    with (out / 'stdout.log').open('w') as stdout, (out / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(list(map(str, command)), cwd=out, env=env,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        finally:
            # Include descendants even when their leader has already exited.
            kill_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            kill_group(process, signal.SIGKILL)
            process.wait()


def run(args):
    pci, uuid = identity(args.presenter_pci, args.presenter_uuid)
    size = png_size(args.image)
    proton, runtime, build, linux = (p.resolve(strict=True) for p in
                                    (args.proton, args.runtime, args.build, args.linux))
    hip = args.hip_library.resolve(strict=True)
    files = proton / 'files'
    required = [files / 'bin/wine', files / 'bin/wineserver', build / 'dlss5nr_harness.exe',
                build / 'dlss5nr_runtime.dll', build / 'zluda/nvcuda.dll', build / 'zluda/nvngx.dll',
                runtime / 'nvngx_dlssnr.dll', runtime / 'zluda/zluda_real.dll', runtime / 'zluda/nvapi64.dll',
                runtime / 'cache/ComputeCache/zluda2.db', runtime / 'cache/ComputeCache/coverage.txt',
                linux / 'libdlss5nr_hip_bridge.so', linux / 'amdhip64_7.dll', hip]
    if any(not p.is_file() for p in required):
        raise ValueError('missing explicit NR runtime prerequisite')
    out = args.out.resolve()
    if any(out == source or source in out.parents or out in source.parents
           for source in (proton, runtime, build, linux)):
        raise ValueError('probe output and supplied prerequisite trees must not overlap')
    manifest = build / 'nr-poc-source.json'
    if not manifest.is_file():
        raise ValueError('prepared source manifest must accompany the built helper')
    baseline = None
    if not args.disabled:
        if args.baseline is None:
            raise ValueError('enabled probe requires a passed, matched --baseline directory')
        saved = json.loads((args.baseline / 'result.json').read_text())
        if (saved.get('status') != 'passed' or saved.get('disabled') is not True
                or saved.get('input_sha256') != digest(args.image) or saved.get('pci') != pci
                or saved.get('vulkan_uuid') != uuid or saved.get('source_manifest_sha256') != digest(manifest)):
            raise ValueError('disabled baseline identity, source or input does not match')
        baseline = (args.baseline / 'output.png.fp16').read_bytes()
        if hashlib.sha256(baseline).hexdigest() != saved.get('raw_sha256'):
            raise ValueError('disabled baseline raw output has changed')
    # One campaign lock covers staging and execution; never overlap timing/builds.
    with (ROOT / '.claude/reviews/tools/spatial-learning/gpu-owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.image, out / 'input.png')
        local = out / 'runtime'
        shutil.copytree(runtime, local, ignore=shutil.ignore_patterns(
            'cache', 'nvngx_dlss.dll', 'nvngx_dlssd.dll', 'nvngx_dlssg.dll'))
        shutil.copy2(build / 'dlss5nr_harness.exe', out / 'nr.exe')
        shutil.copy2(build / 'dlss5nr_runtime.dll', out / 'dlss5nr_runtime.dll')
        for name in ('nvcuda.dll', 'nvngx.dll'):
            shutil.copy2(build / 'zluda' / name, local / 'zluda' / name)
        shutil.copy2(linux / 'amdhip64_7.dll', out / 'amdhip64_7.dll')
        for name in ('d3d12.dll', 'd3d12core.dll'):
            shutil.copy2(files / 'lib/wine/vkd3d-proton/x86_64-windows' / name, out / name)
        shutil.copy2(files / 'lib/wine/dxvk/x86_64-windows/dxgi.dll', out / 'dxgi.dll')
        shutil.copy2(manifest, out / 'source.json')
        data = out / 'data'
        data.mkdir()
        shutil.copytree(runtime / 'cache/ComputeCache', data / 'ComputeCache')
        env = os.environ.copy()
        for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'LD_PRELOAD', 'WINEPREFIX', 'VKD3D_VULKAN_DEVICE',
                    'VKD3D_FILTER_DEVICE_NAME', 'DXVK_FILTER_DEVICE_NAME', 'DLSS5NR_ALLOW_TRANSLATION'):
            env.pop(key, None)
        env.update(WINEPREFIX=str(out / 'prefix'), WINEDEBUG='-all,err+module', WINEESYNC='0', WINEFSYNC='0',
                   WINESERVER=str(files / 'bin/wineserver'), WINEDLLOVERRIDES='dxgi,d3d12,d3d12core,amdhip64_7,nvcuda,nvapi64=n;winemenubuilder.exe=d',
                   WINEDLLPATH=os.pathsep.join(map(str, (files / 'lib/vkd3d', files / 'lib/wine'))),
                   LD_PRELOAD='libdlss5nr_hip_bridge.so',
                   LD_LIBRARY_PATH=os.pathsep.join(map(str, (linux, files / 'lib/x86_64-linux-gnu',
                       files / 'lib/i386-linux-gnu', files / 'lib/wine/x86_64-unix'))) +
                       (os.pathsep + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),
                   DLSSNR_HIP_LIB=str(hip), DLSSNR_BRIDGE_LOG='1', DLSS5NR_COPY='0', DLSS5NR_ARCH_NEURAL='0x1b0',
                   GSC_NR_PCI=pci, GSC_NR_VULKAN_UUID=uuid, PROTON_DISABLE_NVAPI='1',
                   DXVK_LOG_PATH=str(out), DXVK_LOG_LEVEL='info', VKD3D_DEBUG='warn')
        argv = [files / 'bin/wine', out / 'nr.exe', '--image', 'Z:' + str(out / 'input.png'),
                '--out', 'Z:' + str(out / 'output.png'), '--runtime-dir', 'Z:' + str(local),
                '--data-dir', 'Z:' + str(data), '--linear', '--bare', '--scale', '1',
                '--frames', '4', '--evaluations', '0' if args.disabled else '3']
        wait_for_network = not getattr(args, 'async_mode', False)
        if wait_for_network:
            argv.append('--wait')
        if args.disabled:
            argv.append('--disabled')
        receipt = {'schema': 1, 'algorithm': 'dlss5-neural-rendering', 'provider': 'amd-zluda',
                   'scope': 'offscreen correctness only', 'pci': pci, 'vulkan_uuid': uuid, 'image_size': size,
                   'disabled': args.disabled, 'host_copy_fallback': False, 'command': list(map(str, argv)),
                   'model_compatibility_architecture': '0x1b0 through the upstream NVAPI adapter; physical GPU identified separately',
                   'supervisor_pid': os.getpid(),
                   'wait_for_network': wait_for_network,
                   'prerequisite_sha256': {str(p): digest(p) for p in required}, 'input_sha256': digest(out / 'input.png'),
                   'source_manifest_sha256': digest(manifest), 'source': json.loads(manifest.read_text()),
                   'color_domain': 'sRGB PNG decoded to FP16 linear RGB; explicit LINEAR NR proxy; SDR output',
                   'staged_sha256': {str(p.relative_to(out)): digest(p) for p in
                       [*local.rglob('*.dll'), *local.rglob('*.hsaco'), *local.rglob('kernels*.txt'),
                        out / 'nr.exe', out / 'dlss5nr_runtime.dll', out / 'amdhip64_7.dll',
                        out / 'dxgi.dll', out / 'd3d12.dll', out / 'd3d12core.dll', out / 'source.json']},
                   'status': 'starting', 'limits': ['No live compositor integration or latency/jitter/throughput claim.',
                       'Static image only; no motion tracking, HDR, native NVIDIA or perceptual quality qualification.',
                       'Fixture upload and explicit final FP16 readback are diagnostics, not interop transport.']}
        result = out / 'result.json'
        result.write_text(json.dumps(receipt, indent=2) + '\n')
        started = time.monotonic()
        try:
            code = command(argv, env, out, args.timeout)
            receipt['exit_code'] = code
            if code:
                raise RuntimeError(f'NR helper exited {code}; see stderr.log')
            raw = (out / 'output.png.fp16').read_bytes()
            receipt.update(validate_evidence((out / 'stdout.log').read_text(errors='replace'),
                (out / 'stderr.log').read_text(errors='replace'), raw, size, pci, uuid, args.disabled))
            if baseline is not None:
                receipt.update(compare_baseline(raw, baseline, size, receipt['changed_rgb_channels']))
            receipt['status'] = 'passed'
        except BaseException as error:
            receipt.update(status='failed', reason=str(error) or type(error).__name__)
            raise
        finally:
            # Wine's server can daemonize out of the helper group. Kill only
            # the server selected by this newly created, private prefix.
            clean_env = env.copy()
            clean_env.pop('LD_PRELOAD', None)
            with (out / 'cleanup.log').open('w') as log:
                try:
                    cleaned = subprocess.run([files / 'bin/wineserver', '-k', '-w'], env=clean_env,
                        stdout=log, stderr=subprocess.STDOUT, timeout=10)
                    receipt['prefix_cleanup_exit'] = cleaned.returncode
                except (OSError, subprocess.TimeoutExpired) as error:
                    receipt['prefix_cleanup_error'] = str(error)
            if receipt.get('prefix_cleanup_exit') != 0:
                receipt.update(status='failed', cleanup_failed=True)
            receipt['wall_seconds'] = time.monotonic() - started
            result.write_text(json.dumps(receipt, indent=2) + '\n')
        if receipt['status'] != 'passed':
            raise RuntimeError('private Wine prefix cleanup did not complete; see result.json')
        return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('proton', 'runtime', 'build', 'linux', 'hip-library', 'image', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--presenter-pci', required=True)
    parser.add_argument('--presenter-uuid', required=True)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--disabled', action='store_true')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--async', dest='async_mode', action='store_true',
                        help='exercise asynchronous NR; the offscreen host still waits for its D3D12 submissions')
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error('timeout must be between 1 and 600 seconds')
    # Reuse the existing offline helper signal policy (stdlib only). It ignores
    # repeated termination signals while receipts and owned children are retired.
    spec = importlib.util.spec_from_file_location('_nr_common', ROOT / 'tools/framegen-npu/common.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    with common.termination_signals():
        receipt = run(args)
        print(json.dumps({'result': str(args.out.resolve() / 'result.json'), 'status': receipt['status'],
            'pci': receipt['pci'], 'disabled': receipt['disabled'], 'wait_for_network': receipt['wait_for_network'],
            'results_composed': receipt.get('results_composed'), 'evaluation_ms': receipt.get('evaluation_ms'),
            'imports': receipt.get('successful_external_memory_imports'), 'cleanup_exit': receipt.get('prefix_cleanup_exit')}, indent=2))
