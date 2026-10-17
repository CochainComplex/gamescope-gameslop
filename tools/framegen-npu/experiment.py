#!/usr/bin/env python3
"""Offline, opt-in NPU experiment. Never imported or launched by gamescope."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common import HERE, digest, write_json
from preparation import prepare


def config(path):
    value = json.loads(Path(path).read_text())
    if value.get('backend') not in ('amd-vitis', 'intel-openvino'):
        raise ValueError('unknown backend')
    if not value.get('runtime_tag'):
        raise ValueError('runtime_tag must identify the installed SDK/driver/firmware stack')
    if not Path(value['python']).is_absolute():
        raise ValueError('helper python must be an absolute path')
    if not all(isinstance(k, str) and isinstance(v, str)
               for k, v in value.get('environment', {}).items()):
        raise ValueError('environment must map strings to strings')
    return value


def load_fixture(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        features, reference = data['features'], data['reference']
    for array in (features, reference):
        if array.dtype != np.float32 or array.ndim != 5:
            raise ValueError('fixture must contain float32 batches of NCHW tensors')
        if not np.isfinite(array).all():
            raise ValueError('non-finite experiment fixture')
    if features.shape[:3] != (4, 1, 12) or reference.shape[:3] != (4, 1, 4):
        raise ValueError('fixture must contain two real pairs in both directions')
    if features.shape[-2:] != tuple(n + 6 for n in reference.shape[-2:]):
        raise ValueError('fixture has inconsistent input halo/output extent')
    return features, reference


def cache_identity(cfg, probe, devices, model_hash):
    opts = cfg.get('provider_options', {})
    hashes = {str(Path(p).resolve()): digest(p) for p in cfg.get('config_files', [])}
    if 'config_file' in opts and str(Path(opts['config_file']).resolve()) not in hashes:
        raise ValueError('provider config_file must be included in config_files for cache identity')
    return {'model': model_hash, 'config': cfg, 'config_hashes': hashes,
            'runtime': probe, 'hardware': devices, 'helper_sha256': digest(__file__),
            'adapters_sha256': digest(HERE / 'backends.py')}


def worker(args):
    import fcntl
    from backends import BACKENDS, inventory, measure

    cfg = config(args.config)
    backend = BACKENDS[cfg['backend']]()
    probe = backend.probe()
    info = {'schema': 1, 'backend': cfg['backend'], 'probe': probe,
            'hardware_hints': inventory(), 'available_is_not_execution': True}
    if args.action == 'probe':
        write_json(Path(args.result), info)
        return
    if not probe['available']:
        raise RuntimeError('requested NPU runtime/device is unavailable')
    bundle = Path(args.bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    expected = {'model.onnx', 'fixture.npz', 'candidate.gsfr'}
    if manifest.get('schema') != 1 or set(manifest['files']) != expected:
        raise ValueError('unsupported experiment manifest')
    if any(digest(bundle / p) != manifest['files'][p] for p in expected):
        raise ValueError('experiment artifact hash mismatch')
    features, reference = load_fixture(bundle / 'fixture.npz')
    identity = cache_identity(cfg, probe, info['hardware_hints'], manifest['files']['model.onnx'])
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(args.cache).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = root / key
    with (root / (key + '.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache.mkdir(exist_ok=True)
        state = cache / 'state.json'
        if state.exists() and json.loads(state.read_text())['status'] != 'ready':
            raise RuntimeError(f'incomplete/failed cache quarantined: {cache}; use a new cache root')
        write_json(state, {'status': 'building', 'identity': identity})
        start = time.perf_counter()
        backend.load(bundle / 'model.onnx', cache, cfg.get('provider_options', {}))
        load_ms = (time.perf_counter() - start) * 1000
        measured = measure(backend, features, reference, args.runs, args.atol, args.rtol)
        write_json(state, {'status': 'ready', 'identity': identity})
    info.update(status='npu-raw-head-parity-passed', load_ms=load_ms,
                measurement=measured, manifest=manifest, cache_key=key,
                quality_validated=False, live_integration=False)
    write_json(Path(args.result), info)


def supervise(args):
    cfg = config(args.config)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(cfg.get('environment', {}))  # child only, before loading runtime
    for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[k] = '1'
    result = out / 'result.json'
    command = [cfg['python'], str(Path(__file__).resolve()), '_worker',
               '--action', args.action, '--config', str(Path(args.config).resolve()),
               '--result', str(result), '--runs', str(args.runs),
               '--atol', str(args.atol), '--rtol', str(args.rtol)]
    if args.action == 'run':
        if not args.bundle or not args.cache:
            raise ValueError('run requires --bundle and --cache')
        command += ['--bundle', str(Path(args.bundle).resolve()),
                    '--cache', str(Path(args.cache).resolve())]
    with (out / 'worker.log').open('w') as log:
        proc = subprocess.Popen(command, env=env, cwd=out, start_new_session=True,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            # Compiler descendants may survive their leader; terminate the group.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            write_json(result, {'status': 'aborted', 'quality_validated': False})
            raise RuntimeError(f'helper aborted; see {out / "worker.log"}') from None
    if code or not result.exists():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        write_json(result, {'status': 'failed', 'worker_exit': code,
                            'quality_validated': False})
        raise RuntimeError(f'helper failed; see {out / "worker.log"}')
    print(result.read_text())


def positive(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def tolerance(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    from campaign import campaign
    p = sub.add_parser('campaign', help='resumable held-out evaluation, preparation and NPU runs')
    p.add_argument('--spec', required=True, help='absolute campaign JSON path')
    p.set_defaults(func=campaign)
    p = sub.add_parser('prepare', help='CPU train or load weights; export offline NPU bundle')
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True, help='new artifact directory, never a live profile')
    p.add_argument('--profile-key', type=lambda s: int(s, 0), required=True,
                   help='intended GSFP role+encoding key (metadata only)')
    p.add_argument('--net')
    p.add_argument('--net-key', type=lambda s: int(s, 0))
    p.add_argument('--train-steps', type=positive, default=0)
    p.add_argument('--crop', type=positive, default=32)
    p.add_argument('--training-timeout', type=positive, default=600)
    p.set_defaults(func=prepare)
    for name in ('probe', 'run', '_worker'):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--runs', type=positive, default=30)
        p.add_argument('--atol', type=tolerance, default=0.001)
        p.add_argument('--rtol', type=tolerance, default=0.05)
        p.add_argument('--bundle')
        p.add_argument('--cache')
        if name == '_worker':
            p.add_argument('--action', choices=('probe', 'run'), required=True)
            p.add_argument('--result', required=True)
            p.set_defaults(func=worker)
        else:
            p.add_argument('--out', required=True)
            p.add_argument('--timeout', type=positive, default=60,
                           help='explicitly increase for first compile; never game startup')
            p.set_defaults(func=supervise)
    args = ap.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, RuntimeError, ImportError, subprocess.SubprocessError) as e:
        ap.exit(1, f'{e}\n')


if __name__ == '__main__':
    main()
