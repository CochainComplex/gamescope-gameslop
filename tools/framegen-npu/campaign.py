"""Resumable offline orchestration. Evaluation/preparation use this Python, never the SDK."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from common import HERE, ROOT, digest, module, write_json
import preparation


METRICS = ('resid', 'resid_conf', 'bad%', 'ssim', 'edge', 'temporal_sigma')
LABELS = {
    'quality_validated': False,
    'metric_scope': 'field-resolution offline metric, not served RGB',
    'temporal_scope': 'temporal_sigma is residual variation, not displayed flicker',
    'timing_scope': 'NPU timing = host call incl. copies',
}


def fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f'expected exactly {sorted(required)}; optional {sorted(optional)}')


def absolute(value):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError('campaign paths must be absolute')
    return Path(value).resolve()


def integer(value, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError('invalid campaign integer')


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('names must be safe path components (letters, digits, underscore, hyphen)')


def validate_spec(value):
    fields(value, ('config', 'cache', 'out', 'runs', 'atol', 'rtol', 'timeout', 'benches'))
    for key in ('config', 'cache', 'out'):
        absolute(value[key])
    integer(value['runs'])
    integer(value['timeout'])
    for key in ('atol', 'rtol'):
        v = value[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError('tolerances must be finite and nonnegative')
    if not isinstance(value['benches'], list) or not value['benches']:
        raise ValueError('benches must be a nonempty list')
    names = set()
    for bench in value['benches']:
        fields(bench, ('name', 'train', 'heldout', 'profiles'), ('cpu',))
        name(bench['name'])
        if bench['name'] in names:
            raise ValueError('duplicate bench name')
        names.add(bench['name'])
        if absolute(bench['train']) == absolute(bench['heldout']):
            raise ValueError('train and heldout must be separate frozen captures')
        if not isinstance(bench['profiles'], list):
            raise ValueError('profiles must be a list')
        sources = {'neutral', 'cpu'}
        for profile in bench['profiles']:
            fields(profile, ('name', 'net', 'key', 'npu'))
            name(profile['name'])
            if profile['name'] in sources:
                raise ValueError('duplicate/reserved source name')
            sources.add(profile['name'])
            absolute(profile['net'])
            integer(profile['key'], 0, 0xffffffff)
            if type(profile['npu']) is not bool:
                raise ValueError('npu must be boolean')
        if 'cpu' in bench:
            cpu = bench['cpu']
            fields(cpu, ('steps', 'crop', 'profile_key', 'npu'))
            integer(cpu['steps'], 2)  # one cosine-schedule step has zero LR
            integer(cpu['crop'])
            integer(cpu['profile_key'], 0, 0xffffffff)
            if type(cpu['npu']) is not bool:
                raise ValueError('npu must be boolean')
        if not bench['profiles'] and 'cpu' not in bench:
            raise ValueError('bench needs at least one source')
    return value


def capture(path, minimum):
    paths = sorted(absolute(path).glob('fg_*.bin'))
    if len(paths) < minimum:
        raise ValueError(f'at least {minimum} frozen GSFD samples required: {path}')
    return {str(p.resolve()): digest(p) for p in paths}


def identity(spec):
    """Hash every input and implementation, including indirectly referenced SDK config."""
    from experiment import config
    cfg = config(spec['config'])
    absolute(cfg['python'])
    config_files = cfg.get('config_files', [])
    if not isinstance(config_files, list):
        raise ValueError('config_files must be a list')
    files = {str(absolute(p)): digest(absolute(p)) for p in config_files}
    provider_file = cfg.get('provider_options', {}).get('config_file')
    if provider_file is not None and str(absolute(provider_file)) not in files:
        raise ValueError('provider config_file must be included in config_files')
    files[str(absolute(spec['config']))] = digest(spec['config'])
    for bench in spec['benches']:
        train, heldout = capture(bench['train'], 2), capture(bench['heldout'], 1)
        if set(train) & set(heldout):
            raise ValueError('train and heldout share capture paths')
        files.update(train)
        files.update(heldout)
        for profile in bench['profiles']:
            files[str(absolute(profile['net']))] = digest(profile['net'])
    out, cache = absolute(spec['out']), absolute(spec['cache'])
    if out == cache or out in cache.parents or cache in out.parents:
        raise ValueError('out and cache must be separate directories')
    for p in files:
        path = Path(p)
        if any(path == d or d in path.parents for d in (out, cache)):
            raise ValueError('inputs must be outside output/cache directories')
    for path in (*HERE.glob('*.py'), ROOT / 'scripts/framegen-net-eval.py',
                 ROOT / 'scripts/framegen-net-train.py'):
        files[str(path)] = digest(path)
    import numpy as np
    return {'spec': spec, 'files': files, 'python': sys.version,
            'executable': sys.executable, 'numpy': np.__version__}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def artifacts(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob('*'))
            if p.is_file() and p != path / 'campaign-step.json'}


def step(path, request, operation):
    """Only a receipt written after success permits reuse; preserve all failed work."""
    receipt = path / 'campaign-step.json'
    if path.exists():
        try:
            saved = json.loads(receipt.read_text())
            if (saved['schema'] == 1 and saved['request'] == request
                    and saved['data_sha256'] == fingerprint(saved['data'])
                    and saved['files'] and saved['files'] == artifacts(path)):
                return saved['data']
        except (OSError, ValueError, KeyError, TypeError):
            pass
        raise RuntimeError(f'stale/incomplete campaign step: {path}; preserve it and use a new out')
    data = operation(path)
    files = artifacts(path)
    if not files:
        raise RuntimeError(f'campaign step produced no artifacts: {path}')
    write_json(receipt, {'schema': 1, 'request': request, 'files': files, 'data': data,
                         'data_sha256': fingerprint(data)})
    return data


def evaluate(path, heldout, net=None, key=None):
    evaluator = module('framegen-net-eval')
    path.mkdir(parents=True, exist_ok=False)
    try:
        params = evaluator.load_blob(net, key) if net else None
        result = evaluator.evaluate(evaluator.T.load_dataset(heldout), params)
    except SystemExit as exc:
        raise ValueError(f'evaluator rejected source: {exc}') from exc
    metrics = result['refined' if net else 'neutral']
    if set(metrics) != set(METRICS) or not all(math.isfinite(v) for v in metrics.values()):
        raise ValueError('incomplete/non-finite evaluator metrics')
    write_json(path / 'evaluation.json', {**LABELS, 'metrics': metrics})
    return metrics


def prepare(path, bench, source):
    cpu = source['name'] == 'cpu'
    preparation.prepare(argparse.Namespace(
        out=str(path), data=bench['train'], net=None if cpu else source['net'],
        net_key=None if cpu else source['key'],
        profile_key=source['profile_key'] if cpu else source['key'],
        train_steps=source['steps'] if cpu else 0,
        crop=source['crop'] if cpu else 32, training_timeout=source['timeout']))
    manifest = json.loads((path / 'manifest.json').read_text())
    expected = {'model.onnx', 'fixture.npz', 'candidate.gsfr'}
    if (manifest.get('schema') != 1 or set(manifest['files']) != expected
            or any(digest(path / p) != h for p, h in manifest['files'].items())):
        raise ValueError(f'incomplete preparation: {path}')
    return {'cpu_train_steps': manifest['cpu_train_steps'],
            'cpu_train_seconds': manifest['cpu_train_seconds']}


def run_npu(path, bundle, spec):
    from experiment import supervise
    supervise(argparse.Namespace(action='run', config=spec['config'],
                                 out=str(path), bundle=str(bundle), cache=spec['cache'],
                                 runs=spec['runs'], atol=spec['atol'], rtol=spec['rtol'],
                                 timeout=spec['timeout']))
    result = json.loads((path / 'result.json').read_text())
    if result.get('status') != 'npu-raw-head-parity-passed':
        raise RuntimeError(f'incomplete NPU result: {path}')
    m = result['measurement']
    return {'status': result['status'], 'load_ms': result['load_ms'],
            'p50_ms': m['call_ms_p50'], 'p99_ms': m['call_ms_p99'],
            'parity': {'passed': True, 'max_abs_error': m['max_abs_error'],
                       'atol': m['atol'], 'rtol': m['rtol']}}


def markdown(report):
    lines = ['# Offline NPU campaign', '', f"Status: {report['status']}", '',
             'quality_validated: false; ' + LABELS['metric_scope'] + '.',
             LABELS['temporal_scope'] + '. ' + LABELS['timing_scope'] + '.', '']
    def fmt(v):
        return '—' if v is None else f'{v:.6g}'
    for bench in report['benches']:
        lines += [f"## {bench['name']}", '',
                  '| source | resid | conf-resid | bad (%) | ssim | edge | temporal_sigma | delta vs neutral (source − neutral) | CPU train steps/seconds | NPU status | load_ms | p50/p99 ms | parity |',
                  '|---|' + '---:|' * 8 + '---|' + '---:|' * 2 + '---|']
        for row in bench['rows']:
            npu = row['npu']
            delta = ', '.join(f'{k} {v:+.6g}' for k, v in row['delta_vs_neutral'].items())
            parity = npu.get('parity')
            cells = [row['name'], *(fmt(row['metrics'][k]) for k in METRICS), delta,
                     f"{row['cpu_train_steps']} / {fmt(row['cpu_train_seconds'])}",
                     npu['status'], fmt(npu.get('load_ms')),
                     f"{fmt(npu.get('p50_ms'))} / {fmt(npu.get('p99_ms'))}",
                     ('passed; max_abs_error=' + fmt(parity['max_abs_error'])) if parity else '—']
            lines.append('| ' + ' | '.join(cells) + ' |')
        lines.append('')
    if report.get('error'):
        lines += ['Stopped: ' + report['error'], '']
    return '\n'.join(lines)


def campaign(args):
    spec = validate_spec(json.loads(absolute(args.spec).read_text()))
    inputs = identity(spec)
    token = fingerprint(inputs)
    out = absolute(spec['out'])
    report = {'schema': 1, **LABELS, 'identity': inputs, 'fingerprint': token,
              'status': 'running', 'benches': []}
    if out.exists():
        try:
            previous = json.loads((out / 'campaign.json').read_text())
            valid = previous['schema'] == 1 and previous['identity'] == inputs
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            raise RuntimeError(f'stale/incomplete campaign root: {out}; use a new out')
    else:
        out.mkdir(parents=True, exist_ok=False)

    def save():
        write_json(out / 'campaign.json', report)
        (out / 'campaign.md').write_text(markdown(report))

    def perform(path, operation):
        return step(path, {'campaign': token, 'step': str(path.relative_to(out))}, operation)

    pending = []
    save()
    try:
        for bench in spec['benches']:
            base = out / bench['name']
            summary = {'name': bench['name'], 'rows': []}
            report['benches'].append(summary)
            neutral = perform(base / 'eval-neutral',
                              lambda p: evaluate(p, bench['heldout']))
            summary['rows'].append({'name': 'neutral', 'metrics': neutral,
                                    'delta_vs_neutral': {k: 0.0 for k in METRICS},
                                    'cpu_train_steps': 0, 'cpu_train_seconds': None,
                                    'npu': {'status': 'not-requested'}})
            save()
            sources = list(bench['profiles'])
            if 'cpu' in bench:
                sources.append({'name': 'cpu', **bench['cpu']})
            for source in sources:
                source = {**source, 'timeout': spec['timeout']}
                bundle = base / ('bundle-' + source['name'])
                if source['name'] != 'cpu':
                    metrics = perform(base / ('eval-' + source['name']),
                                      lambda p: evaluate(p, bench['heldout'],
                                                         source['net'], source['key']))
                training = perform(bundle, lambda p: prepare(p, bench, source))
                if source['name'] == 'cpu':
                    metrics = perform(base / 'eval-cpu', lambda p: evaluate(
                        p, bench['heldout'], bundle / 'candidate.gsfr'))
                row = {'name': source['name'], 'metrics': metrics, **training,
                       'delta_vs_neutral': {k: metrics[k] - neutral[k] for k in METRICS},
                       'npu': {'status': 'pending' if source['npu'] else 'not-requested'}}
                summary['rows'].append(row)
                if source['npu']:
                    pending.append((base / ('run-' + source['name']), bundle, row))
                save()
        # All CPU work precedes sequential SDK execution, matching the absorbed pipeline.
        for path, bundle, row in pending:
            row['npu'] = {'status': 'running'}
            save()
            try:
                row['npu'] = perform(path, lambda p: run_npu(p, bundle, spec))
            except Exception:
                row['npu'] = {'status': 'failed-or-incomplete'}
                raise
            save()
        if identity(spec) != inputs:
            raise RuntimeError('campaign inputs changed; use frozen copies and a new out')
        report['status'] = 'complete'
        save()
    except Exception as exc:
        report.update(status='stopped', error=str(exc))
        save()
        raise
    print(json.dumps({'campaign': str(out / 'campaign.json'), 'quality_validated': False}))
