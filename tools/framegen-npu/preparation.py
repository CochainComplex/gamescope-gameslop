"""CPU learning/export stage. Never imports a vendor inference runtime."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import ROOT, digest, module, write_json


def validate_request(args):
    if args.net_key is not None and (not args.net or args.net_key != args.profile_key):
        raise ValueError('--net-key must match the intended --profile-key')
    if not 0 <= args.profile_key <= 0xffffffff:
        raise ValueError('profile key must be uint32')
    if bool(args.net) == bool(args.train_steps):
        raise ValueError('choose exactly one of --net or --train-steps')
    if args.net and Path(args.net).read_bytes()[:4] == b'GSFP' and args.net_key is None:
        raise ValueError('GSFP requires explicit --net-key, even for a single set')
    files = sorted(Path(args.data).resolve().glob('fg_*.bin'))
    if len(files) < 2:
        raise ValueError('at least two recorded GSFD samples required')
    return files


def candidate_params(args, out):
    train, evaluate = module('framegen-net-train'), module('framegen-net-eval')
    if args.net:
        params = evaluate.load_blob(Path(args.net).resolve(), args.net_key)
        train.export(params, out / 'candidate.gsfr')
        return params, None
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    cmd = [sys.executable, str(ROOT / 'scripts/framegen-net-train.py'),
           '--data', str(Path(args.data).resolve()), '--out', str(out / 'candidate.gsfr'),
           '--steps', str(args.train_steps), '--batch', '1', '--crop', str(args.crop)]
    with (out / 'cpu-training.log').open('w') as log:
        start = time.perf_counter()
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=args.training_timeout)
        seconds = time.perf_counter() - start
    return evaluate.load_blob(out / 'candidate.gsfr'), seconds


def fixtures(params, paths):
    import numpy as np
    train, evaluate = module('framegen-net-train'), module('framegen-net-eval')
    features, references = [], []
    for path in paths:
        lp, lc, ff, fr = train.load_sample(path)
        if not all(np.isfinite(a).all() for a in (lp, lc, ff, fr)):
            raise ValueError('recorded sample contains non-finite values')
        for x, y, src, dst in ((ff, fr, lp, lc), (fr, ff, lc, lp)):
            f = evaluate.full_features(x, y, src, dst)[None]
            f = np.pad(f, ((0, 0), (0, 0), (3, 3), (3, 3)), mode='edge')
            o, _ = train.net_fwd(params, f)
            features.append(f)
            references.append(o)
    features, references = np.stack(features), np.stack(references)
    if not np.isfinite(references).all():
        raise ValueError('CPU reference contains non-finite values')
    return features, references


def export_model(params, input_shape, output_shape, path):
    import onnx
    from onnx import helper as H, numpy_helper as N, TensorProto as P
    nodes, init, previous = [], [], 'features'
    for i in range(1, 4):
        init += [N.from_array(params[f'W{i}'], f'W{i}'),
                 N.from_array(params[f'b{i}'], f'b{i}')]
        output = 'head' if i == 3 else f'conv{i}'
        nodes.append(H.make_node('Conv', [previous, f'W{i}', f'b{i}'], [output],
                                 kernel_shape=[3, 3]))
        previous = output
        if i != 3:
            previous = f'relu{i}'
            nodes.append(H.make_node('Relu', [output], [previous]))
    graph = H.make_graph(nodes, 'framegen-raw-head-experiment',
                         [H.make_tensor_value_info('features', P.FLOAT, input_shape)],
                         [H.make_tensor_value_info('head', P.FLOAT, output_shape)], init)
    model = H.make_model(graph, opset_imports=[H.make_opsetid('', 17)])
    model.ir_version = 8  # compatible with the existing local SDK
    onnx.checker.check_model(model)
    onnx.save(model, path)  # embedded weights, no external data


def prepare(args):
    import numpy as np
    __import__('onnx')  # check the optional exporter before starting training
    files = validate_request(args)
    # Input data must be copies, not a directory being recorded into.
    source = {str(p): digest(p) for p in files}
    source_net = digest(args.net) if args.net else None
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    params, train_seconds = candidate_params(args, out)
    if not np.any(params['W3']) and not np.any(params['b3']):
        raise ValueError('zero-head model is not a useful execution canary; supply trained weights')
    selected = (files[0], files[-1])
    features, reference = fixtures(params, selected)
    export_model(params, features.shape[1:], reference.shape[1:], out / 'model.onnx')
    np.savez(out / 'fixture.npz', features=features, reference=reference)
    after = {str(p): digest(p) for p in sorted(Path(args.data).resolve().glob('fg_*.bin'))}
    if after != source or (args.net and digest(args.net) != source_net):
        raise ValueError('source captures/weights changed during preparation; use frozen copies')
    manifest = {'schema': 1, 'experimental': True, 'profile_key': args.profile_key,
                'source_net_key': args.net_key, 'cpu_train_steps': args.train_steps,
                'cpu_train_seconds': train_seconds,
                'source_net_sha256': source_net,
                'training_files': source if args.train_steps else {},
                'fixture_files': {str(p): source[str(p)] for p in selected},
                'files': {p: digest(out / p) for p in
                          ('model.onnx', 'fixture.npz', 'candidate.gsfr')},
                'trainer_sha256': digest(ROOT / 'scripts/framegen-net-train.py'),
                'evaluator_sha256': digest(ROOT / 'scripts/framegen-net-eval.py'),
                'semantics': 'offline-12ch-mode0-edgehalo3-validconv-rawhead-v1',
                'quality_validated': False}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'prepared': str(out), 'quality_validated': False}))
