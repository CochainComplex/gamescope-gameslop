"""CPU-only contracts. No accelerator or optional SDK required."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backends
import experiment as E
import campaign as C


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def run(self, _):
        return next(self.outputs)

    def evidence(self):
        return {'test_only': True}


class Contracts(unittest.TestCase):
    def test_measure_rejects_wrong_finite_nan_and_shape(self):
        x = np.ones((2, 1, 1, 2, 2), np.float32)
        x[1] *= 2
        for y in (np.zeros_like(x[0]), x[0] * np.nan, x[0][..., :1]):
            with self.subTest(shape=y.shape), self.assertRaises(RuntimeError):
                backends.measure(FakeBackend([y]), x, x, 1, .001, .01)

    def test_measure_counts_both_directions_and_omits_warmup(self):
        x = np.ones((2, 1, 1, 2, 2), np.float32)
        x[1] *= 2
        result = backends.measure(FakeBackend([x[0]] * 3 + [x[1]] * 3), x, x, 2, 0, 0)
        self.assertEqual(result['calls'], 4)
        self.assertEqual(result['directions'], 2)
        self.assertEqual(result['max_abs_error'], 0)

    def test_weak_execution_fixtures_rejected(self):
        x = np.ones((2, 1, 1, 2, 2), np.float32)
        for ref in (x, x * .00001):
            with self.assertRaises(RuntimeError):
                backends.measure(FakeBackend([]), x, ref, 1, .001, .05)

    def test_intel_rejects_non_npu_execution_devices(self):
        class Compiled:
            def get_property(self, _):
                return ['CPU']
        backend = backends.IntelOpenVino.__new__(backends.IntelOpenVino)
        backend.compiled = Compiled()
        with self.assertRaises(RuntimeError):
            backend.evidence()

    def test_amd_requires_kernel_evidence_not_provider_listing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'trace.json'
            class Session:
                def end_profiling(self):
                    return str(path)
            backend = backends.AmdOrt.__new__(backends.AmdOrt)
            backend.session = Session()
            for providers in ([], ['CPUExecutionProvider'],
                              [backend.provider, 'CPUExecutionProvider']):
                path.write_text(json.dumps([{'cat': 'Node', 'args': {'provider': p}}
                                            for p in providers]))
                with self.assertRaises(RuntimeError):
                    backend.evidence()
            path.write_text(json.dumps([{'cat': 'Node', 'args': {'provider': backend.provider}}]))
            self.assertEqual(backend.evidence()['profiled_kernel_events'], 1)

    def test_invalid_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'config.json'
            for cfg in ({'backend': 'cpu'}, {'backend': 'amd-vitis'},
                        {'backend': 'amd-vitis', 'runtime_tag': 'test', 'python': 'python'}):
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    E.config(path)

    def test_numerical_tolerance_rejects_nan_infinity_negative(self):
        for value in ('nan', 'inf', '-1'):
            with self.assertRaises(argparse.ArgumentTypeError):
                E.tolerance(value)

    def test_helper_failure_is_local_and_records_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = root / 'config.json'
            cfg.write_text(json.dumps({'backend': 'amd-vitis', 'runtime_tag': 'test',
                                       'python': '/bin/false',
                                       'environment': {'XILINX_XRT': 'child-only'}}))
            before = dict(os.environ)
            args = argparse.Namespace(config=str(cfg), out=str(root / 'run'),
                                      action='probe', runs=1, atol=0, rtol=0, timeout=5)
            with self.assertRaises(RuntimeError):
                E.supervise(args)
            self.assertEqual(dict(os.environ), before)
            result = json.loads((root / 'run/result.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertFalse(result['quality_validated'])

    def test_helper_timeout_records_abort(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sleeper = root / 'sdk-python'
            sleeper.write_text('#!/bin/sh\nexec sleep 30\n')
            sleeper.chmod(0o700)
            cfg = root / 'config.json'
            cfg.write_text(json.dumps({'backend': 'amd-vitis', 'runtime_tag': 'test',
                                       'python': str(sleeper)}))
            args = argparse.Namespace(config=str(cfg), out=str(root / 'run'),
                                      action='probe', runs=1, atol=0, rtol=0, timeout=1)
            with self.assertRaises(RuntimeError):
                E.supervise(args)
            self.assertEqual(json.loads((root / 'run/result.json').read_text())['status'],
                             'aborted')

    def test_failed_compile_cache_is_not_reused(self):
        class BrokenBackend:
            calls = 0

            def probe(self):
                return {'available': True, 'runtime': 'test-only'}

            def load(self, *_):
                type(self).calls += 1
                raise RuntimeError('synthetic compiler failure')

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            bundle = root / 'bundle'
            bundle.mkdir()
            paths = ('model.onnx', 'fixture.npz', 'candidate.gsfr')
            for name in paths:
                (bundle / name).write_bytes(b'test-only')
            np.savez(bundle / 'fixture.npz',
                     features=np.ones((4, 1, 12, 8, 8), np.float32),
                     reference=np.ones((4, 1, 4, 2, 2), np.float32))
            E.write_json(bundle / 'manifest.json',
                         {'schema': 1, 'files': {p: E.digest(bundle / p) for p in paths}})
            cfg = root / 'config.json'
            E.write_json(cfg, {'backend': 'amd-vitis', 'runtime_tag': 'test',
                               'python': sys.executable})
            args = argparse.Namespace(config=str(cfg), action='run', bundle=str(bundle),
                                      cache=str(root / 'cache'), result=str(root / 'result'))
            with mock.patch.dict(backends.BACKENDS, {'amd-vitis': BrokenBackend}):
                with self.assertRaisesRegex(RuntimeError, 'compiler failure'):
                    E.worker(args)
                with self.assertRaisesRegex(RuntimeError, 'quarantined'):
                    E.worker(args)
            self.assertEqual(BrokenBackend.calls, 1)

    def test_optional_real_cpu_training_export(self):
        if importlib.util.find_spec('onnx') is None:
            self.skipTest('onnx needed for actual model-export check')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / 'data'
            data.mkdir()
            rng = np.random.default_rng(12)
            for i in range(2):
                size = 16
                prev = rng.random((size, size)).astype('<f2')
                cur = np.roll(prev, 1, axis=1)
                field = np.zeros((size, size, 4), '<f2')
                field[..., 2] = .6
                raw = struct.pack('<8IQ', 0x44465347, 1, size, size, 2, 8, 1, 0, i)
                raw += prev.tobytes() + cur.tobytes() + field.tobytes() * 2
                (data / f'fg_{i:04d}.bin').write_bytes(raw)
            args = argparse.Namespace(out=str(root / 'bundle'), data=str(data), net=None,
                                      net_key=None, profile_key=0x100f0,
                                      train_steps=2, crop=8, training_timeout=60)
            E.prepare(args)
            bundle = root / 'bundle'
            manifest = json.loads((bundle / 'manifest.json').read_text())
            self.assertEqual(manifest['profile_key'], 0x100f0)
            self.assertFalse(manifest['quality_validated'])
            self.assertTrue(all(E.digest(bundle / p) == h for p, h in manifest['files'].items()))
            args.net = str(bundle / 'candidate.gsfr')
            args.net_key = 0xf0
            args.out = str(root / 'incorrect-role')
            with self.assertRaisesRegex(ValueError, 'must match'):
                E.prepare(args)
            self.assertFalse(Path(args.out).exists())
            if importlib.util.find_spec('onnxruntime') is not None:
                import onnxruntime as ort
                session = ort.InferenceSession(str(bundle / 'model.onnx'),
                                               providers=['CPUExecutionProvider'])
                with np.load(bundle / 'fixture.npz') as fixture:
                    for x, expected in zip(fixture['features'], fixture['reference']):
                        np.testing.assert_allclose(session.run(None, {'features': x})[0],
                                                   expected, atol=1e-6, rtol=1e-5)


class CampaignContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = {
            'config': str(self.root / 'config.json'), 'cache': str(self.root / 'cache'),
            'out': str(self.root / 'out'), 'runs': 2, 'atol': .001, 'rtol': .05,
            'timeout': 5, 'benches': [{
                'name': 'bench', 'train': str(self.root / 'train'),
                'heldout': str(self.root / 'heldout'), 'profiles': [{
                    'name': 'gpu', 'net': str(self.root / 'profile.bin'),
                    'key': 0x100f0, 'npu': True}],
                'cpu': {'steps': 2, 'crop': 8, 'profile_key': 0x100f0, 'npu': False}}]}

    def fixture(self):
        E.write_json(self.root / 'config.json', {
            'backend': 'amd-vitis', 'runtime_tag': 'test', 'python': '/bin/false'})
        rng = np.random.default_rng(12)
        for folder in ('train', 'heldout'):
            data = self.root / folder
            data.mkdir()
            for i in range(2):
                prev = rng.random((16, 16)).astype('<f2')
                cur = np.roll(prev, 1, axis=1)
                field = np.zeros((16, 16, 4), '<f2')
                field[..., 2] = .6
                raw = struct.pack('<8IQ', 0x44465347, 1, 16, 16, 2, 8, 1, 0, i)
                raw += prev.tobytes() + cur.tobytes() + field.tobytes() * 2
                (data / f'fg_{i:04d}.bin').write_bytes(raw)
        self.trainer = C.module('framegen-net-train')
        self.params = self.trainer.init_params(rng)
        self.params['b3'][0] = .1
        blob = self.root / 'weights.gsfr'
        self.trainer.export(self.params, blob)
        (self.root / 'profile.bin').write_bytes(
            struct.pack('<4I', 0x50465347, 1, 1, 0x100f0) + blob.read_bytes())

    def fake_prepare(self, args):
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        self.trainer.export(self.params, out / 'candidate.gsfr')
        for p in ('model.onnx', 'fixture.npz'):
            (out / p).write_bytes(b'CPU-only mock export')
        E.write_json(out / 'manifest.json', {
            'schema': 1, 'cpu_train_steps': args.train_steps,
            'cpu_train_seconds': 1.25 if args.train_steps else None,
            'files': {p: E.digest(out / p) for p in
                      ('candidate.gsfr', 'model.onnx', 'fixture.npz')}})

    def fake_supervise(self, args):
        self.assertEqual(args.action, 'run')
        self.assertEqual(args.config, self.spec['config'])
        self.assertEqual(args.runs, 2)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        E.write_json(out / 'result.json', {
            'status': 'npu-raw-head-parity-passed', 'load_ms': 12,
            'measurement': {'call_ms_p50': 2, 'call_ms_p99': 3,
                            'max_abs_error': .0001, 'atol': .001, 'rtol': .05}})

    def launch(self):
        path = self.root / 'spec.json'
        E.write_json(path, self.spec)
        C.campaign(argparse.Namespace(spec=str(path)))

    def test_strict_spec_validation(self):
        self.assertEqual(C.validate_spec(self.spec), self.spec)
        variants = []
        for key, value in [('config', 'relative.json'), ('cache', '/tmp'),
                           ('out', 42), ('runs', True), ('runs', 0),
                           ('timeout', -1), ('atol', float('nan')), ('rtol', -1),
                           ('benches', []), ('extra', 1)]:
            s = copy.deepcopy(self.spec)
            if key == 'cache':
                del s[key]
            else:
                s[key] = value
            variants.append(s)
        for target, key, value in [('bench', 'name', '../escape'),
                                   ('bench', 'train', 'relative'),
                                   ('bench', 'extra', 0), ('profile', 'npu', 1),
                                   ('profile', 'name', 'neutral'),
                                   ('profile', 'net', 'relative'),
                                   ('profile', 'key', 0x100000000),
                                   ('cpu', 'steps', 1), ('cpu', 'crop', False),
                                   ('cpu', 'profile_key', -1)]:
            s = copy.deepcopy(self.spec)
            b = s['benches'][0]
            t = b if target == 'bench' else b['cpu'] if target == 'cpu' else b['profiles'][0]
            t[key] = value
            variants.append(s)
        for kind in ('bench', 'profile'):
            s = copy.deepcopy(self.spec)
            target = s['benches'] if kind == 'bench' else s['benches'][0]['profiles']
            target.append(copy.deepcopy(target[0]))
            variants.append(s)
        for s in variants:
            with self.subTest(spec=s), self.assertRaises(ValueError):
                C.validate_spec(s)

    def test_resume_and_summary_with_real_evaluator(self):
        self.fixture()
        with mock.patch.object(C.preparation, 'prepare', side_effect=self.fake_prepare) as prep, \
                mock.patch.object(E, 'supervise', side_effect=self.fake_supervise) as npu:
            self.launch()
            self.launch()
        self.assertEqual(prep.call_count, 2)
        self.assertEqual(npu.call_count, 1)
        out = Path(self.spec['out'])
        report = json.loads((out / 'campaign.json').read_text())
        self.assertEqual(report['status'], 'complete')
        self.assertFalse(report['quality_validated'])
        neutral, gpu, cpu = report['benches'][0]['rows']
        self.assertEqual([r['name'] for r in (neutral, gpu, cpu)], ['neutral', 'gpu', 'cpu'])
        self.assertEqual(cpu['cpu_train_steps'], 2)
        self.assertEqual(cpu['cpu_train_seconds'], 1.25)
        self.assertTrue(gpu['npu']['parity']['passed'])
        self.assertEqual(cpu['npu']['status'], 'not-requested')
        self.assertEqual(gpu['npu']['p99_ms'], 3)
        for k in C.METRICS:
            self.assertAlmostEqual(cpu['delta_vs_neutral'][k], cpu['metrics'][k] - neutral['metrics'][k])
        md = (out / 'campaign.md').read_text()
        for text in ('not served RGB', 'host call incl. copies', 'conf-resid',
                     'temporal_sigma', 'CPU train steps/seconds', 'delta vs neutral',
                     '| neutral |', '| gpu |', '| cpu |', 'passed; max_abs_error='):
            self.assertIn(text, md)
        candidate = out / 'bench/bundle-cpu/candidate.gsfr'
        candidate.write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'stale/incomplete campaign step'):
            self.launch()

    def test_resume_completed_cpu_steps_after_npu_failure(self):
        self.fixture()
        def fail(args):
            Path(args.out).mkdir(parents=True)
            E.write_json(Path(args.out) / 'result.json', {'status': 'failed'})
            raise RuntimeError('mock NPU failure')
        with mock.patch.object(C.preparation, 'prepare', side_effect=self.fake_prepare) as prep, \
                mock.patch.object(E, 'supervise', side_effect=fail) as npu:
            with self.assertRaisesRegex(RuntimeError, 'mock NPU failure'):
                self.launch()
            with self.assertRaisesRegex(RuntimeError, 'stale/incomplete campaign step'):
                self.launch()
        self.assertEqual(prep.call_count, 2)
        self.assertEqual(npu.call_count, 1)
        report = json.loads((Path(self.spec['out']) / 'campaign.json').read_text())
        self.assertEqual(report['status'], 'stopped')
        self.assertEqual(report['benches'][0]['rows'][1]['npu']['status'], 'failed-or-incomplete')

    def test_stale_inputs_and_incomplete_steps_preserved(self):
        self.fixture()
        with mock.patch.object(C.preparation, 'prepare', side_effect=self.fake_prepare), \
                mock.patch.object(E, 'supervise', side_effect=self.fake_supervise):
            self.launch()
        (self.root / 'profile.bin').write_bytes(b'changed input')
        with self.assertRaisesRegex(RuntimeError, 'stale/incomplete campaign root'):
            self.launch()
        path = self.root / 'partial'
        path.mkdir()
        (path / 'diagnostic.log').write_text('keep me')
        with self.assertRaisesRegex(RuntimeError, 'stale/incomplete campaign step'):
            C.step(path, {}, mock.Mock())
        self.assertEqual((path / 'diagnostic.log').read_text(), 'keep me')


if __name__ == '__main__':
    unittest.main()
