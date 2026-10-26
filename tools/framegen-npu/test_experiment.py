"""CPU-only contracts. No accelerator or optional SDK required."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import struct
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backends
import experiment as E
import campaign as C
import preparation as P
from common import Terminated, termination_signals


def worker_fixture(root):
    bundle = root / 'bundle'
    bundle.mkdir()
    for name in ('model.onnx', 'candidate.gsfr'):
        (bundle / name).write_bytes(b'CPU-only fake')
    features = np.ones((4, 1, 12, 8, 8), np.float32)
    reference = np.ones((4, 1, 4, 2, 2), np.float32)
    reference *= np.arange(1, 5, dtype=np.float32)[:, None, None, None, None]
    np.savez(bundle / 'fixture.npz', features=features, reference=reference)
    E.write_json(bundle / 'manifest.json', {'schema': 1, 'files': {
        p: E.digest(bundle / p) for p in ('model.onnx', 'fixture.npz', 'candidate.gsfr')}})
    cfg = root / 'config.json'
    E.write_json(cfg, {'backend': 'amd-vitis', 'runtime_tag': 'test', 'python': sys.executable})
    return argparse.Namespace(config=str(cfg), action='run', bundle=str(bundle),
                              cache=str(root / 'cache'), result=str(root / 'result.json'),
                              runs=1, atol=0, rtol=0)


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def run(self, _):
        return next(self.outputs)

    def evidence(self):
        return {'test_only': True}


def supervised_fixture(root, phase='load', exit_code=None):
    """Patch only the child runtime; launch the real worker with a real interpreter."""
    args = worker_fixture(root)
    (root / 'sitecustomize.py').write_text(f'''
import hashlib, os
from pathlib import Path
import backends
def blocked(*_):
    Path({str(root / 'worker.pid')!r}).write_text(str(os.getpid()))
    if {exit_code!r} is not None:
        os._exit({exit_code!r})
    # One long native call defers Python signal handlers, like an SDK compile.
    hashlib.pbkdf2_hmac('sha256', b'x', b'y', 1_000_000_000)
class CpuOnlyBackend:
    def probe(self):
        return {{'available': True}}
    def load(self, *_):
        if {phase!r} == 'load':
            blocked()
backends.BACKENDS['amd-vitis'] = CpuOnlyBackend
backends.inventory = lambda: []
backends.measure = blocked
''')
    cfg = json.loads(Path(args.config).read_text())
    cfg['environment'] = {'PYTHONPATH': os.pathsep.join((str(root), str(E.HERE)))}
    E.write_json(Path(args.config), cfg)
    args.out, args.timeout = str(root / 'run'), 20
    return args


def supervisor_command(args):
    return [sys.executable, '-c', f'''
import sys, argparse
sys.path.insert(0, {str(E.HERE)!r})
import experiment
experiment.supervise(argparse.Namespace(**{vars(args)!r}))
''']


class Contracts(unittest.TestCase):
    def test_termination_preserves_nohup_and_restores_handlers(self):
        previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        term = signal.getsignal(signal.SIGTERM)
        try:
            with termination_signals():
                self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
                self.assertNotEqual(signal.getsignal(signal.SIGTERM), term)
            self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
            self.assertEqual(signal.getsignal(signal.SIGTERM), term)
        finally:
            signal.signal(signal.SIGHUP, previous)

    def test_termination_escapes_exception_and_protects_cleanup(self):
        previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
        with termination_signals():
            handler = signal.getsignal(signal.SIGTERM)
            with self.assertRaises(Terminated):
                try:
                    handler(signal.SIGTERM, None)
                except Exception:
                    self.fail('termination swallowed by a library handler')
            for sig in previous:
                self.assertEqual(signal.getsignal(sig), signal.SIG_IGN)
        for sig, handler in previous.items():
            self.assertEqual(signal.getsignal(sig), signal.SIG_IGN)
            signal.signal(sig, handler)

    def test_measure_rejects_wrong_finite_nan_and_shape(self):
        x = np.ones((2, 1, 1, 2, 2), np.float32)
        x[1] *= 2
        for y in (np.zeros_like(x[0]), x[0] * np.nan, x[0][..., :1]):
            with self.subTest(shape=y.shape), self.assertRaises(RuntimeError):
                backends.measure(FakeBackend([y] * 4), x, x, 1, .001, .01)

    def test_measure_counts_both_directions_and_omits_warmup(self):
        x = np.ones((2, 1, 1, 2, 2), np.float32)
        x[1] *= 2
        result = backends.measure(FakeBackend([x[0]] * 3 + [x[1]] * 3), x, x, 2, 0, 0)
        self.assertEqual(result['calls'], 4)
        self.assertEqual(result['directions'], 2)
        self.assertEqual(result['max_abs_error'], 0)

    def test_parity_failure_exposes_max_abs_error_including_warmup(self):
        x = np.ones((2, 1, 1, 2, 2), np.float32)
        x[1] *= 2
        for worst_call in range(4):
            outputs = [x[0] + .125, x[0].copy(), x[1].copy(), x[1].copy()]
            outputs[worst_call] = outputs[worst_call].copy()
            outputs[worst_call][..., 0, 0] += .5
            expected = .625 if worst_call == 0 else .5
            with self.subTest(worst_call=worst_call), self.assertRaises(backends.ParityError) as caught:
                backends.measure(FakeBackend(outputs), x, x, 1, .001, .01)
            self.assertEqual(caught.exception.parity['max_abs_error'], expected)
            self.assertIn(f'max_abs_error={expected}', str(caught.exception))

    def test_compiled_cache_survives_parity_failure_and_warm_load_failure(self):
        class Backend:
            def probe(self):
                return {'available': True}

            def load(self, _model, cache, _options):
                states.append(json.loads((cache / 'state.json').read_text())['status'])
                if fail_load[0]:
                    raise RuntimeError('warm load interrupted')

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = worker_fixture(root)
            states, fail_load = [], [False]
            with mock.patch.dict(backends.BACKENDS, {'amd-vitis': Backend}), \
                    mock.patch.object(backends, 'inventory', return_value=[]), \
                    mock.patch.object(backends, 'measure', return_value={'test_only': True}) as measure:
                E.run_worker(args)
                result = json.loads(Path(args.result).read_text())
                self.assertEqual(result['cache_state_before'], 'cold')
                self.assertIn('load_ms', result)
                measure.side_effect = backends.ParityError(.25, 0, 0)
                with self.assertRaises(backends.ParityError):
                    E.run_worker(args)
                result = json.loads(Path(args.result).read_text())
                self.assertEqual(result['cache_state_before'], 'warm')
                self.assertEqual(result['parity']['max_abs_error'], .25)
                state = root / 'cache' / result['cache_key'] / 'state.json'
                saved = json.loads(state.read_text())
                self.assertEqual(saved['status'], 'failed')
                self.assertTrue(saved['compiled'])
                self.assertIn('max_abs_error=0.25', saved['reason'])
                measure.side_effect = None
                E.run_worker(args)
                fail_load[0] = True
                with self.assertRaisesRegex(RuntimeError, 'warm load interrupted'):
                    E.run_worker(args)
                fail_load[0] = False
                E.run_worker(args)
            self.assertEqual(states, ['building', 'ready', 'failed', 'ready', 'failed'])

    def test_fixture_selection_uses_motion_and_confidence_not_endpoints(self):
        paths = [Path(f'fg_{i}.bin') for i in range(5)]
        fields = []
        for motion, confidence in ((0, 0), (8, .1), (2, .9), (1, .4), (0, 0)):
            f = np.zeros((2, 2, 4), np.float32)
            f[..., 0], f[..., 2] = motion, confidence
            fields.append((np.zeros((2, 2)), np.ones((2, 2)), f, f))
        trainer = mock.Mock(load_sample=lambda p: fields[paths.index(p)])
        with mock.patch.object(P, 'module', return_value=trainer):
            self.assertEqual(P.select_fixtures(paths), (paths[1], paths[2]))
            fields[1][2][..., 2] = 1  # the moving pair now wins both scores
            self.assertEqual(P.select_fixtures(paths), (paths[1], paths[2]))
            fields[1][2][0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, 'non-finite'):
                P.select_fixtures(paths)

    def test_parent_death_setup_checks_race(self):
        with mock.patch('ctypes.CDLL') as libc, mock.patch.object(E.os, 'getppid', return_value=2):
            libc.return_value.prctl.return_value = 0
            with self.assertRaisesRegex(RuntimeError, 'supervisor exited'):
                E.arm_parent_death(1)
            libc.return_value.prctl.assert_called_once_with(1, signal.SIGKILL, 0, 0, 0)

    def test_worker_signal_handler_flushes_failure_without_signalling_runner(self):
        def interrupt(_args):
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler), 'worker termination handler is missing')
            handler(signal.SIGTERM, None)
        with tempfile.TemporaryDirectory() as d:
            args = argparse.Namespace(result=str(Path(d) / 'result.json'), parent_pid=123)
            with mock.patch.object(E, 'arm_parent_death') as arm, \
                    mock.patch.object(E, 'run_worker', side_effect=interrupt), \
                    self.assertRaises(Terminated):
                E.worker(args)
            arm.assert_called_once_with(123)
            saved = json.loads(Path(args.result).read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertIn('SIGTERM', saved['reason'])

    def test_prepare_uses_motion_and_distinct_confidence_fixtures(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / 'data'
            data.mkdir()
            paths = [data / f'fg_{i}.bin' for i in range(5)]
            samples = {}
            for path, motion, confidence in zip(paths, (0, 8, 2, 1, 0), (0, 1, .9, .4, 0)):
                path.write_bytes(b'frozen sample')
                f = np.zeros((2, 2, 4), np.float32)
                f[..., 0], f[..., 2] = motion, confidence
                samples[path] = (np.zeros((2, 2)), np.ones((2, 2)), f, f)
            args = argparse.Namespace(out=str(root / 'bundle'), data=str(data), net=None,
                                      net_key=None, profile_key=0x100f0,
                                      train_steps=2, crop=8, training_timeout=5)
            def candidate(_args, out):
                (out / 'candidate.gsfr').write_bytes(b'candidate')
                return {'W3': np.ones(1), 'b3': np.zeros(1)}, .1
            def export(_params, _input, _output, path):
                path.write_bytes(b'mock export')
            tensors = (np.ones((4, 1, 12, 8, 8), np.float32),
                       np.ones((4, 1, 4, 2, 2), np.float32))
            with mock.patch.dict(sys.modules, {'onnx': mock.Mock()}), \
                    mock.patch.object(P, 'candidate_params', side_effect=candidate), \
                    mock.patch.object(P, 'module', return_value=mock.Mock(load_sample=samples.__getitem__)), \
                    mock.patch.object(P, 'fixtures', return_value=tensors) as fixtures, \
                    mock.patch.object(P, 'export_model', side_effect=export):
                P.prepare(args)
            self.assertEqual(fixtures.call_args.args[1], (paths[1], paths[2]))
            manifest = json.loads((root / 'bundle/manifest.json').read_text())
            self.assertEqual(manifest['fixture_files'], {str(p): E.digest(p) for p in paths[1:3]})
            self.assertEqual(manifest['fixture_selection'],
                             'highest-mean-motion-then-distinct-highest-mean-confidence')

    def test_supervisor_preserves_parity_diagnostics(self):
        with tempfile.TemporaryDirectory() as d:
            result = Path(d) / 'result.json'
            E.write_json(result, {'status': 'failed', 'parity': {'max_abs_error': .5},
                                  'reason': 'parity failed'})
            E.failed_receipt(result, 'failed', 'worker exit', worker_exit=1)
            saved = json.loads(result.read_text())
            self.assertEqual(saved['parity']['max_abs_error'], .5)
            self.assertEqual(saved['reason'], 'parity failed')
            E.failed_receipt(result, 'aborted', 'timeout after 2 seconds')
            saved = json.loads(result.read_text())
            self.assertEqual(saved['reason'], 'timeout after 2 seconds')
            self.assertEqual(saved['parity']['max_abs_error'], .5)

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
            self.assertIn('timeout', json.loads((root / 'run/result.json').read_text())['reason'])

    def run_blocked_supervisor(self, phase, cause):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            args = supervised_fixture(root, phase, 7 if cause == 'exit' else None)
            if cause == 'timeout':
                args.timeout = 5
            with (root / 'supervisor.log').open('w') as log:
                proc = subprocess.Popen(supervisor_command(args), stdout=log,
                                        stderr=subprocess.STDOUT)
                pid = None
                try:
                    deadline = time.monotonic() + 8
                    while not (root / 'worker.pid').exists() and time.monotonic() < deadline:
                        if proc.poll() is not None:
                            self.fail((root / 'supervisor.log').read_text())
                        time.sleep(.02)
                    self.assertTrue((root / 'worker.pid').exists())
                    pid = int((root / 'worker.pid').read_text())
                    if cause in ('signal', 'parent-death'):
                        # Allow the child to enter its long native call.
                        time.sleep(.1)
                        proc.send_signal(signal.SIGTERM if cause == 'signal' else signal.SIGKILL)
                    proc.wait(timeout=8)
                    self.assertEqual(proc.returncode, -signal.SIGKILL if cause == 'parent-death' else 1)
                    result = json.loads((root / 'run/result.json').read_text())
                    state = json.loads((root / 'cache' / result['cache_key'] / 'state.json').read_text())
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        stat = Path(f'/proc/{pid}/stat')
                        dead = not stat.exists() or stat.read_text().split()[2] == 'Z'
                        if dead:
                            break
                        time.sleep(.02)
                    self.assertTrue(dead, 'worker survived its supervisor')
                    if cause == 'parent-death':
                        # SIGKILL cannot flush receipts: the half compile stays quarantined.
                        self.assertEqual(result['status'], 'running')
                        self.assertEqual(state['status'], 'building')
                        self.assertFalse(state.get('compiled', False))
                    else:
                        self.assertEqual(result['status'], 'failed' if cause == 'exit' else 'aborted')
                        self.assertIn({'signal': 'SIGTERM', 'timeout': 'timeout',
                                       'exit': 'status 7'}[cause], result['reason'])
                        self.assertEqual(state['status'], 'failed')
                        self.assertEqual(state['compiled'], phase == 'measure')
                        self.assertEqual(state['reason'], result['reason'])
                finally:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait()
                    if pid is not None:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

    def test_parent_death_kills_worker_blocked_in_native_call(self):
        self.run_blocked_supervisor('load', 'parent-death')

    def test_supervisor_escalates_and_cleans_interrupted_cache(self):
        for phase in ('load', 'measure'):
            for cause in ('signal', 'timeout', 'exit'):
                with self.subTest(phase=phase, cause=cause):
                    self.run_blocked_supervisor(phase, cause)

    def test_forced_kill_marks_cache_failed_without_losing_compile(self):
        for status in ('building', 'ready'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                state = root / 'key/state.json'
                state.parent.mkdir()
                E.write_json(state, {'status': status, 'identity': {}})
                E.interrupted_cache(argparse.Namespace(cache=str(root)),
                                    {'cache_key': 'key', 'reason': 'timeout'})
                saved = json.loads(state.read_text())
                self.assertEqual(saved['status'], 'failed')
                self.assertEqual(saved['reason'], 'timeout')
                self.assertEqual(saved['compiled'], status == 'ready')

    def test_interrupted_cache_does_not_touch_live_owner(self):
        import fcntl
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = root / 'key/state.json'
            state.parent.mkdir()
            E.write_json(state, {'status': 'building', 'identity': {}})
            before = state.read_bytes()
            with (root / 'key.lock').open('w') as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                E.interrupted_cache(argparse.Namespace(cache=str(root)),
                                    {'cache_key': 'key', 'reason': 'timeout'})
            self.assertEqual(state.read_bytes(), before)

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
                    E.run_worker(args)
                with self.assertRaisesRegex(RuntimeError, 'quarantined'):
                    E.run_worker(args)
            self.assertEqual(BrokenBackend.calls, 1)
            state = next((root / 'cache').glob('*/state.json'))
            saved = json.loads(state.read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertFalse(saved['compiled'])
            self.assertIn('compiler failure', saved['reason'])

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
        self.assertEqual(args.timeout, C.timeouts(self.spec)[1])
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        E.write_json(out / 'result.json', {
            'status': 'npu-raw-head-parity-passed', 'load_ms': 12, 'cache_state_before': 'warm',
            'measurement': {'call_ms_p50': 2, 'call_ms_p99': 3,
                            'max_abs_error': .0001, 'atol': .001, 'rtol': .05}})

    def fake_candidate_params(self, args, out):
        self.assertEqual(args.training_timeout, C.timeouts(self.spec)[0])
        self.trainer.export(self.params, out / 'candidate.gsfr')
        return self.params, 1.25

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

    def test_separate_timeouts_and_legacy_fallback_validation(self):
        for timeouts, expected in (({'train_timeout': 3, 'npu_timeout': 90}, (3, 90)),
                                   ({'timeout': 5, 'train_timeout': 3}, (3, 5)),
                                   ({'timeout': 5, 'npu_timeout': 90}, (5, 90))):
            spec = {k: v for k, v in self.spec.items() if k != 'timeout'}
            spec.update(timeouts)
            self.assertEqual(C.validate_spec(spec), spec)
            self.assertEqual(C.timeouts(spec), expected)
        for timeouts in ({}, {'train_timeout': 3}, {'npu_timeout': 90},
                         {'train_timeout': 0, 'npu_timeout': 90},
                         {'train_timeout': 3, 'npu_timeout': True}):
            spec = {k: v for k, v in self.spec.items() if k != 'timeout'}
            spec.update(timeouts)
            with self.subTest(timeouts=timeouts), self.assertRaises(ValueError):
                C.validate_spec(spec)

    def test_separate_timeouts_reach_training_and_npu(self):
        self.fixture()
        del self.spec['timeout']
        self.spec.update(train_timeout=3, npu_timeout=90)
        self.spec['benches'][0]['cpu']['npu'] = True
        def prepare(args):
            self.assertEqual(args.training_timeout, 3)
            self.fake_prepare(args)
        with mock.patch.object(C.preparation, 'prepare', side_effect=prepare) as prep, \
                mock.patch.object(E, 'supervise', side_effect=self.fake_supervise) as npu:
            self.launch()
        self.assertEqual(prep.call_count, 2)
        self.assertEqual(npu.call_count, 2)

    def test_no_npu_sources_skip_export_even_with_zero_head(self):
        self.fixture()
        self.spec['benches'][0]['profiles'][0]['npu'] = False
        self.params['W3'].fill(0)
        self.params['b3'].fill(0)
        blob = self.root / 'weights.gsfr'
        self.trainer.export(self.params, blob)
        (self.root / 'profile.bin').write_bytes(
            struct.pack('<4I', 0x50465347, 1, 1, 0x100f0) + blob.read_bytes())
        with mock.patch.object(C.preparation, 'prepare') as prep, \
                mock.patch.object(C.preparation, 'candidate_params',
                                  side_effect=self.fake_candidate_params) as train, \
                mock.patch.object(E, 'supervise') as npu:
            self.launch()
            self.launch()
        prep.assert_not_called()
        npu.assert_not_called()
        train.assert_called_once()
        out = Path(self.spec['out']) / 'bench'
        self.assertFalse(list(out.glob('bundle-*')))
        self.assertTrue((out / 'train-cpu/candidate.gsfr').exists())

    def test_real_cpu_training_without_exporter(self):
        self.fixture()
        source = {'name': 'cpu', **self.spec['benches'][0]['cpu'], 'train_timeout': 30}
        out = self.root / 'cpu-only'
        with mock.patch.object(P, 'export_model', side_effect=AssertionError('unexpected export')):
            result = C.train_cpu(out, self.spec['benches'][0], source)
        self.assertEqual(result['cpu_train_steps'], 2)
        self.assertGreater(result['cpu_train_seconds'], 0)
        params = C.module('framegen-net-eval').load_blob(out / 'candidate.gsfr')
        self.assertTrue(np.any(params['W3']) or np.any(params['b3']))
        self.assertFalse((out / 'model.onnx').exists())

    def test_campaign_signal_records_stopped_receipt(self):
        self.fixture()
        spec = self.root / 'spec.json'
        E.write_json(spec, self.spec)
        script = f'''
import argparse, os, signal, sys
sys.path.insert(0, {str(E.HERE)!r})
import campaign
campaign.evaluate = lambda *_: os.kill(os.getpid(), signal.SIGTERM)
campaign.campaign(argparse.Namespace(spec={str(spec)!r}))
'''
        proc = subprocess.run([sys.executable, '-c', script], capture_output=True,
                              text=True, timeout=10)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        saved = json.loads((Path(self.spec['out']) / 'campaign.json').read_text())
        self.assertEqual(saved['status'], 'stopped')
        self.assertIn('SIGTERM', saved['error'])

    def test_second_signal_during_campaign_cleanup_records_stopped(self):
        self.fixture()
        spec = self.root / 'spec.json'
        E.write_json(spec, self.spec)
        script = f'''
import argparse, os, signal, sys
sys.path.insert(0, {str(E.HERE)!r})
import campaign as C
from common import Terminated, termination_signals
# Keep campaign orchestration/receipt writes real; substitute the SDK and CPU work.
C.evaluate = lambda *_: dict.fromkeys(C.METRICS, 0.0)
C.prepare = C.train_cpu = lambda *_: {{'cpu_train_steps': 0, 'cpu_train_seconds': None}}
C.step = lambda path, request, operation: operation(path)
def run_npu(*_):
    with termination_signals():
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Terminated:
            raise RuntimeError('helper aborted') from None
C.run_npu = run_npu
write = C.write_json
def cleanup_write(path, value):
    if path.name == 'campaign.json' and value.get('status') == 'stopped':
        os.kill(os.getpid(), signal.SIGTERM)
    write(path, value)
C.write_json = cleanup_write
C.campaign(argparse.Namespace(spec={str(spec)!r}))
'''
        proc = subprocess.run([sys.executable, '-c', script], capture_output=True,
                              text=True, timeout=10)
        self.assertEqual(proc.returncode, 1, proc.stderr)
        saved = json.loads((Path(self.spec['out']) / 'campaign.json').read_text())
        self.assertEqual(saved['status'], 'stopped')
        self.assertEqual(saved['error'], 'helper aborted')
        self.assertEqual(saved['benches'][0]['rows'][1]['npu']['status'],
                         'failed-or-incomplete')

    def test_resume_and_summary_with_real_evaluator(self):
        self.fixture()
        with mock.patch.object(C.preparation, 'prepare', side_effect=self.fake_prepare) as prep, \
                mock.patch.object(C.preparation, 'candidate_params', side_effect=self.fake_candidate_params), \
                mock.patch.object(E, 'supervise', side_effect=self.fake_supervise) as npu:
            self.launch()
            self.launch()
        self.assertEqual(prep.call_count, 1)
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
        self.assertEqual(gpu['npu']['cache_state_before'], 'warm')
        for k in C.METRICS:
            self.assertAlmostEqual(cpu['delta_vs_neutral'][k], cpu['metrics'][k] - neutral['metrics'][k])
        md = (out / 'campaign.md').read_text()
        header = next(line for line in md.splitlines() if line.startswith('| source |'))
        cells = next(line for line in md.splitlines() if line.startswith('| gpu |'))
        self.assertEqual(cells.split('|')[header.split('|').index(' cache before ')].strip(), 'warm')
        for text in ('not served RGB', 'host call incl. copies', 'conf-resid',
                     'temporal_sigma', 'CPU train steps/seconds', 'delta vs neutral',
                     '| neutral |', '| gpu |', '| cpu |', 'passed; max_abs_error='):
            self.assertIn(text, md)
        self.assertFalse((out / 'bench/bundle-cpu').exists())
        candidate = out / 'bench/train-cpu/candidate.gsfr'
        candidate.write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'stale/incomplete campaign step'):
            self.launch()

    def test_resume_completed_cpu_steps_after_npu_failure(self):
        self.fixture()
        def fail(args):
            Path(args.out).mkdir(parents=True)
            E.write_json(Path(args.out) / 'result.json', {
                'status': 'failed', 'reason': 'parity tolerance exceeded',
                'cache_state_before': 'warm', 'load_ms': 1,
                'parity': {'passed': False, 'max_abs_error': .25, 'atol': 0, 'rtol': 0}})
            raise RuntimeError('mock NPU failure')
        with mock.patch.object(C.preparation, 'prepare', side_effect=self.fake_prepare) as prep, \
                mock.patch.object(C.preparation, 'candidate_params', side_effect=self.fake_candidate_params), \
                mock.patch.object(E, 'supervise', side_effect=fail) as npu:
            with self.assertRaisesRegex(RuntimeError, 'mock NPU failure'):
                self.launch()
            with self.assertRaisesRegex(RuntimeError, 'stale/incomplete campaign step'):
                self.launch()
        self.assertEqual(prep.call_count, 1)
        self.assertEqual(npu.call_count, 1)
        report = json.loads((Path(self.spec['out']) / 'campaign.json').read_text())
        self.assertEqual(report['status'], 'stopped')
        self.assertEqual(report['benches'][0]['rows'][1]['npu']['status'], 'failed-or-incomplete')
        # Resuming the failed step keeps the original run's numerical receipt intact.
        self.assertEqual(report['benches'][0]['rows'][1]['npu']['parity']['max_abs_error'], .25)
        self.assertEqual(
            {k: report['benches'][0]['rows'][1]['npu'][k]
             for k in ('reason', 'load_ms', 'cache_state_before')},
            {'reason': 'parity tolerance exceeded', 'load_ms': 1, 'cache_state_before': 'warm'})
        md = (Path(self.spec['out']) / 'campaign.md').read_text()
        self.assertIn('failed; max_abs_error=0.25', md)

    def test_stale_inputs_and_incomplete_steps_preserved(self):
        self.fixture()
        with mock.patch.object(C.preparation, 'prepare', side_effect=self.fake_prepare), \
                mock.patch.object(C.preparation, 'candidate_params', side_effect=self.fake_candidate_params), \
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
