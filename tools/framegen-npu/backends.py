"""Optional runtime adapters. Imported only in the experimental helper process."""

from pathlib import Path
import json
import time


def inventory():
    """Linux accelerator nodes are hints, never proof of executable support."""
    devices = []
    for node in sorted(Path('/sys/class/accel').glob('accel*')):
        dev = node / 'device'
        info = {'node': str(Path('/dev/accel') / node.name)}
        for name in ('vendor', 'device', 'revision'):
            if (dev / name).exists():
                info[name] = (dev / name).read_text().strip()
        driver = dev / 'driver'
        if driver.exists():
            info['driver'] = driver.resolve().name
            version = driver / 'module/version'
            if version.exists():
                info['driver_version'] = version.read_text().strip()
        devices.append(info)
    return devices


class AmdOrt:
    name = 'amd-vitis'
    provider = 'VitisAIExecutionProvider'

    def __init__(self):
        import onnxruntime as ort
        self.ort = ort

    def probe(self):
        return {'runtime': 'onnxruntime', 'version': self.ort.__version__,
                'available': self.provider in self.ort.get_available_providers(),
                'providers': self.ort.get_available_providers()}

    def load(self, model, cache, options):
        opts = self.ort.SessionOptions()
        opts.intra_op_num_threads = opts.inter_op_num_threads = 1
        opts.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        opts.enable_profiling = True
        opts.profile_file_prefix = str(cache / 'ort-profile')
        provider_options = dict(options)
        # The helper owns cache identity; caller options cannot override it.
        provider_options.update(cache_dir=str(cache), cache_key='compiled',
                                enable_cache_file_io_in_mem='0')
        self.session = self.ort.InferenceSession(
            str(model), sess_options=opts, providers=[self.provider],
            provider_options=[provider_options])
        self.session.disable_fallback()
        # Some ORT versions register CPU even with assignment fallback disabled.
        # Registration is not execution; kernel evidence below is authoritative.
        if self.session.get_providers()[0] != self.provider:
            raise RuntimeError('strict NPU session was not established')

    def run(self, features):
        return self.session.run(['head'], {'features': features})[0]

    def evidence(self):
        events = json.loads(Path(self.session.end_profiling()).read_text())
        providers = [e.get('args', {}).get('provider') for e in events
                     if e.get('cat') == 'Node' and e.get('args', {}).get('provider')]
        if not providers or set(providers) != {self.provider}:
            raise RuntimeError(f'no exclusive NPU execution evidence: {set(providers)}')
        return {'provider': self.provider, 'profiled_kernel_events': len(providers)}


class IntelOpenVino:
    name = 'intel-openvino'

    def __init__(self):
        import openvino as ov
        self.ov = ov
        self.core = ov.Core()

    def probe(self):
        devices = self.core.available_devices
        return {'runtime': 'openvino', 'version': self.ov.__version__,
                'available': any(d == 'NPU' or d.startswith('NPU.') for d in devices),
                'devices': devices}

    def load(self, model, cache, options):
        if options:
            raise ValueError('Intel adapter currently accepts no provider options')
        self.compiled = self.core.compile_model(
            str(model), 'NPU', {'CACHE_DIR': str(cache)})

    def run(self, features):
        result = self.compiled({'features': features})
        return next(iter(result.values())).copy()

    def evidence(self):
        devices = list(self.compiled.get_property('EXECUTION_DEVICES'))
        if not devices or any(d != 'NPU' and not d.startswith('NPU.') for d in devices):
            raise RuntimeError(f'non-NPU execution devices: {devices}')
        return {'execution_devices': devices}


BACKENDS = {b.name: b for b in (AmdOrt, IntelOpenVino)}


def measure(backend, features, reference, runs, atol, rtol):
    """Raw-head parity only: not a field, image-quality or gameplay gate."""
    import numpy as np
    if np.allclose(reference, np.zeros_like(reference), atol=atol, rtol=rtol):
        raise RuntimeError('reference cannot distinguish zero output at these tolerances')
    if np.allclose(reference, reference[0:1], atol=atol, rtol=rtol):
        raise RuntimeError('fixtures cannot distinguish repeated/stale output at these tolerances')
    times, errors = [], []
    for direction in range(len(features)):
        x, ref = features[direction], reference[direction]
        for i in range(runs + 1):  # one untimed warmup per direction
            start = time.perf_counter()
            y = np.asarray(backend.run(x))
            elapsed = (time.perf_counter() - start) * 1000
            if y.shape != ref.shape or not np.isfinite(y).all():
                raise RuntimeError('invalid NPU output shape or non-finite values')
            if not np.allclose(y, ref, atol=atol, rtol=rtol):
                raise RuntimeError('NPU output exceeds declared raw-head parity tolerance')
            if i:
                times.append(elapsed)
                errors.append(float(np.max(np.abs(y - ref))))
    return {'calls': len(times), 'directions': len(features),
            'call_ms_p50': float(np.percentile(times, 50)),
            'call_ms_p99': float(np.percentile(times, 99)),
            'call_ms_max': max(times), 'max_abs_error': max(errors),
            'atol': atol, 'rtol': rtol, 'evidence': backend.evidence()}
