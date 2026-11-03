#!/usr/bin/env python3
"""Locked offscreen comparison with the original online GLSL arithmetic.

This validates the plugin's arithmetic and ownership, not production image
interop, frame pacing, live learning or visual quality.
"""
import argparse
import ctypes as C
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess

import numpy as np
import train as T

ROOT = Path(__file__).resolve().parents[2]
P = T.P


def sample(width=17, height=19):
    y, x = np.mgrid[:height, :width].astype(np.float32)
    a = .2 + .011 * x + .017 * y + .025 * np.sin(.7 * x) * np.cos(.37 * y)
    b = .22 + .010 * x + .018 * y + .023 * np.sin(.7 * (x - .4)) * np.cos(.37 * (y + .25))
    forward = np.stack((np.full_like(x, .4), np.full_like(x, -.25),
                        .52 + .03 * np.cos(x), np.full_like(x, .03)), axis=-1)
    reverse = np.stack((np.full_like(x, -.35), np.full_like(x, .2),
                        .57 + .02 * np.sin(y), np.full_like(x, .02)), axis=-1)
    # Same stored tensor value domain as GSFD's half-float planes.
    return tuple(value.astype(np.float16).astype(np.float32) for value in (a, b, forward, reverse))


def prior(zero_head=False):
    rng = np.random.default_rng(20261102)
    weights = rng.normal(0, .025, P).astype(np.float32)
    weights[1728:1744] = .05
    weights[4048:4064] = .04
    weights[4640:] = (0, 0, -.01, 0)
    if zero_head:
        weights[4064:] = 0
    return weights


def initial(weights):
    state = np.zeros((4, P), np.float32)
    state[0] = state[3] = weights
    return state


def reference(args, folder, name, data, weights, state, step, flags, cut):
    height, width = data[0].shape
    input_path, output_path = folder / (name + '.input'), folder / (name + '.output')
    header = struct.pack('<8I3f', 0x54534347, 1, width, height, step, flags, cut, 0,
                         3e-4, .0625, .0005)
    arrays = (weights, state, *data)
    input_path.write_bytes(header + b''.join(np.asarray(a, '<f4').tobytes() for a in arrays))
    command = [str(args.reference), '--device-uuid', args.vulkan_device_uuid,
               '--input', str(input_path), '--output', str(output_path), '--validation']
    environment = os.environ.copy()
    for key in ('DISPLAY', 'WAYLAND_DISPLAY'):
        environment.pop(key, None)
    process = subprocess.Popen(command, env=environment, start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise RuntimeError('Vulkan reference timed out; child retired')
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    (folder / (name + '.stdout')).write_text(stdout)
    (folder / (name + '.stderr')).write_text(stderr)
    if process.returncode:
        raise RuntimeError(f'{name}: Vulkan reference exit {process.returncode}: {stderr}')
    result = np.fromfile(output_path, '<f4')
    if result.size != 21 * P or not np.isfinite(result).all():
        raise RuntimeError('malformed reference output')
    return result[:16 * P].reshape(16, P), result[16 * P:20 * P].reshape(4, P), result[20 * P:]


def compare(name, actual, expected):
    result = {'name': name}
    for label, value, truth, atol, rtol in zip(
            ('gradients', 'state', 'served'), actual, expected,
            (2e-6, 5e-7, 5e-7), (3e-4, 5e-5, 5e-5)):
        if value.shape != truth.shape or not np.isfinite(value).all():
            raise AssertionError(f'{name}: malformed {label}')
        delta = np.abs(value.astype(np.float64) - truth)
        allowed = atol + rtol * np.abs(truth.astype(np.float64))
        result[label] = {'max_abs': float(delta.max()), 'max_scaled': float((delta / allowed).max()),
                         'atol': atol, 'rtol': rtol, 'count': value.size}
        if np.any(delta > allowed):
            raise AssertionError(f'{name} {label}: {result[label]}')
    return result


def check_ownership(context, data, weights, args, folder):
    state = initial(weights)
    data = T.prepare_sample(data)
    height, width = data[0].shape
    job = T.Job(C.sizeof(T.Job), T.MODEL_VERSION, context.epoch, width, height,
                1, 0, 0, 3e-4, .0625, .0005, width * height, 4 * width * height,
                P, 4 * P, *(T.pointer(a) for a in (*data, weights, state)))
    ticket = T.Ticket()
    api, handle, error = context.api, context.handle, context.error
    call = lambda op, *a: op(handle, *a, error, len(error))
    for field, bad in (('size', 0), ('model_version', 99), ('flags', 2),
                       ('epoch', context.epoch + 1), ('state_count', 0), ('width', 0)):
        saved = getattr(job, field)
        setattr(job, field, bad)
        assert call(api.submit, C.byref(job), C.byref(ticket)) == T.INVALID
        setattr(job, field, saved)
    assert call(api.submit, C.byref(job), C.byref(ticket)) == T.OK
    retained = T.Ticket(ticket.epoch, ticket.serial)
    # submit's result ticket may be cleared on rejection; preserve the owner.
    assert call(api.submit, C.byref(job), C.byref(ticket)) == T.BUSY
    assert call(api.reset, C.byref(T.U64())) == T.BUSY
    assert call(api.discard, retained) == T.OK
    assert call(api.reset, C.byref(T.U64())) == T.BUSY
    # A discarded result cannot commit any optimizer state.
    data[0].fill(0)
    import time
    deadline = time.monotonic() + 30
    while call(api.poll, retained) == T.PENDING:
        if time.monotonic() > deadline:
            raise TimeoutError('HIP ownership fixture timed out')
        time.sleep(.001)
    assert call(api.poll, retained) == T.DISCARDED
    assert call(api.retire, retained) == T.OK
    context.reset()
    assert call(api.poll, retained) == T.STALE

    # A separate accepted job checks copied-input ownership by comparing its
    # output with the pre-mutation snapshot, not merely by discarding it.
    data = sample()
    weights = prior()
    state = initial(weights)
    snapshots = tuple(a.copy() for a in data), weights.copy(), state.copy()
    height, width = data[0].shape
    job = T.Job(C.sizeof(T.Job), T.MODEL_VERSION, context.epoch, width, height,
                1, 0, 0, 3e-4, .0625, .0005, width * height, 4 * width * height,
                P, 4 * P, *(T.pointer(a) for a in (*data, weights, state)))
    assert call(api.submit, C.byref(job), C.byref(ticket)) == T.OK
    for value in (*data, weights, state):
        value.fill(7)
    deadline = time.monotonic() + 30
    while call(api.poll, ticket) == T.PENDING:
        if time.monotonic() > deadline:
            raise TimeoutError('HIP copied-input fixture timed out')
        time.sleep(.001)
    assert call(api.poll, ticket) == T.COMPLETE
    actual = (np.full((16, P), np.nan, np.float32),
              np.full((4, P), np.nan, np.float32), np.full(P, np.nan, np.float32))
    output = T.Output(C.sizeof(T.Output), 16 * P, 4 * P, P, *(T.pointer(a) for a in actual))
    assert call(api.collect, ticket, C.byref(output)) == T.OK
    expected = reference(args, folder, 'copied-inputs', *snapshots, 1, 0, 0)
    result = compare('copied-inputs', actual, expected)
    context.reset()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--device-pci', required=True)
    parser.add_argument('--vulkan-device-uuid', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.plugin, args.reference, args.out = (p.resolve() for p in (args.plugin, args.reference, args.out))
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        with (ROOT / '.claude/reviews/tools/spatial-learning/gpu-owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            plugin = T.Plugin(args.plugin)
            devices = [d for d in plugin.devices() if d.pci_text == args.device_pci]
            if len(devices) != 1:
                raise RuntimeError('PCI must match exactly one plugin device')
            getter = plugin.module.gsc_training_get_api
            assert not getter(T.ABI + 1, C.sizeof(T.Api)) and not getter(T.ABI, C.sizeof(T.Api) - 1)
            invalid = T.Device.from_buffer_copy(devices[0])
            invalid.pci = b'ffff:ff:ff.f'
            info = T.Create(C.sizeof(T.Create), T.ABI, invalid)
            empty = C.c_void_p()
            assert plugin.api.create(C.byref(info), C.byref(empty), plugin.error, len(plugin.error)) == T.INVALID
            assert not empty.value
            with plugin.open(devices[0]) as context:
                results.append(check_ownership(context, sample(), prior(), args, args.out))
                for zero, flags, tag in ((True, 0, 'zero-head'), (False, 0, 'causal'), (False, 4, 'bidir')):
                    context.reset()
                    weights = prior(zero)
                    hip_state, vk_state = initial(weights), initial(weights)
                    for step in range(1, 4):
                        data = sample(17, 19) if step != 2 else sample(16, 16)
                        name = f'{tag}-{step}'
                        actual = context.run(data, weights, hip_state, step, flags=flags)
                        expected = reference(args, args.out, name, data, weights, vk_state, step, flags, 0)
                        result = compare(name, actual, expected)
                        if zero and step == 1:
                            assert not np.any(actual[0][:, :4064])
                            assert np.any(actual[0][:, 4064:])
                        if flags == 4:
                            assert not np.any(actual[0][:, :4064])
                            assert not np.any(actual[0][:, 4064:4352])
                            assert not np.any(actual[0][:, 4496:4642])
                            assert not np.any(actual[0][:, 4643])
                        hip_state, vk_state = actual[1], expected[1]
                        result['changed_served'] = int(np.count_nonzero(actual[2] != weights))
                        results.append(result)
                        print(json.dumps(result), flush=True)
                    # A cut must zero gradients while still advancing optimizer
                    # momentum/decay. Explicit existing-momentum bidir tests the
                    # distinction between gradient freezing and state masking.
                    for next_flags, cut, suffix in ((1 | flags, 1, 'cut'), (4, 0, 'bidir-momentum')):
                        name = f'{tag}-{suffix}'
                        data = sample()
                        actual = context.run(data, weights, hip_state, 4, flags=next_flags, scene_cut=cut)
                        expected = reference(args, args.out, name, data, weights, vk_state, 4, next_flags, cut)
                        result = compare(name, actual, expected)
                        if cut:
                            assert not np.any(actual[0])
                        results.append(result)
                        print(json.dumps(result), flush=True)
                # Independent optimizer corner: a constant contrast and large
                # hidden activations produce gradient components >1. The online
                # optimizer must component-clip them, and bound fast weights.
                context.reset()
                weights = np.zeros(P, np.float32)
                weights[1728:1744] = weights[4048:4064] = 64
                # Negative confidence correction selects its ungated suppression
                # branch. At exact zero the causal evidence gate rejects this
                # deliberately high-contrast fixture, giving no gradient.
                weights[4642] = -.01
                state = initial(weights)
                state[0, 4643] = 65
                luma = np.full((17, 19), .1, np.float32)
                field = np.zeros((17, 19, 4), np.float32)
                field[..., 2] = .5
                data = (luma, np.full_like(luma, .8), field, field.copy())
                actual = context.run(data, weights, state, 1)
                expected = reference(args, args.out, 'optimizer-bounds', data, weights, state, 1, 0, 0)
                result = compare('optimizer-bounds', actual, expected)
                assert np.all(actual[0][:, 4352] > 1)
                assert abs(float(actual[1][1, 4352]) - .1) < 1e-7
                assert abs(float(actual[1][2, 4352]) - .001) < 1e-8
                assert actual[1][0, 4643] == 64 and actual[2][4643] == 4
                results.append(result)
                print(json.dumps(result), flush=True)
                device = {'name': devices[0].name_text, 'pci': devices[0].pci_text, 'uuid': devices[0].uuid_hex}
    except BaseException as error:
        (args.out / 'failure.txt').write_text(f'{type(error).__name__}: {error}\n')
        raise
    paths = [args.plugin, args.reference, *Path(__file__).parent.glob('*.py'),
             *Path(__file__).parent.glob('*.hip'), *Path(__file__).parent.glob('*.cpp'),
             *Path(__file__).parent.glob('*.h'), *Path(__file__).parent.glob('*.hpp'),
             ROOT / 'src/shaders/cs_framegen_motion_net_train.comp',
             ROOT / 'src/shaders/cs_framegen_motion_net_opt.comp', ROOT / 'src/shaders/framegen_evidence.h',
             ROOT / 'src/framegen/net_layout.hpp', ROOT / 'src/framegen/compute_job.hpp',
             args.reference.parent / 'training-reference-grad.spv',
             args.reference.parent / 'training-reference-opt.spv',
             *args.out.glob('*.input'), *args.out.glob('*.output')]
    receipt = {'scope': __doc__, 'device': device,
               'reference': {'path': str(args.reference), 'vulkan_device_uuid': args.vulkan_device_uuid,
                             'validation': True, 'transport': 'original GLSL arithmetic with SSBO views'},
               'results': results,
               'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.out / 'parity.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('training arithmetic/ownership parity PASS; production image parity remains unverified')


if __name__ == '__main__':
    def terminate(signum, frame):
        raise InterruptedError(f'training check interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    main()
