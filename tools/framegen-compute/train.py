#!/usr/bin/env python3
"""Offline consumer of an explicitly loaded native framegen training plugin.

No accelerator SDK is imported here. GSFR exports contain served EMA weights,
not Adam moments: loading one starts a new optimizer warm start.
"""

import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time

import numpy as np


ABI, MODEL_VERSION, P, SLICES = 1, 3, 4644, 16
OK, PENDING, COMPLETE, DISCARDED = 0, 1, 2, 3
INVALID, UNAVAILABLE, BUSY, STALE, FAILED = -1, -2, -3, -4, -5
LAYERS = ((12, 16), (16, 16), (16, 4))
U32, U64, F32P = C.c_uint32, C.c_uint64, C.POINTER(C.c_float)
ERRORP = C.POINTER(C.c_char)
_retained_contexts = []  # Failed destroy must retain both code and GPU owner.


class BackendContract(C.Structure):
    _fields_ = [(name, U32) for name in ("size", "abi", "operation", "execution", "payload",
                "model_version", "semantics", "state", "transport", "features", "max_inflight", "reserved")]


class Device(C.Structure):
    _fields_ = [("size", U32), ("name", C.c_char * 128),
                ("pci", C.c_char * 32), ("uuid", C.c_uint8 * 16)]

    @property
    def name_text(self):
        return self.name.decode("utf-8", errors="replace")

    @property
    def pci_text(self):
        return self.pci.decode("ascii", errors="strict")

    @property
    def uuid_hex(self):
        return bytes(self.uuid).hex()


class Create(C.Structure):
    _fields_ = [("size", U32), ("abi", U32), ("device", Device)]


class Ticket(C.Structure):
    _fields_ = [("epoch", U64), ("serial", U64)]


class Job(C.Structure):
    _fields_ = [("size", U32), ("model_version", U32), ("epoch", U64),
                ("width", U32), ("height", U32), ("step", U32),
                ("flags", U32), ("scene_cut", U32),
                ("learning_rate", C.c_float), ("ema_alpha", C.c_float),
                ("prior_decay", C.c_float), ("luma_count", U64),
                ("field_count", U64), ("prior_count", U64), ("state_count", U64),
                ("source", F32P), ("destination", F32P), ("forward", F32P),
                ("reverse", F32P), ("prior", F32P), ("state", F32P)]


class Output(C.Structure):
    _fields_ = [("size", U32), ("gradient_count", U64), ("state_count", U64),
                ("served_count", U64), ("gradients", F32P), ("state", F32P),
                ("served", F32P)]


_Devices = C.CFUNCTYPE(C.c_int, C.POINTER(Device), U32, C.POINTER(U32), ERRORP, U32)
_Create = C.CFUNCTYPE(C.c_int, C.POINTER(Create), C.POINTER(C.c_void_p), ERRORP, U32)
_Submit = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(Job), C.POINTER(Ticket), ERRORP, U32)
_TicketOp = C.CFUNCTYPE(C.c_int, C.c_void_p, Ticket, ERRORP, U32)
_Collect = C.CFUNCTYPE(C.c_int, C.c_void_p, Ticket, C.POINTER(Output), ERRORP, U32)
_Reset = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(U64), ERRORP, U32)
_Destroy = C.CFUNCTYPE(C.c_int, C.c_void_p, ERRORP, U32)


class Api(C.Structure):
    _fields_ = [("size", U32), ("abi", U32), ("model_version", U32),
                ("parameters", U32), ("slices", U32), ("name", C.c_char_p),
                ("devices", _Devices), ("create", _Create), ("submit", _Submit),
                ("poll", _TicketOp), ("collect", _Collect), ("discard", _TicketOp),
                ("retire", _TicketOp), ("reset", _Reset), ("destroy", _Destroy)]


def uint32(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= 0xffffffff:
        raise ValueError(f"{name} must be in [{minimum}, UINT32_MAX]")
    return int(value)


def float_array(value, shape, name):
    """Convert real numeric input to contiguous finite native float32."""
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != tuple(shape):
        raise ValueError(f"{name} must be a real array with shape {tuple(shape)}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(raw, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite float32 values")
    return result


def prepare_sample(sample):
    if len(sample) != 4:
        raise ValueError("sample must contain source, destination, forward and reverse")
    source = np.asarray(sample[0])
    if source.ndim != 2:
        raise ValueError("source luma must have two dimensions")
    h, w = source.shape
    if w < 16 or h < 16 or w * h > (1 << 20):
        raise ValueError("sample dimensions must be at least 16x16 and at most 1048576 pixels")
    shapes = ((h, w), (h, w), (h, w, 4), (h, w, 4))
    names = ("source", "destination", "forward", "reverse")
    return tuple(float_array(a, shape, name) for a, shape, name in zip(sample, shapes, names))


def optimizer_controls(step, flags, scene_cut, learning_rate, ema_alpha, prior_decay):
    step = uint32(step, "step", 1)
    flags = uint32(flags, "flags")
    scene_cut = uint32(scene_cut, "scene_cut")
    if flags & ~5 or scene_cut > 1:
        raise ValueError("only cut guard (mask 1), conservative bidir (mask 4), and scene_cut 0/1 are supported")
    controls = []
    for name, value, upper in (("learning_rate", learning_rate, None),
                                ("ema_alpha", ema_alpha, 1), ("prior_decay", prior_decay, 1)):
        value = float(value)
        stored = C.c_float(value).value
        if not math.isfinite(stored) or stored < 0 or (upper is not None and stored > upper):
            raise ValueError(f"invalid {name}")
        controls.append(stored)
    return step, flags, scene_cut, *controls


def pointer(array):
    return array.ctypes.data_as(F32P)


class Plugin:
    """The public .api/.module/.error/.check also support ABI negative checks."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file():
            raise ValueError("plugin must be an explicit local file")
        self.module = C.CDLL(str(self.path), mode=os.RTLD_LOCAL)
        self.error = C.create_string_buffer(1024)
        getter = self.module.gsc_training_get_api
        getter.argtypes = [U32, U32]
        getter.restype = C.POINTER(Api)
        table = getter(ABI, C.sizeof(Api))
        if not table:
            raise RuntimeError("plugin rejected the exact training ABI/table size")
        self.api = table.contents
        expected = (C.sizeof(Api), ABI, MODEL_VERSION, P, SLICES)
        actual = (self.api.size, self.api.abi, self.api.model_version,
                  self.api.parameters, self.api.slices)
        if actual != expected or not self.api.name:
            raise RuntimeError(f"incompatible plugin ABI/model/layout: {actual}")
        for name in ("devices", "create", "submit", "poll", "collect", "discard", "retire", "reset", "destroy"):
            if not getattr(self.api, name):
                raise RuntimeError(f"null plugin callback: {name}")
        self.contract_api = _leaf("framegen_backend")
        expected_contract = self.contract_api.PROFILES['snapshot_training']
        descriptor = getattr(self.module, 'gsc_compute_get_contract', None)
        if descriptor is None:
            # Compatibility is inferred only from the validated, frozen v1 ABI,
            # never from a provider name or a speculative resident capability.
            self.contract = expected_contract
            self.contract_source = 'validated-training-v1'
        else:
            descriptor.argtypes = [U32, U32, U32]
            descriptor.restype = C.POINTER(BackendContract)
            result = descriptor(expected_contract.abi, C.sizeof(BackendContract), expected_contract.payload)
            if not result:
                raise RuntimeError('plugin rejected the shared typed backend contract')
            self.contract = self.contract_api.Contract(*(getattr(result.contents, name)
                for name, _ in BackendContract._fields_))
            self.contract_source = 'native-descriptor'
        self.contract_api.require_contract(self.contract, self.contract_api.requirements_for(expected_contract))

    def check(self, result, operation="plugin operation", expected=OK):
        if result != expected:
            message = self.error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"{operation}: result {result}: {message}")
        return result

    def devices(self):
        count = U32()
        self.check(self.api.devices(None, 0, C.byref(count), self.error, len(self.error)), "device count")
        if count.value == 0:
            return []
        records = (Device * count.value)()
        for record in records:
            record.size = C.sizeof(Device)
        capacity = count.value
        self.check(self.api.devices(records, capacity, C.byref(count), self.error, len(self.error)), "device inventory")
        if count.value != capacity:
            raise RuntimeError("device inventory changed while enumerating")
        result = [Device.from_buffer_copy(record) for record in records]
        for device in result:
            if device.size != C.sizeof(Device) or not device.pci or not any(device.uuid):
                raise RuntimeError("device record has incompatible size or missing PCI/UUID identity")
            # Missing NUL termination cannot be forwarded as a C string safely.
            if len(device.pci) >= 32 or len(device.name) >= 128:
                raise RuntimeError("device record strings are not terminated")
        return result

    def open(self, device):
        return Context(self, device)


class Context:
    """Single owner; direct tests may use .api, .handle, .error and .check.

    run() returns gradients (16,P), candidate state (4,P), and served EMA (P).
    Timeout marks usefulness discarded; it does not cancel or retire GPU work.
    close() may wait for actual completion, including after timeout.
    """

    def __init__(self, plugin, device):
        if not isinstance(device, Device) or device.size != C.sizeof(Device):
            raise ValueError("open requires an enumerated Device record")
        self.plugin, self.api = plugin, plugin.api
        self.error = C.create_string_buffer(1024)
        self.handle = C.c_void_p()
        self.epoch = 1
        self._active = None
        self._retained = False
        # Cache the original native callbacks once. No forwarding function,
        # tensor conversion or changed polling policy is introduced by binding.
        self.binding = plugin.contract_api.bind_backend(plugin, plugin.contract,
            plugin.contract_api.requirements_for(plugin.contract_api.PROFILES['snapshot_training']),
            submit=self.api.submit, poll=self.api.poll, collect=self.api.collect,
            discard=self.api.discard, retire=self.api.retire, reset=self.api.reset, close=self.api.destroy)
        info = Create(C.sizeof(Create), ABI, device)
        self.check(self.api.create(C.byref(info), C.byref(self.handle), self.error, len(self.error)), "create")
        if not self.handle.value:
            raise RuntimeError("plugin returned an empty context")

    def check(self, result, operation="context operation", expected=OK):
        if result != expected:
            message = self.error.value.decode("utf-8", errors="replace")
            raise RuntimeError(f"{operation}: result {result}: {message}")
        return result

    def _require_open(self):
        if not self.handle.value:
            raise RuntimeError("training context is closed")
        if self._retained:
            raise RuntimeError("training context retirement is unresolved; only close may be retried")

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if not self.handle.value:
            return
        result = self.api.destroy(self.handle, self.error, len(self.error))
        if result != OK:
            if not self._retained:
                _retained_contexts.append(self)
                self._retained = True
            self.check(result, "destroy; context and module retained")
        self.handle = C.c_void_p()
        self._active = None
        if self._retained:
            _retained_contexts.remove(self)
            self._retained = False

    def reset(self):
        self._require_open()
        epoch = U64()
        self.check(self.api.reset(self.handle, C.byref(epoch), self.error, len(self.error)), "reset")
        if epoch.value <= self.epoch:
            raise RuntimeError("reset did not advance the epoch")
        self.epoch = epoch.value
        self._active = None
        return self.epoch

    def run(self, sample, prior, state, step, flags=0, scene_cut=0,
            learning_rate=3e-4, ema_alpha=0.0625, prior_decay=0.0005, timeout=30):
        self._require_open()
        if self._active is not None:
            raise RuntimeError("context still owns an outstanding training ticket")
        controls = optimizer_controls(step, flags, scene_cut, learning_rate, ema_alpha, prior_decay)
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        source, destination, forward, reverse = prepare_sample(sample)
        prior = float_array(prior, (P,), "prior")
        state = np.asarray(state)
        if state.shape == (4 * P,):
            state = state.reshape(4, P)
        state = float_array(state, (4, P), "state")
        h, w = source.shape
        job = Job(C.sizeof(Job), MODEL_VERSION, self.epoch, w, h, *controls,
                  source.size, forward.size, P, 4 * P,
                  pointer(source), pointer(destination), pointer(forward), pointer(reverse),
                  pointer(prior), pointer(state))
        ticket = Ticket()
        self.check(self.binding.submit(self.handle, C.byref(job), C.byref(ticket), self.error, len(self.error)), "submit")
        if ticket.epoch != self.epoch or ticket.serial == 0:
            raise RuntimeError("submit returned an invalid ticket")
        self._active = ticket
        deadline = time.monotonic() + timeout
        while True:
            result = self.binding.poll(self.handle, ticket, self.error, len(self.error))
            if result == COMPLETE:
                break
            if result != PENDING:
                self.check(result, "poll", COMPLETE)
            if time.monotonic() >= deadline:
                self.check(self.binding.discard(self.handle, ticket, self.error, len(self.error)), "discard expired job")
                raise TimeoutError("training observation timed out; work remains owned until completion/teardown")
            time.sleep(0.001)
        gradients = np.full((SLICES, P), np.nan, np.float32)
        candidate = np.full((4, P), np.nan, np.float32)
        served = np.full(P, np.nan, np.float32)
        output = Output(C.sizeof(Output), gradients.size, candidate.size, served.size,
                        pointer(gradients), pointer(candidate), pointer(served))
        self.check(self.binding.collect(self.handle, ticket, C.byref(output), self.error, len(self.error)), "collect")
        self._active = None
        if (output.size, output.gradient_count, output.state_count, output.served_count) != (C.sizeof(Output), SLICES * P, 4 * P, P):
            raise RuntimeError("plugin changed output layout/counts")
        if not all(np.isfinite(a).all() for a in (gradients, candidate, served)):
            raise RuntimeError("non-finite training result; candidate must not be committed")
        return gradients, candidate, served


def _leaf(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_compute_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load repository reader {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _SampleSnapshot:
    """Let the existing reader decode exactly the bytes admitted and hashed."""

    def __init__(self, path, raw):
        self.path, self.raw = path, raw

    def read_bytes(self):
        return self.raw

    def __str__(self):
        return str(self.path)


def load_sample_strict(path, expected_sha256=None):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 40:
        raise ValueError(f"{path}: truncated GSFD header")
    magic, version, w, h, luma_bpp, field_bpp, flags, reserved = struct.unpack_from("<8I", raw)
    if magic != 0x44465347 or version != 1:
        raise ValueError(f"{path}: expected GSFD v1")
    if field_bpp != 8 or luma_bpp not in (2, 8) or not flags & 1:
        raise ValueError(f"{path}: unsupported GSFD layout")
    if w < 16 or h < 16 or w * h > (1 << 20):
        raise ValueError(f"{path}: unsupported sample dimensions")
    expected = 40 + w * h * (2 * luma_bpp + 2 * field_bpp)
    if len(raw) != expected:
        raise ValueError(f"{path}: GSFD payload length {len(raw)} != {expected}")
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{path}: sample changed after receipt hashing")
    if not np.isfinite(np.frombuffer(raw, dtype="<f2", offset=40)).all():
        raise ValueError(f"{path}: non-finite GSFD payload")
    # The existing reader owns half-float decoding and RGBA-luma extraction.
    return prepare_sample(_leaf("framegen_field_capture").load_sample(_SampleSnapshot(path, raw)))


def flatten_params(params):
    chunks = []
    for i, (ci, co) in enumerate(LAYERS, 1):
        w = float_array(params[f"W{i}"], (co, ci, 3, 3), f"W{i}")
        b = float_array(params[f"b{i}"], (co,), f"b{i}")
        chunks.extend((w.transpose(0, 2, 3, 1).reshape(-1), b))
    return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def unflatten_params(flat):
    flat = float_array(flat, (P,), "parameters")
    params, off = {}, 0
    for i, (ci, co) in enumerate(LAYERS, 1):
        count = co * 3 * 3 * ci
        params[f"W{i}"] = np.ascontiguousarray(flat[off:off + count].reshape(co, 3, 3, ci).transpose(0, 3, 1, 2))
        off += count
        params[f"b{i}"] = flat[off:off + co].copy()
        off += co
    return params


def load_net_strict(path, expected_sha256=None):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) != 48 + 4 * P or struct.unpack_from("<3I", raw) != (0x52465347, MODEL_VERSION, 3):
        raise ValueError(f"{path}: expected exact bare GSFR v3 (no profile sets)")
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"{path}: prior changed after receipt hashing")
    flat = flatten_params(_leaf("framegen_net_formats").load_blob(path))
    if flat.astype("<f4").tobytes() != raw[48:]:
        raise ValueError(f"{path}: prior changed while decoding")
    return flat


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_paths(path, inputs):
    output = Path(path).expanduser().absolute()
    receipt = Path(str(output) + ".json")
    if not output.parent.is_dir():
        raise ValueError("output parent directory must already exist")
    if any(output.resolve() == Path(p).resolve() for p in inputs):
        raise ValueError("output must not be an input/source path")
    if os.path.lexists(output) or os.path.lexists(receipt):
        raise ValueError("output and receipt must both be new paths")
    return output, receipt


def publish(served, output, receipt_path, receipt):
    """Publish each complete artifact with a non-overwriting same-dir hard link."""
    temporary, linked = [], []
    try:
        for suffix in (".gsfr.tmp", ".json.tmp"):
            fd, path = tempfile.mkstemp(prefix=".framegen-train-", suffix=suffix, dir=output.parent)
            os.close(fd)
            temporary.append(Path(path))
        _leaf("framegen_net_formats").export(unflatten_params(served), temporary[0])
        receipt["output_sha256"] = sha256(temporary[0])
        temporary[1].write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        for source, destination in zip(temporary, (output, receipt_path)):
            os.link(source, destination)
            linked.append(destination)
    except BaseException:
        for path in reversed(linked):
            path.unlink()
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device-pci")
    parser.add_argument("--net", type=Path)
    parser.add_argument("--samples", nargs="+", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--mode", choices=("causal", "bidir"), default="causal")
    parser.add_argument("--input-key", type=lambda s: int(s, 0))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ema-alpha", type=float, default=0.0625)
    parser.add_argument("--prior-decay", type=float, default=0.0005)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)
    plugin = Plugin(args.plugin)
    devices = plugin.devices()
    if args.list_devices:
        for device in devices:
            print(f"pci={device.pci_text} uuid={device.uuid_hex} name={device.name_text}")
        return 0
    if any(value is None for value in (args.device_pci, args.net, args.samples, args.input_key, args.output)):
        parser.error("training requires --device-pci, --net, --samples, --input-key and --output")
    key = uint32(args.input_key, "input-key")
    steps = len(args.samples) if args.steps is None else args.steps
    if not 1 <= steps <= len(args.samples):
        parser.error("--steps must be positive and no larger than the sample file count; samples never cycle")
    matches = [device for device in devices if device.pci_text == args.device_pci]
    if len(matches) != 1:
        parser.error("--device-pci must match exactly one enumerated record")
    selected_samples = args.samples[:steps]
    output, receipt_path = output_paths(args.output, [args.plugin, args.net, *args.samples])
    prior_sha256 = sha256(args.net)
    prior = load_net_strict(args.net, expected_sha256=prior_sha256)
    state = np.zeros((4, P), np.float32)
    state[0], state[3] = prior, prior
    served = prior.copy()
    flags = 4 if args.mode == "bidir" else 0
    optimizer_controls(1, flags, 0, args.learning_rate, args.ema_alpha, args.prior_decay)
    receipt = {
        "scope": "offline recorded tensors; not live compositor training; no quality or performance claim",
        "model_version": MODEL_VERSION, "abi": ABI,
        "backend_contract": plugin.contract.as_dict(), "backend_contract_source": plugin.contract_source,
        "plugin": str(plugin.path), "plugin_sha256": sha256(plugin.path),
        "plugin_name": plugin.api.name.decode("utf-8", errors="replace"),
        "prior": str(args.net.resolve()), "prior_sha256": prior_sha256,
        "samples": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in selected_samples],
        "declared_input_key": key, "input_key_provenance": "user declaration; GSFD v1 does not encode this key",
        "mode": args.mode, "flags": flags, "steps": steps,
        "device": {"pci": matches[0].pci_text, "uuid": matches[0].uuid_hex, "name": matches[0].name_text},
        "optimizer": {"learning_rate": args.learning_rate, "ema_alpha": args.ema_alpha, "prior_decay": args.prior_decay},
        "warm_start": "fast=EMA=prior; m=v=0; GSFR contains no optimizer moments",
    }
    with plugin.open(matches[0]) as context:
        for step, (path, sample_receipt) in enumerate(zip(selected_samples, receipt["samples"]), 1):
            sample = load_sample_strict(path, expected_sha256=sample_receipt["sha256"])
            gradients, candidate, updated = context.run(sample, prior, state, step,
                flags=flags, learning_rate=args.learning_rate, ema_alpha=args.ema_alpha,
                prior_decay=args.prior_decay, timeout=args.timeout)
            # Commit only the complete, finite candidate; the prior stays fixed.
            state, served = candidate, updated
            print(f"accepted_step={step} gradient_nonzero={np.count_nonzero(gradients)} "
                  f"served_changed={np.count_nonzero(served != prior)}")
    delta = served.astype(np.float64) - prior.astype(np.float64)
    receipt["served_changed_count"] = int(np.count_nonzero(served != prior))
    receipt["served_max_abs_delta"] = float(np.max(np.abs(delta)))
    receipt["served_l2_delta"] = float(np.linalg.norm(delta))
    publish(served, output, receipt_path, receipt)
    print(f"trained served weights: accepted_steps={steps} changed={receipt['served_changed_count']}/{P} "
          f"max_abs_delta={receipt['served_max_abs_delta']:.9g} output={output}")
    print("offline warm start only; no live compositor integration or quality claim")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        raise SystemExit(f"framegen training: {error}")
