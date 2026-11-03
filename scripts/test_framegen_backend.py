#!/usr/bin/env python3
"""SDK-free wire/cold-negotiation/direct-binding contracts; no accelerator calls."""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import unittest
import weakref

import framegen_backend as B


CONTRACT_BINARY = None
SCRIPTS = Path(__file__).resolve().parent
# Independent public wire values, not generated from the declaration parser.
EXPECTED = {
    "generation": [48, 1, 1, 1, 1, 0, 1, 0, 1, 24, 1, 0],
    "snapshot_training": [48, 1, 2, 1, 2, 3, 2, 1, 1, 28, 1, 0],
    "resident_training": [48, 1, 2, 1, 3, 3, 2, 2, 3, 23, 1, 0],
    "npu_raw_head": [48, 1, 3, 2, 4, 3, 3, 0, 1, 32, 1, 0],
}
REQUIRED = {
    "generation": ("submit", "poll", "collect", "discard", "retire"),
    "snapshot_training": ("submit", "poll", "collect", "discard", "retire"),
    "resident_training": ("submit", "poll", "collect", "checkpoint"),
    "npu_raw_head": ("execute", "evidence"),
}


class Module:
    pass


class Owner:
    def __init__(self):
        self.module = Module()
        self.calls = []

    def called(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        return args[0] if args else self.module

    def execute(self, *args, **kwargs): return self.called("execute", *args, **kwargs)
    def submit(self, *args, **kwargs): return self.called("submit", *args, **kwargs)
    def poll(self, *args, **kwargs): return self.called("poll", *args, **kwargs)
    def collect(self, *args, **kwargs): return self.called("collect", *args, **kwargs)
    def checkpoint(self, *args, **kwargs): return self.called("checkpoint", *args, **kwargs)
    def discard(self, *args, **kwargs): return self.called("discard", *args, **kwargs)
    def retire(self, *args, **kwargs): return self.called("retire", *args, **kwargs)
    def reset(self, *args, **kwargs): return self.called("reset", *args, **kwargs)
    def close(self, *args, **kwargs): return self.called("close", *args, **kwargs)
    def evidence(self, *args, **kwargs): return self.called("evidence", *args, **kwargs)


def callbacks(owner, profile):
    return {name: getattr(owner, name) for name in REQUIRED[profile]}


class BackendContract(unittest.TestCase):
    def test_independent_public_wire_values_and_order(self):
        self.assertEqual(B.Contract._fields, (
            "size", "abi", "operation", "execution", "payload", "model_version",
            "semantics", "state", "transport", "features", "max_inflight", "reserved"))
        self.assertEqual(set(B.PROFILES), set(EXPECTED))
        for name, expected in EXPECTED.items():
            contract = B.PROFILES[name]
            with self.subTest(profile=name):
                self.assertIsInstance(contract, B.Contract)
                self.assertEqual(list(contract), expected)
                self.assertEqual(list(contract.as_dict().values()), expected)
                required = B.requirements_for(contract)
                self.assertEqual(required.features, contract.features)
                self.assertIsNone(B.require_contract(contract, required))
                self.assertIsNone(B.require_contract(contract, B.requirements_for(contract, features=0)))

    def test_compiled_c_profile_words_match_python(self):
        if CONTRACT_BINARY is None:
            self.skipTest("pass --contract-binary for the compiled C/C++ wire cross-check")
        result = subprocess.run([str(CONTRACT_BINARY), "--dump"], capture_output=True,
                                text=True, timeout=10, check=True)
        self.assertEqual(json.loads(result.stdout), EXPECTED)
        self.assertEqual(json.loads(result.stdout), {name: list(value) for name, value in B.PROFILES.items()})

    def test_malformed_wire_and_schema_combinations_reject(self):
        for name, contract in B.PROFILES.items():
            required = B.requirements_for(contract)
            for field in B.Contract._fields:
                with self.subTest(profile=name, field=field), self.assertRaises(ValueError):
                    B.require_contract(contract._replace(**{field: 47 if field == "size" else 0xffffffff}), required)
            for bad in (True, -1, 2**32, 1.5, "1", None):
                with self.subTest(profile=name, bad=bad), self.assertRaises(ValueError):
                    B.require_contract(contract._replace(abi=bad), required)
            with self.assertRaises(ValueError):
                B.require_contract(tuple(contract), required)
        # Each field is a known value, but it belongs to another payload.
        resident = B.PROFILES["resident_training"]
        snapshot = B.PROFILES["snapshot_training"]
        with self.assertRaises(ValueError):
            B.require_contract(resident._replace(state=snapshot.state), B.requirements_for(resident))
        npu = B.PROFILES["npu_raw_head"]
        with self.assertRaises(ValueError):
            B.require_contract(npu._replace(operation=resident.operation, execution=resident.execution),
                               B.requirements_for(resident))

    def test_requirements_are_typed_and_missing_features_never_fallback(self):
        for name, contract in B.PROFILES.items():
            required = B.requirements_for(contract)
            for field in B.Requirements._fields:
                with self.subTest(profile=name, field=field), self.assertRaises(ValueError):
                    B.require_contract(contract, required._replace(**{field: 0xffffffff}))
            with self.assertRaises(ValueError):
                B.require_contract(contract, tuple(required))
            with self.assertRaises(ValueError):
                B.require_contract(contract, required._replace(features=True))
        for actual in B.PROFILES.values():
            for wanted in B.PROFILES.values():
                if actual.payload != wanted.payload:
                    with self.assertRaises(ValueError):
                        B.require_contract(actual, B.requirements_for(wanted))

    def test_direct_callback_identity_and_payload_objects(self):
        for profile, contract in B.PROFILES.items():
            owner = Owner()
            direct = callbacks(owner, profile)
            binding = B.bind_backend(owner, contract, B.requirements_for(contract), **direct)
            self.assertIs(binding.owner, owner)
            self.assertIs(binding.contract, contract)
            for name, original in direct.items():
                alias = getattr(binding, name)
                with self.subTest(profile=profile, callback=name):
                    self.assertIs(alias, original)  # No wrapper/dispatcher was inserted.
                    self.assertIs(alias.__self__, owner)
                    self.assertIs(alias.__func__, getattr(Owner, name))
                    payload, keyword = object(), object()
                    self.assertIs(alias(payload, marker=keyword), payload)
                    self.assertEqual(len(owner.calls), 1)
                    op, args, kwargs = owner.calls.pop()
                    self.assertEqual(op, name)
                    self.assertIs(args[0], payload)
                    self.assertIs(kwargs["marker"], keyword)
            with self.assertRaises(AttributeError):
                binding.unadvertised_operation = lambda: None

    def test_owner_and_native_module_lifetime_survive_caller_release(self):
        owner = Owner()
        owner_ref, module_ref = weakref.ref(owner), weakref.ref(owner.module)
        contract = B.PROFILES["resident_training"]
        binding = B.bind_backend(owner, contract, B.requirements_for(contract), **callbacks(owner, "resident_training"))
        del owner
        gc.collect()
        self.assertIsNotNone(owner_ref())
        self.assertIsNotNone(module_ref())
        token = object()
        self.assertIs(binding.submit(token), token)
        del binding
        gc.collect()
        self.assertIsNone(owner_ref())
        self.assertIsNone(module_ref())

    def test_required_callbacks_and_explicit_optional_callbacks(self):
        for profile, contract in B.PROFILES.items():
            owner = Owner()
            direct = callbacks(owner, profile)
            for missing in REQUIRED[profile]:
                partial = dict(direct); partial.pop(missing)
                with self.subTest(profile=profile, missing=missing), self.assertRaises(ValueError):
                    B.bind_backend(owner, contract, B.requirements_for(contract), **partial)
            for extra in ({"not_an_operation": owner.close}, {"close": 123}):
                with self.assertRaises(ValueError):
                    B.bind_backend(owner, contract, B.requirements_for(contract), **direct, **extra)
            binding = B.bind_backend(owner, contract, B.requirements_for(contract),
                                     **direct, close=owner.close, reset=owner.reset)
            self.assertIs(binding.close.__self__, owner)
            self.assertIs(binding.reset.__self__, owner)

    def test_no_blocking_async_adaptation_or_unadvertised_operations(self):
        for profile, extra_names in (("npu_raw_head", ("submit", "poll", "collect", "discard", "retire", "checkpoint")),
                                     ("resident_training", ("execute", "discard", "retire", "evidence")),
                                     ("snapshot_training", ("execute", "checkpoint", "evidence"))):
            owner = Owner(); contract = B.PROFILES[profile]
            direct = callbacks(owner, profile)
            for extra in extra_names:
                with self.subTest(profile=profile, extra=extra), self.assertRaises(ValueError):
                    B.bind_backend(owner, contract, B.requirements_for(contract), **direct, **{extra: getattr(owner, extra)})
            self.assertEqual(owner.calls, [])  # Cold rejection never invokes backend code.

    def test_fresh_import_cannot_load_sdk_or_native_library(self):
        program = r'''
import builtins
import ctypes
from pathlib import Path
import sys

forbidden = {'onnxruntime', 'openvino', 'hip', 'hiprt', 'cuda', 'cupy', 'torch', 'pyopencl'}
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in forbidden:
        raise AssertionError('optional SDK import: ' + name)
    return original(name, *args, **kwargs)
def no_native(*args, **kwargs):
    raise AssertionError('native library loaded by common contract')
builtins.__import__ = guarded
ctypes.CDLL = no_native
sys.path.insert(0, sys.argv[1])
import framegen_backend
assert Path(framegen_backend.__file__).resolve() == Path(sys.argv[1]) / 'framegen_backend.py'
assert set(framegen_backend.PROFILES) == {'generation', 'snapshot_training', 'resident_training', 'npu_raw_head'}
assert not forbidden.intersection(sys.modules)
'''
        result = subprocess.run([sys.executable, "-I", "-c", program, str(SCRIPTS)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--contract-binary", type=Path)
    args, remaining = parser.parse_known_args()
    CONTRACT_BINARY = args.contract_binary.resolve() if args.contract_binary is not None else None
    unittest.main(argv=[sys.argv[0], *remaining])
