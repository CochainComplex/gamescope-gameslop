#!/usr/bin/env python3
"""Numerical transport contracts; corpus parity remains a separate gate."""
import contextlib
import dataclasses
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    print("numpy unavailable; offline field model not exercised")
    raise SystemExit(77)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G = load("field_model_contract", "framegen-field-gate.py")


class FieldModelContract(unittest.TestCase):
    def test_rtz_bits_and_exceptional_values(self):
        values = [0., -0., 1., -1., 1.000732421875, -1.000732421875,
                  2**-24, -2**-24, 2**-25, -2**-25, 3*2**-25, -3*2**-25,
                  65520., -65520., 70000., -70000.]
        expected = [0, 0x8000, 0x3c00, 0xbc00, 0x3c00, 0xbc00,
                    1, 0x8001, 0, 0x8000, 1, 0x8001,
                    0x7bff, 0xfbff, 0x7bff, 0xfbff]
        stored = G.half_store(np.array(values, np.float32))
        self.assertEqual(stored.dtype, np.float32)
        self.assertEqual(stored.astype(np.float16).view(np.uint16).tolist(), expected)
        unusual = G.half_store(np.array([np.nan, np.inf, -np.inf], np.float32))
        self.assertTrue(np.isnan(unusual[0]))
        self.assertEqual(unusual[1:].tolist(), [np.inf, -np.inf])

    def test_odd_grid_ramp(self):
        y, x = np.mgrid[:3, :5].astype(np.float32)
        result = G.half(x / 8 + y / 32, numerics=G.Numerics(G.REFERENCE_MODEL))
        expected = np.array([[101, 528, 955], [197, 624, 1051]], np.float32) / 2048
        np.testing.assert_array_equal(result, expected)

    def test_signed_half_boundary(self):
        for cell_bits, expected in [([0x3800, 0x3801, 0x3800, 0x3801], 0x3800),
                                    ([0x3801, 0x3802, 0x3801, 0x3802], 0x3801)]:
            for sign_bit in (0, 0x8000):
                cell = (np.array(cell_bits, np.uint16) | sign_bit).view(np.float16).astype(np.float32).reshape(2, 2)
                result = G.half(np.tile(cell, (2, 2)), numerics=G.Numerics(G.REFERENCE_MODEL))
                self.assertTrue(np.all(result.astype(np.float16).view(np.uint16) == expected | sign_bit))

    def test_configuration_and_binding_isolation(self):
        original = tuple(getattr(G, name) for name in ("bilinear", "_match", "_refine", "fbcheck"))
        first, second = G.Numerics(G.REFERENCE_MODEL), G.Numerics(G.REFERENCE_MODEL)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.model_id = "automatic"
        for model_id in ("automatic", "subtexel8-fp16-rtz"):
            with self.assertRaises(ValueError):
                G.Numerics(model_id)
        image = np.zeros((48, 48), np.float32)
        for model in (first, second, first):
            forward, reverse, scale = G.arm_fields(image, image, "checked/8", None, numerics=model)
            self.assertEqual((forward.shape, reverse.shape, scale), ((6, 6, 4), (6, 6, 4), 8))
        self.assertEqual(original, tuple(getattr(G, name) for name in ("bilinear", "_match", "_refine", "fbcheck")))
        temporal = load("temporal_model_contract", "framegen-temporal-gate.py")
        for model in (first, second):
            fields = temporal.fields("checked/8", image, image, "coded", numerics=model)
            self.assertEqual(fields[0].shape, (6, 6, 4))

    def test_parity_keeps_zero_confidence_and_point02_boundary(self):
        image = np.zeros((2, 2), np.float32)
        output = np.zeros((2, 2, 4), np.float32)
        gpu = output.copy()
        gpu[0, 0, 0] = .02
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "fg_000000.bin").touch()
            with mock.patch.object(G.T, "load_sample", return_value=(image, image, gpu, output)), \
                 mock.patch.object(G, "checked_pair", return_value=(output, output)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(G.parity(directory, numerics=G.Numerics(G.REFERENCE_MODEL)), 1)
            nonfinite = output.copy()
            nonfinite[0, 0, 0] = np.nan
            with mock.patch.object(G.T, "load_sample", return_value=(image, image, output, output)), \
                 mock.patch.object(G, "checked_pair", return_value=(nonfinite, output)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(G.parity(directory, numerics=G.Numerics(G.REFERENCE_MODEL)), 1)

    def test_cli_requires_numerical_provenance(self):
        command = [sys.executable, str(SCRIPTS / "framegen-field-gate.py"), "--parity", "/unused"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("GSFD v1 carries no numerical provenance", result.stderr)
        result = subprocess.run(command + ["--reference-model", "subtexel8-fp16-rtz"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)
        temporal = [sys.executable, str(SCRIPTS / "framegen-temporal-gate.py"), "--size", "8x8", "unused"]
        result = subprocess.run(temporal, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("GSFD v1 carries no numerical provenance", result.stderr)

    def test_file_loaded_consumers_share_one_independent_core(self):
        core_only = r'''
import sys
sys.path.insert(0, sys.argv[1])
import framegen_field_reference as core
assert "framegen_net_train" not in sys.modules
assert "framegen_net_eval" not in sys.modules
assert core.Numerics(core.REFERENCE_MODEL).subtexel_bits == 8
'''
        consumers = r'''
import importlib.util
from pathlib import Path
import sys

scripts = Path(sys.argv[1])
def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, scripts / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Exercise each entrypoint's bootstrap independently in a fresh process.
load("first_consumer_contract", sys.argv[2])
trainer = load("trainer_contract", "framegen-net-train.py")
import framegen_field_reference as core
gate = load("gate_contract", "framegen-field-gate.py")
temporal = load("temporal_contract", "framegen-temporal-gate.py")
evaluator = load("evaluator_contract", "framegen-net-eval.py")
hindsight = load("hindsight_contract", "framegen-hindsight-eval.py")
assert Path(core.__file__).resolve() == scripts / "framegen_field_reference.py"
assert gate.Numerics is temporal.G.Numerics is core.Numerics
assert gate.checked_pair is temporal.G.checked_pair is core.checked_pair
assert trainer.bilinear is gate.T.bilinear is evaluator.T.bilinear
assert trainer.bilinear is hindsight.T.bilinear is core.bilinear_with_grad
'''
        with tempfile.TemporaryDirectory() as directory:
            env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
            for first in ("core", "framegen-net-train.py", "framegen-field-gate.py"):
                program = core_only if first == "core" else consumers
                with self.subTest(first=first):
                    result = subprocess.run([sys.executable, "-c", program, str(SCRIPTS), first],
                                            cwd=directory, env=env, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_shared_sampler_preserves_values_and_analytic_gradients(self):
        image = np.array([[0., 2.], [4., 6.]], np.float32)
        coordinate = np.array([.25, 1., 2.], np.float32)
        value, dy, dx = G.T.bilinear(image, coordinate, coordinate)
        for actual, expected in ((value, [0., 3., 6.]), (dy, [0., 4., 0.]), (dx, [0., 2., 0.])):
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_array_equal(actual, np.array(expected, np.float32))
        np.testing.assert_array_equal(G.bilinear(image, coordinate, coordinate), value)


if __name__ == "__main__":
    unittest.main()
