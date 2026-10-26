#!/usr/bin/env python3
"""Crop feature ABI, sampler coordinates and confidence-evidence contracts."""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    print("numpy unavailable; offline net features not exercised")
    raise SystemExit(77)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
F = importlib.import_module("framegen_net_features")


def inputs(dtype=np.float32):
    # Both fields are strided views; typed literals avoid fixture promotion.
    x = np.zeros((12, 14, 4), dtype=dtype)[::2, ::2]
    y = np.zeros((12, 14, 4), dtype=dtype)[::2, ::2]
    x[:] = (1, -1, -.25, 4)
    y[:] = (-2, 3, 1.25, -2)
    source = np.arange(42, dtype=np.float32).reshape(6, 7) + np.float32(16)
    destination = np.full((6, 7), 32, np.float32)
    return x, y, source, destination


class NetFeatureContract(unittest.TestCase):
    def test_all_twelve_channels_nonzero_crop_and_one_sided_distance_cap(self):
        for dtype in (np.float16, np.float32):
            with self.subTest(dtype=str(dtype)):
                x, y, source, destination = inputs(dtype)
                feature = F.features(x, y, source, destination, 1, 2, 2, mode=1.75)
                expected = np.empty((12, 2, 2), np.float32)
                for channel, value in enumerate((.125, -.125, -.25, 1, -.25, .375, 1.25, -1)):
                    expected[channel] = value
                expected[8] = 32
                # Flow (1,-1) samples one column left and one row down.
                expected[9] = source[2:4, 1:3]
                expected[10] = np.array([
                    abs(32. - float(value)) / max(1., 32., abs(float(value)))
                    for value in source[2:4, 1:3].flat
                ], np.float32).reshape(2, 2)
                expected[11] = 1.75
                np.testing.assert_array_equal(feature, expected)
                self.assertEqual(feature.dtype, np.float32)
                self.assertTrue(feature.flags.c_contiguous)

    def test_sampler_clamps_motion_to_source_edges(self):
        for motion, value in ((100, 16), (-100, 57)):
            with self.subTest(motion=motion):
                x, y, source, destination = inputs()
                x[..., :2] = motion
                feature = F.features(x, y, source, destination, 0, 0, 2)
                np.testing.assert_array_equal(feature[9], np.full((2, 2), value, np.float32))
                np.testing.assert_array_equal(feature[11], 0)

    def test_half_subnormal_is_cast_before_feature_scaling(self):
        x = np.zeros((2, 3, 4), np.float16)
        y = np.zeros_like(x)
        x[0, 0, 0] = np.array([1], np.uint16).view(np.float16)[0]
        luma = np.ones((2, 3), np.float32)
        feature = F.features(x, y, luma, luma, 0, 0, 1)
        self.assertEqual(feature[0, 0, 0], np.float32(2. ** -27))
        self.assertGreater(feature[0, 0, 0], 0)
        self.assertEqual(feature.dtype, np.float32)

    def test_strided_inputs_are_not_mutated_or_retained_in_output(self):
        arrays = inputs(np.float16)
        before = [value.copy() for value in arrays]
        feature = F.features(*arrays, 1, 2, 2, mode=-1.5)
        np.testing.assert_array_equal(feature[11], -1.5)
        for value, saved in zip(arrays, before):
            np.testing.assert_array_equal(value, saved)
            self.assertFalse(np.shares_memory(feature, value))
        feature[:] = 0
        for value, saved in zip(arrays, before):
            np.testing.assert_array_equal(value, saved)

    def test_evidence_static_explanation_floor_fade_and_hdr_normalization(self):
        warped = np.array([0, 0, .03, .065, .1, 0, .02, 8, 4, -8], np.float32)
        destination = np.array([0, 0, 0, 0, 0, 0, 0, 8, 8, -8], np.float32)
        source = np.array([0, .2, .2, .2, .2, .01, .01, 4, 4, -4], np.float32)
        before = [value.copy() for value in (warped, destination, source)]
        gate = F.evidence_gate(warped, destination, source)
        expected = np.array([0, 1, .85, .3375, 0, 1/3, 0, 1, 0, 1], np.float32)
        np.testing.assert_allclose(gate, expected, rtol=0, atol=2e-7)
        self.assertEqual(gate.dtype, np.float32)
        for value, saved in zip((warped, destination, source), before):
            np.testing.assert_array_equal(value, saved)
        error = np.linspace(0, .12, 17, dtype=np.float32)
        gate = F.evidence_gate(error, np.zeros_like(error), np.ones_like(error))
        self.assertTrue(np.all((gate >= 0) & (gate <= 1)))
        self.assertTrue(np.all(np.diff(gate) <= 0))

    def test_fresh_file_loaded_consumers_share_features_and_canonical_sampler(self):
        program = r'''
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

if sys.argv[2] == "leaf":
    sys.path.insert(0, str(scripts))
    import framegen_net_features as features
    for name in ("framegen_net", "framegen_net_formats", "framegen_field_capture", "framegen_net_train"):
        assert name not in sys.modules
else:
    load("first_consumer", sys.argv[2])
    import framegen_net_features as features
    trainer = load("trainer_contract", "framegen-net-train.py")
    evaluator = load("eval_contract", "framegen-net-eval.py")
    hindsight = load("hindsight_contract", "framegen-hindsight-eval.py")
    field = load("field_contract", "framegen-field-gate.py")
    for consumer in (trainer, evaluator.T, hindsight.T, field.T, field.NE.T):
        assert consumer.features is features.features
        assert consumer.evidence_gate is features.evidence_gate
        assert consumer.bilinear is features.bilinear
import framegen_field_reference as reference
assert features.bilinear is reference.bilinear_with_grad
assert Path(features.__file__).resolve() == scripts / "framegen_net_features.py"
'''
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        with tempfile.TemporaryDirectory() as directory:
            for first in ("leaf", "framegen-net-train.py", "framegen-net-eval.py",
                          "framegen-hindsight-eval.py", "framegen-field-gate.py", "framegen-temporal-gate.py"):
                with self.subTest(first=first):
                    result = subprocess.run([sys.executable, "-c", program, str(SCRIPTS), first],
                                            cwd=directory, env=env, capture_output=True,
                                            text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
