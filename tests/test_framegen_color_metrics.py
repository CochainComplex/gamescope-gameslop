#!/usr/bin/env python3
"""Stored-color metric contracts; no capture, driver or perceptual model needed."""
import importlib
import math
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
    print("numpy unavailable; offline color metrics not exercised")
    raise SystemExit(77)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
M = importlib.import_module("framegen_color_metrics")


class ColorMetricsContract(unittest.TestCase):
    def test_box_mean_uses_edge_padding_and_float32_result(self):
        image = np.array([[1., 2., 9.], [4., -3., 6.]], np.float32)
        for radius in (0, 1, 3):
            with self.subTest(radius=radius):
                padded = np.pad(image, radius, mode="edge")
                expected = np.empty(image.shape, np.float32)
                size = radius * 2 + 1
                for y, x in np.ndindex(image.shape):
                    expected[y, x] = padded[y:y+size, x:x+size].mean(dtype=np.float64)
                actual = M.box_mean(image, radius)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float32)

    def test_luma_preserves_channel_weights_and_hdr_range(self):
        np.testing.assert_array_equal(M.luma(np.eye(3, dtype=np.float32)),
                                      np.array([.2126, .7152, .0722], np.float32))
        rgb = np.array([[[-4., -4., -4.], [12., 12., 12.]]], np.float32)
        np.testing.assert_allclose(M.luma(rgb), [[-4., 12.]], rtol=1e-7)

    def test_ssim_identity_and_constant_reference_range_floor(self):
        image = np.arange(35, dtype=np.float32).reshape(5, 7) / 9 - 2
        self.assertEqual(M.ssim(image, image), 1.)
        zero, one = np.zeros((3, 5), np.float32), np.ones((3, 5), np.float32)
        self.assertAlmostEqual(M.ssim(zero, one), 1e-10 / (1. + 1e-10), delta=1e-15)

    def test_grade_uses_reference_hdr_scale_and_strict_bad_threshold(self):
        reference = np.full((3, 5, 3), 4., np.float32)
        generated = np.full_like(reference, 2.)
        result = M.grade({"reference": reference}, generated)
        self.assertEqual((result["mae"], result["bad%"], result["p95"], result["edge"]),
                         (.5, 100., .5, 0.))
        self.assertAlmostEqual(result["psnr"], 20 * math.log10(2))
        self.assertEqual(M.grade({"reference": generated}, reference)["mae"], 1.)
        boundary = np.float32(.05)
        pixels = np.array([0., boundary, np.nextafter(boundary, np.float32(1)),
                           np.nextafter(boundary, np.float32(0))], np.float32)
        generated = np.broadcast_to(pixels.reshape(2, 2, 1), (2, 2, 3)).copy()
        self.assertEqual(M.grade({"reference": np.zeros_like(generated)}, generated)["bad%"], 25.)

    def test_grade_sanitizes_components_counts_pixels_and_preserves_inputs(self):
        reference = np.zeros((2, 2, 3), np.float32)
        generated = reference.copy()
        reference[0, 0, 0] = np.nan
        generated[0, 1, 1] = np.inf
        generated[1, 0, 2] = -np.inf
        saved_reference, saved_generated = reference.copy(), generated.copy()
        result = M.grade({"reference": reference}, generated)
        self.assertEqual(result["nonfinite%"], 75.)
        self.assertEqual((result["mae"], result["bad%"], result["p95"], result["edge"]),
                         (0., 0., 0., 0.))
        self.assertEqual((result["psnr"], result["ssim"]), (math.inf, 1.))
        np.testing.assert_array_equal(reference, saved_reference)
        np.testing.assert_array_equal(generated, saved_generated)
        # A finite component beside Inf still contributes to the score.
        generated[0, 1, 0] = .25
        result = M.grade({"reference": reference}, generated)
        self.assertAlmostEqual(result["mae"], .25 / 12, places=8)
        self.assertEqual((result["bad%"], result["nonfinite%"]), (25., 75.))
        with self.assertRaises(ValueError):
            M.grade({"reference": reference[:1]}, generated[:1])

    def test_temporal_continuity_candidate_identity_and_block_partition(self):
        def sample(anchor, endpoint, timestamp, errors):
            reference = np.full((5, 3, 3), 4., np.float32)
            return {"anchor_id": anchor, "endpoint_id": endpoint,
                    "reference_ns": timestamp, "height": 5, "reference": reference,
                    "candidates": [(None, reference + error) for error in errors]}

        first = sample(10, 20, 100, (1., -1.))
        second = sample(20, 30, 200, (3., 0.))
        for rows in (1, 2, 3, 128):
            with self.subTest(rows=rows):
                self.assertEqual(M.temporal_residual_change([first, second], 0, rows), (.5, 1))
                self.assertEqual(M.temporal_residual_change([first, second], 1, rows), (.25, 1))
        broken_id = sample(21, 30, 200, (3., 0.))
        same_time = sample(20, 30, 100, (3., 0.))
        for samples in ([], [first], [first, broken_id], [first, same_time]):
            self.assertEqual(M.temporal_residual_change(samples, 0), (None, 0))
        # Temporal scoring intentionally differs from grade's sanitization.
        second["candidates"][0][1][0, 0, 0] = np.nan
        value, transitions = M.temporal_residual_change([first, second], 0)
        self.assertTrue(math.isnan(value))
        self.assertEqual(transitions, 1)

    def test_fresh_evaluator_and_leaf_share_metrics_without_heavy_imports(self):
        program = r'''
import importlib.util
from pathlib import Path
import sys
scripts = Path(sys.argv[1])
if sys.argv[2] == "leaf":
    sys.path.insert(0, str(scripts))
    import framegen_color_metrics as metrics
else:
    spec = importlib.util.spec_from_file_location("color_eval_contract", scripts / "framegen-color-eval.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    import framegen_color_metrics as metrics
    for name in ("box_mean", "ssim", "luma", "grade", "temporal_residual_change"):
        assert getattr(evaluator, name) is getattr(metrics, name)
assert Path(metrics.__file__).resolve() == scripts / "framegen_color_metrics.py"
for name in ("torch", "lpips", "framegen_net", "framegen_net_train", "framegen_field_reference"):
    assert name not in sys.modules
if sys.argv[2] == "leaf":
    assert "framegen_color_capture" not in sys.modules
'''
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        with tempfile.TemporaryDirectory() as directory:
            for first in ("leaf", "evaluator"):
                with self.subTest(first=first):
                    result = subprocess.run([sys.executable, "-c", program, str(SCRIPTS), first],
                                            cwd=directory, env=env, capture_output=True,
                                            text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
