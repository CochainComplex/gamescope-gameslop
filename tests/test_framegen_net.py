#!/usr/bin/env python3
"""CPU refiner math, neutral initialization and shared-import contracts."""
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
    print("numpy unavailable; offline network core not exercised")
    raise SystemExit(77)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
N = importlib.import_module("framegen_net")


class NetContract(unittest.TestCase):
    def test_fixed_layout_neutral_head_and_rng_progression(self):
        rng = np.random.default_rng(7)
        params = N.init_params(rng)
        self.assertEqual(list(params), ["W1", "b1", "W2", "b2", "W3", "b3"])
        for i, (ci, co) in enumerate(((12, 16), (16, 16), (16, 4)), 1):
            self.assertEqual(params[f"W{i}"].shape, (co, ci, 3, 3))
            self.assertEqual(params[f"b{i}"].shape, (co,))
            for key in (f"W{i}", f"b{i}"):
                self.assertEqual(params[key].dtype, np.float32)
                self.assertTrue(params[key].flags.c_contiguous)
            np.testing.assert_array_equal(params[f"b{i}"], 0)
        np.testing.assert_array_equal(params["W3"], 0)
        # Only the two hidden weight tensors draw from the caller's generator.
        expected = np.random.default_rng(7)
        expected.standard_normal(16 * 12 * 9 + 16 * 16 * 9)
        self.assertEqual(rng.bit_generator.state, expected.bit_generator.state)
        np.testing.assert_array_equal(rng.random(16), expected.random(16))
        output, cache = N.net_fwd(params, np.ones((2, 12, 9, 11), np.float32))
        self.assertEqual(output.shape, (2, 4, 3, 5))
        np.testing.assert_array_equal(output, 0)
        self.assertEqual(cache[1].shape, (2, 16, 7, 9))
        self.assertEqual(cache[2].shape, (2, 16, 5, 7))

    def test_valid_cross_correlation_with_asymmetric_weights_and_strided_input(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(2, 2, 10, 14))[:, :, ::2, ::2]
        weights = rng.normal(size=(3, 2, 3, 3))
        bias = rng.normal(size=3)
        expected = np.empty((2, 3, 3, 5))
        for b, o, y, x0 in np.ndindex(expected.shape):
            # Scalar oracle checks orientation and valid spatial extents.
            expected[b, o, y, x0] = bias[o] + sum(
                x[b, i, y + ky, x0 + kx] * weights[o, i, ky, kx]
                for i, ky, kx in np.ndindex(2, 3, 3))
        np.testing.assert_allclose(N.conv_fwd(x, weights, bias), expected,
                                   rtol=1e-13, atol=1e-13)

    def test_convolution_gradients_against_directional_derivatives(self):
        rng = np.random.default_rng(23)
        x = rng.normal(size=(2, 2, 5, 7))
        weights = rng.normal(size=(3, 2, 3, 3))
        bias = rng.normal(size=3)
        dout = rng.normal(size=(2, 3, 3, 5))
        inputs = (x, weights, bias)
        before = [v.copy() for v in (*inputs, dout)]
        grads = N.conv_bwd(x, weights, dout)
        for i, (value, grad) in enumerate(zip(inputs, grads)):
            direction = rng.normal(size=value.shape)
            direction /= np.linalg.norm(direction)
            plus, minus = list(inputs), list(inputs)
            plus[i] = value + 1e-5 * direction
            minus[i] = value - 1e-5 * direction
            numerical = np.sum((N.conv_fwd(*plus) - N.conv_fwd(*minus)) * dout) / 2e-5
            self.assertAlmostEqual(float(np.sum(grad * direction)), float(numerical), places=8)
            self.assertEqual(grad.shape, value.shape)
        for value, saved in zip((*inputs, dout), before):
            np.testing.assert_array_equal(value, saved)

    def test_nonzero_head_network_gradients_and_cache_ownership(self):
        rng = np.random.default_rng(29)
        params = {key: value.astype(np.float64) for key, value in N.init_params(rng).items()}
        params["W3"][:] = rng.normal(0, .1, params["W3"].shape)
        x = rng.normal(size=(2, 12, 9, 11))
        output, cache = N.net_fwd(params, x)
        self.assertIs(cache[0], x)
        self.assertFalse(np.shares_memory(cache[1], x))
        self.assertFalse(np.shares_memory(cache[2], cache[1]))
        dout = rng.normal(size=output.shape)
        arrays = [*params.values(), *cache, dout]
        before = [v.copy() for v in arrays]
        grads = N.net_bwd(params, cache, dout)
        self.assertEqual(list(grads), list(params))
        for key, value in params.items():
            with self.subTest(parameter=key):
                self.assertEqual(grads[key].shape, value.shape)
                self.assertGreater(np.linalg.norm(grads[key]), 0)
                direction = rng.normal(size=value.shape)
                direction /= np.linalg.norm(direction)
                plus, minus = dict(params), dict(params)
                plus[key] = value + 1e-5 * direction
                minus[key] = value - 1e-5 * direction
                a, _ = N.net_fwd(plus, x)
                b, _ = N.net_fwd(minus, x)
                numerical = np.sum((a - b) * dout) / 2e-5
                self.assertAlmostEqual(float(np.sum(grads[key] * direction)), float(numerical), places=8)
        for value, saved in zip(arrays, before):
            np.testing.assert_array_equal(value, saved)

    def test_relu_zero_has_no_hidden_layer_gradient(self):
        params = {key: np.zeros_like(value) for key, value in
                  N.init_params(np.random.default_rng(31)).items()}
        params["W3"][:] = 1
        output, cache = N.net_fwd(params, np.ones((1, 12, 7, 7), np.float32))
        grads = N.net_bwd(params, cache, np.ones_like(output))
        for key in ("W1", "b1", "W2", "b2", "W3"):
            np.testing.assert_array_equal(grads[key], 0)
        np.testing.assert_array_equal(grads["b3"], 1)

    def test_fresh_file_loaded_consumers_share_independent_core(self):
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

if sys.argv[2] == "core":
    sys.path.insert(0, str(scripts))
    import framegen_net as core
    assert "framegen_field_reference" not in sys.modules
    assert "framegen_net_train" not in sys.modules
    assert "framegen_net_eval" not in sys.modules
else:
    load("first_consumer", sys.argv[2])
    import framegen_net as core
    trainer = load("trainer_contract", "framegen-net-train.py")
    evaluator = load("eval_contract", "framegen-net-eval.py")
    hindsight = load("hindsight_contract", "framegen-hindsight-eval.py")
    field = load("field_contract", "framegen-field-gate.py")
    for name in ("conv_fwd", "conv_bwd", "net_fwd", "net_bwd", "init_params"):
        for consumer in (trainer, evaluator.T, hindsight.T, field.T, field.NE.T):
            assert getattr(consumer, name) is getattr(core, name)
assert Path(core.__file__).resolve() == scripts / "framegen_net.py"
'''
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        with tempfile.TemporaryDirectory() as directory:
            for first in ("core", "framegen-net-train.py", "framegen-net-eval.py",
                          "framegen-hindsight-eval.py", "framegen-field-gate.py"):
                with self.subTest(first=first):
                    result = subprocess.run([sys.executable, "-c", program, str(SCRIPTS), first],
                                            cwd=directory, env=env, capture_output=True,
                                            text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
