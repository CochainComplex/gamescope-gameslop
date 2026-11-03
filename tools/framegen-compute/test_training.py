#!/usr/bin/env python3
"""CPU wire/admission/publication contracts; never load an accelerator plugin."""

import ctypes as C
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest
import weakref
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    print("numpy unavailable; training client CPU contracts not exercised")
    raise SystemExit(77)


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("compute_training_client_contract", HERE / "train.py")
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)


def sample(height=19, width=17):
    y, x = np.mgrid[:height, :width].astype(np.float32)
    source = x / 64 + y / 128
    destination = .125 + x / 128 - y / 256
    forward = np.stack((x / 32, -y / 16, np.full_like(x, .5), np.full_like(x, 1.5)), -1)
    reverse = np.stack((-x / 16, y / 32, np.full_like(x, .75), np.full_like(x, -.25)), -1)
    return source, destination, forward, reverse


def gsfd(luma_bpp=2, flags=1):
    """Independent GSFD wire fixture, including distinguishable unused channels."""
    arrays = sample()
    raw = struct.pack("<8IQ", 0x44465347, 1, 17, 19, luma_bpp, 8, flags, 0, 1234567)
    for luma in arrays[:2]:
        if luma_bpp == 8:
            luma = np.stack((luma, luma + 2, luma + 3, luma + 4), -1)
        raw += luma.astype("<f2").tobytes()
    for field in arrays[2:]:
        raw += field.astype("<f2").tobytes()
    return raw


def parameters():
    rng = np.random.default_rng(9102)
    result = {}
    for layer, (ci, co) in enumerate(((12, 16), (16, 16), (16, 4)), 1):
        result[f"W{layer}"] = rng.normal(0, .1, (co, ci, 3, 3)).astype(np.float32)
        result[f"b{layer}"] = rng.normal(0, .1, co).astype(np.float32)
    return result


def wire_weights(params):
    # Scalar enumeration is independent of the client's transpose/reshape path.
    values = []
    for layer, (ci, co) in enumerate(((12, 16), (16, 16), (16, 4)), 1):
        weights = params[f"W{layer}"]
        for out in range(co):
            for y in range(3):
                for x in range(3):
                    for channel in range(ci):
                        values.append(weights[out, channel, y, x])
        values.extend(params[f"b{layer}"])
    return np.array(values, np.float32)


def gsfr(flat):
    return (struct.pack("<12I", 0x52465347, 3, 3, 12, 16, 3, 16, 16, 3, 16, 4, 3)
            + np.asarray(flat, dtype="<f4").tobytes())


class TrainingClientContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def file(self, name, raw):
        path = self.root / name
        path.write_bytes(raw)
        return path

    @unittest.skipUnless(C.sizeof(C.c_void_p) == 8, "ABI offsets specify naturally aligned 64-bit hosts")
    def test_public_c_abi_sizes_and_alignment(self):
        for kind, expected in ((T.Device, 180), (T.Create, 188), (T.Ticket, 16),
                               (T.Job, 128), (T.Output, 56), (T.Api, 104)):
            with self.subTest(structure=kind.__name__):
                self.assertEqual(C.sizeof(kind), expected)
        for kind, offsets in ((T.Job, {"epoch": 8, "width": 16, "step": 24,
                                      "luma_count": 48, "state_count": 72,
                                      "source": 80, "state": 120}),
                              (T.Output, {"gradient_count": 8, "served_count": 24,
                                          "gradients": 32, "served": 48}),
                              (T.Api, {"name": 24, "devices": 32, "destroy": 96})):
            for field, offset in offsets.items():
                self.assertEqual(getattr(kind, field).offset, offset)

    def test_sample_converts_strided_real_input_without_mutating_it(self):
        originals = sample(38, 34)
        inputs = tuple(a[::2, ::2].astype(np.float64) for a in originals)
        # Make genuinely noncontiguous views after conversion.
        inputs = tuple(np.repeat(a, 2, axis=1)[:, ::2] for a in inputs)
        before = tuple(a.copy() for a in inputs)
        prepared = T.prepare_sample(inputs)
        for actual, original, saved in zip(prepared, inputs, before):
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(actual.flags.c_contiguous)
            np.testing.assert_array_equal(actual, original.astype(np.float32))
            np.testing.assert_array_equal(original, saved)

    def test_sample_rejects_dimensions_shapes_and_nonreal_inputs(self):
        valid = sample()
        cases = [valid[:3], (np.zeros(17), *valid[1:]),
                 sample(15, 17), sample(19, 15),
                 (np.broadcast_to(0., (1025, 1024)), *valid[1:]),
                 (valid[0], valid[1][:, :-1], *valid[2:]),
                 (valid[0], valid[1], valid[2][..., :3], valid[3])]
        for dtype in (np.complex64, np.bool_, object, "U16"):
            cases.append((valid[0].astype(dtype), *valid[1:]))
        for index, case in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                T.prepare_sample(case)

    def test_sample_rejects_nonfinite_and_float32_overflow_in_every_tensor(self):
        for tensor in range(4):
            for value in (np.nan, np.inf, -np.inf, 1e100):
                arrays = [a.astype(np.float64) for a in sample()]
                arrays[tensor].flat[-1] = value
                with self.subTest(tensor=tensor, value=value), self.assertRaises(ValueError):
                    T.prepare_sample(arrays)

    def test_optimizer_supported_flags_and_closed_scalar_bounds(self):
        for flags in (0, 1, 4, 5):
            for edge in (0., 1.):
                result = T.optimizer_controls(1, flags, 1, edge, edge, edge)
                self.assertEqual(result, (1, flags, 1, edge, edge, edge))
        for step, flags, cut in ((0, 0, 0), (-1, 0, 0), (True, 0, 0),
                                 (1., 0, 0), (2**32, 0, 0), (1, 2, 0),
                                 (1, 8, 0), (1, 0, 2), (1, 0, -1)):
            with self.subTest(step=step, flags=flags, cut=cut), self.assertRaises(ValueError):
                T.optimizer_controls(step, flags, cut, .001, .0625, .0005)

    def test_optimizer_rejects_nonfinite_overflow_and_out_of_range_scalars(self):
        for index in range(3):
            for value in (np.nan, np.inf, -np.inf, -0.01, 1e100):
                controls = [.001, .0625, .0005]
                controls[index] = value
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    T.optimizer_controls(1, 0, 0, *controls)
        for index in (1, 2):
            controls = [.001, .0625, .0005]
            controls[index] = 1.001
            with self.assertRaises(ValueError):
                T.optimizer_controls(1, 0, 0, *controls)

    def test_parameter_wire_order_uses_output_tap_input_not_numpy_order(self):
        params = parameters()
        expected = wire_weights(params)
        actual = T.flatten_params(params)
        self.assertEqual(actual.shape, (4644,))
        np.testing.assert_array_equal(actual, expected)
        # Fixed ABI boundary probes distinguish biases from the next layer.
        for index, value in ((1728, params["b1"][0]), (1744, params["W2"][0, 0, 0, 0]),
                             (4048, params["b2"][0]), (4064, params["W3"][0, 0, 0, 0]),
                             (4640, params["b3"][0]), (4643, params["b3"][3])):
            self.assertEqual(actual[index], value)
        restored = T.unflatten_params(expected)
        for key in params:
            np.testing.assert_array_equal(restored[key], params[key])
            self.assertTrue(restored[key].flags.c_contiguous)
        with self.assertRaises(ValueError):
            T.unflatten_params(expected[:-1])
        bad = expected.copy(); bad[4064] = np.inf
        with self.assertRaises(ValueError):
            T.unflatten_params(bad)

    def test_gsfr_independent_wire_and_served_codec_roundtrip(self):
        expected = wire_weights(parameters())
        path = self.file("prior.gsfr", gsfr(expected))
        np.testing.assert_array_equal(T.load_net_strict(path), expected)
        output, receipt = T.output_paths(self.root / "trained.gsfr", [path])
        T.publish(expected, output, receipt, {"scope": "CPU wire fixture"})
        self.assertEqual(output.read_bytes(), gsfr(expected))
        np.testing.assert_array_equal(T.load_net_strict(output), expected)
        self.assertEqual(json.loads(receipt.read_text())["output_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["prior.gsfr", "trained.gsfr", "trained.gsfr.json"])

    def test_gsfr_rejects_versions_truncation_tail_bad_shape_and_nonfinite(self):
        raw = gsfr(wire_weights(parameters()))
        wrong_version = bytearray(raw); struct.pack_into("<I", wrong_version, 4, 2)
        wrong_shape = bytearray(raw); struct.pack_into("<I", wrong_shape, 12, 13)
        nonfinite = bytearray(raw); struct.pack_into("<f", nonfinite, 48 + 4064 * 4, np.nan)
        for index, bad in enumerate((raw[:20], raw[:-1], raw + b"x", wrong_version, wrong_shape, nonfinite)):
            with self.subTest(case=index), self.assertRaises((ValueError, SystemExit)):
                T.load_net_strict(self.file(f"bad-{index}.gsfr", bad))
        path = self.file("good.gsfr", raw)
        with self.assertRaisesRegex(ValueError, "changed after receipt hashing"):
            T.load_net_strict(path, expected_sha256="0" * 64)

    def test_gsfd_odd_dimensions_half_and_rgba_luma_keep_plane_identity(self):
        for bpp in (2, 8):
            path = self.file(f"capture-{bpp}.bin", gsfd(bpp))
            actual = T.load_sample_strict(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for decoded, expected in zip(actual, sample()):
                np.testing.assert_array_equal(decoded, expected.astype(np.float16).astype(np.float32))
                self.assertEqual(decoded.dtype, np.float32)
                self.assertTrue(decoded.flags.c_contiguous)

    def test_gsfd_rejects_length_layout_flags_dimensions_and_nonfinite_padding(self):
        raw = gsfd()
        cases = [raw[:39], raw[:-1], raw + b"tail"]
        for offset, value in ((0, 0), (4, 2), (8, 15), (16, 4), (20, 4), (24, 0)):
            bad = bytearray(raw); struct.pack_into("<I", bad, offset, value); cases.append(bad)
        for bpp, offset in ((2, 40), (8, 42)):
            bad = bytearray(gsfd(bpp)); struct.pack_into("<e", bad, offset, float("nan")); cases.append(bad)
        for index, bad in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                T.load_sample_strict(self.file(f"bad-{index}.bin", bad))
        path = self.file("good.bin", raw)
        with self.assertRaisesRegex(ValueError, "changed after receipt hashing"):
            T.load_sample_strict(path, expected_sha256="0" * 64)

    def test_output_paths_reject_input_alias_receipt_collision_and_dangling_link(self):
        source = self.file("prior.gsfr", b"immutable")
        with self.assertRaises(ValueError):
            T.output_paths(source, [source])
        alias = self.root / "alias.gsfr"; alias.symlink_to(source)
        with self.assertRaises(ValueError):
            T.output_paths(alias, [source])
        for suffix in ("", ".json"):
            target = self.root / f"collision{suffix}"
            target.write_bytes(b"existing")
            base = self.root / "collision"
            with self.assertRaises(ValueError):
                T.output_paths(base, [])
            target.unlink()
        dangling = self.root / "dangling"; dangling.symlink_to(self.root / "missing")
        with self.assertRaises(ValueError):
            T.output_paths(dangling, [])
        with self.assertRaises(ValueError):
            T.output_paths(self.root / "missing-dir/out.gsfr", [])

    def test_atomic_publication_never_overwrites_collision_and_rolls_back_own_link(self):
        served = wire_weights(parameters())
        for collision in ("model", "receipt"):
            output = self.root / "result.gsfr"
            receipt = self.root / "result.gsfr.json"
            existing = output if collision == "model" else receipt
            existing.write_bytes(b"keep this exact existing artifact")
            with self.assertRaises(FileExistsError):
                T.publish(served, output, receipt, {})
            self.assertEqual(existing.read_bytes(), b"keep this exact existing artifact")
            self.assertFalse((receipt if collision == "model" else output).exists())
            self.assertEqual(list(self.root.iterdir()), [existing])
            existing.unlink()

    def fake_api(self):
        callbacks = []
        api = T.Api()
        api.size, api.abi, api.model_version, api.parameters, api.slices = 104, 1, 3, 4644, 16
        api.name = b"CPU function table fixture"
        for field, kind in T.Api._fields_:
            if field in ("devices", "create", "submit", "poll", "collect", "discard", "retire", "reset", "destroy"):
                callback = kind(lambda *args: 0)
                callbacks.append(callback)
                setattr(api, field, callback)
        return api, callbacks

    def test_failed_destroy_retains_owner_and_forbids_reuse_until_close_succeeds(self):
        context = T.Context.__new__(T.Context)
        context.handle = C.c_void_p(123)
        context.error = C.create_string_buffer(64)
        context._retained = False
        context._active = None
        context.epoch = 1
        context.api = types.SimpleNamespace(destroy=lambda *args: T.FAILED)
        with self.assertRaisesRegex(RuntimeError, "context and module retained"):
            context.close()
        self.assertIn(context, T._retained_contexts)
        self.assertEqual(context.handle.value, 123)
        with self.assertRaisesRegex(RuntimeError, "retirement is unresolved"):
            context.reset()
        with self.assertRaisesRegex(RuntimeError, "retirement is unresolved"):
            context.run(None, None, None, 1)
        context.api.destroy = lambda *args: T.OK
        context.close()
        self.assertNotIn(context, T._retained_contexts)
        self.assertFalse(context.handle.value)

    def test_plugin_rejects_wrong_table_identity_and_null_callbacks_without_loading(self):
        path = self.file("not-a-library.so", b"not native code")
        cases = [("size", 103), ("abi", 2), ("model_version", 2),
                 ("parameters", 4643), ("slices", 15), ("name", None), ("collect", T._Collect())]
        for field, value in cases:
            api, retained_callbacks = self.fake_api()
            setattr(api, field, value)
            getter = mock.Mock(return_value=C.pointer(api))
            with self.subTest(field=field), mock.patch.object(T.C, "CDLL", return_value=types.SimpleNamespace(gsc_training_get_api=getter)) as loader:
                with self.assertRaises(RuntimeError):
                    T.Plugin(path)
                loader.assert_called_once()
                getter.assert_called_once_with(1, C.sizeof(T.Api))
            self.assertEqual(len(retained_callbacks), 9)

    def test_plugin_accepts_complete_mock_table_and_rejects_null_getter_result(self):
        path = self.file("not-a-library.so", b"not native code")
        api, retained_callbacks = self.fake_api()
        getter = mock.Mock(return_value=C.pointer(api))
        with mock.patch.object(T.C, "CDLL", return_value=types.SimpleNamespace(gsc_training_get_api=getter)):
            plugin = T.Plugin(path)
            self.assertEqual(plugin.api.name, b"CPU function table fixture")
            getter.assert_called_once_with(1, C.sizeof(T.Api))
        getter = mock.Mock(return_value=C.POINTER(T.Api)())
        with mock.patch.object(T.C, "CDLL", return_value=types.SimpleNamespace(gsc_training_get_api=getter)):
            with self.assertRaisesRegex(RuntimeError, "rejected the exact training ABI"):
                T.Plugin(path)
        self.assertEqual(len(retained_callbacks), 9)

    def test_shared_descriptor_native_and_legacy_v1_sources(self):
        path = self.file("descriptor-fixture.so", b"CPU mock, not native code")
        for native in (False, True):
            api, retained = self.fake_api()
            getter = mock.Mock(return_value=C.pointer(api))
            module = types.SimpleNamespace(gsc_training_get_api=getter)
            # Independent shared snapshot-training wire: all 12 uint32 words.
            wire = T.BackendContract(48, 1, 2, 1, 2, 3, 2, 1, 1, 28, 1, 0)
            descriptor = mock.Mock(return_value=C.pointer(wire))
            if native:
                module.gsc_compute_get_contract = descriptor
            with self.subTest(native=native), mock.patch.object(T.C, "CDLL", return_value=module):
                plugin = T.Plugin(path)
            self.assertEqual(C.addressof(plugin.api), C.addressof(api))
            self.assertEqual(list(plugin.contract), [48, 1, 2, 1, 2, 3, 2, 1, 1, 28, 1, 0])
            self.assertEqual(plugin.contract_source, "native-descriptor" if native else "validated-training-v1")
            getter.assert_called_once_with(1, 104)
            if native:
                descriptor.assert_called_once_with(1, 48, 2)
            else:
                descriptor.assert_not_called()
            self.assertEqual(len(retained), 9)

    def test_shared_descriptor_rejects_malformed_or_other_operation_without_fallback(self):
        path = self.file("descriptor-fixture.so", b"CPU mock")
        for field, value in (("size", 44), ("abi", 2), ("payload", 99),
                             ("operation", 3), ("execution", 2), ("model_version", 2),
                             ("semantics", 3), ("state", 2), ("transport", 3),
                             ("features", 0), ("features", 12), ("features", 0xffffffff),
                             ("max_inflight", 2), ("reserved", 1)):
            api, retained = self.fake_api()
            wire = T.BackendContract(48, 1, 2, 1, 2, 3, 2, 1, 1, 28, 1, 0)
            setattr(wire, field, value)
            descriptor = mock.Mock(return_value=C.pointer(wire))
            module = types.SimpleNamespace(gsc_training_get_api=mock.Mock(return_value=C.pointer(api)),
                                           gsc_compute_get_contract=descriptor)
            with self.subTest(field=field, value=value), mock.patch.object(T.C, "CDLL", return_value=module):
                with self.assertRaises(ValueError):
                    T.Plugin(path)
            descriptor.assert_called_once_with(1, 48, 2)
            self.assertEqual(len(retained), 9)
        api, retained = self.fake_api()
        # Fully coherent resident schema is nevertheless the wrong payload for v1.
        resident = T.BackendContract(48, 1, 2, 1, 3, 3, 2, 2, 3, 23, 1, 0)
        descriptor = mock.Mock(return_value=C.pointer(resident))
        module = types.SimpleNamespace(gsc_training_get_api=mock.Mock(return_value=C.pointer(api)),
                                       gsc_compute_get_contract=descriptor)
        with mock.patch.object(T.C, "CDLL", return_value=module), self.assertRaises(ValueError):
            T.Plugin(path)
        descriptor.return_value = C.POINTER(T.BackendContract)()
        with mock.patch.object(T.C, "CDLL", return_value=module), self.assertRaisesRegex(RuntimeError, "rejected the shared typed"):
            T.Plugin(path)
        self.assertEqual(len(retained), 9)

    def test_v1_validation_precedes_optional_descriptor_negotiation(self):
        path = self.file("descriptor-fixture.so", b"CPU mock")
        api, retained = self.fake_api()
        api.abi = 99
        descriptor = mock.Mock(side_effect=AssertionError("must not negotiate after invalid v1"))
        module = types.SimpleNamespace(gsc_training_get_api=mock.Mock(return_value=C.pointer(api)),
                                       gsc_compute_get_contract=descriptor)
        with mock.patch.object(T.C, "CDLL", return_value=module), self.assertRaisesRegex(RuntimeError, "incompatible plugin ABI"):
            T.Plugin(path)
        descriptor.assert_not_called()
        self.assertEqual(len(retained), 9)

    def test_context_binding_retains_native_callbacks_and_plugin_module(self):
        class FakeModule:
            pass
        path = self.file("descriptor-fixture.so", b"CPU mock")
        api, retained = self.fake_api()
        def create(info, handle, error, size):
            handle[0] = 123
            return T.OK
        create_callback = T._Create(create)
        retained.append(create_callback)
        api.create = create_callback
        module = FakeModule()
        module.gsc_training_get_api = mock.Mock(return_value=C.pointer(api))
        module_ref = weakref.ref(module)
        with mock.patch.object(T.C, "CDLL", return_value=module):
            plugin = T.Plugin(path)
        plugin_ref = weakref.ref(plugin)
        device = T.Device(); device.size = C.sizeof(T.Device)
        context = T.Context(plugin, device)
        binding = context.binding
        self.assertIs(binding.owner, plugin)
        for alias, native in (("submit", "submit"), ("poll", "poll"), ("collect", "collect"),
                              ("discard", "discard"), ("retire", "retire"),
                              ("reset", "reset"), ("close", "destroy")):
            self.assertEqual(C.cast(getattr(binding, alias), C.c_void_p).value,
                             C.cast(getattr(api, native), C.c_void_p).value)
        self.assertIsInstance(binding.submit, T._Submit)
        del module, plugin
        gc.collect()
        self.assertIsNotNone(module_ref())
        self.assertIsNotNone(plugin_ref())
        context.close()
        del context
        gc.collect()
        self.assertIsNotNone(plugin_ref())  # Binding alone retains code ownership.
        del binding
        gc.collect()
        self.assertIsNone(plugin_ref())
        self.assertIsNone(module_ref())


if __name__ == "__main__":
    unittest.main()
