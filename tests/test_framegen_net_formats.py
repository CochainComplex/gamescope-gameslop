#!/usr/bin/env python3
"""Offline GSFR/GSFP wire and admission contracts, independent of GPU parity."""
import contextlib
import importlib
import io
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    print("numpy unavailable; offline net formats not exercised")
    raise SystemExit(77)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
F = importlib.import_module("framegen_net_formats")


def blob(version=3):
    # Independent public wire fixture: c_out/ky/kx/c_in weights, then biases.
    raw = struct.pack("<12I", 0x52465347, version, 3, 12, 16, 3, 16, 16, 3, 16, 4, 3)
    for ci, co in ((12, 16), (16, 16), (16, 4)):
        raw += ((np.arange(co * 9 * ci, dtype=np.float32) + 1) / 128).astype("<f4").tobytes()
        raw += ((np.arange(co, dtype=np.float32) + 1) / 8).astype("<f4").tobytes()
    return raw


def profile(entries, version=1):
    return struct.pack("<3I", 0x50465347, version, len(entries)) + b"".join(
        struct.pack("<I", key) + data for key, data in entries)


class NetFormatContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.bin"

    def load(self, raw, key=None):
        self.path.write_bytes(raw)
        return F.load_blob(self.path, key)

    def assert_exit(self, raw, message, key=None):
        with self.assertRaises(SystemExit) as caught:
            self.load(raw, key)
        self.assertEqual(caught.exception.code, f"{self.path}: {message}")

    def test_wire_layout_and_export_round_trip(self):
        raw = blob()
        params = self.load(raw)
        self.assertEqual(len(raw), 48 + 4644 * 4)
        self.assertEqual(params["W1"].shape, (16, 12, 3, 3))
        # A single non-symmetric coordinate checks the shader's storage order.
        self.assertEqual(params["W1"][1, 3, 2, 1], (1 * 9 * 12 + 2 * 3 * 12 + 1 * 12 + 3 + 1) / 128)
        self.assertEqual(params["b3"][3], .5)
        for value in params.values():
            self.assertEqual(value.dtype, np.float32)
            self.assertTrue(value.flags.c_contiguous)
            self.assertTrue(value.flags.writeable)
        output = Path(self.directory.name) / "roundtrip.gsfr"
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            F.export(params, output)
        self.assertEqual(output.read_bytes(), raw)
        self.assertEqual(stream.getvalue(), f"exported 4644 floats -> {output}\n")

    def test_noncurrent_bare_versions_warn_without_shading_migration(self):
        for version in (0, 1, 2, 4, 0xffffffff):
            with self.subTest(version=version):
                stream = io.StringIO()
                with contextlib.redirect_stderr(stream):
                    params = self.load(blob(version))
                self.assertEqual(stream.getvalue(),
                                 f"note: blob version {version} != trainer 3; shapes match, evaluating anyway\n")
                self.assertTrue(np.all(params["W3"][3] != 0))
                self.assertEqual(params["b3"][3], .5)

    def test_bare_trailing_bytes_and_key_argument_are_rejected(self):
        self.assert_exit(blob() + b"x", "1 trailing bytes")
        self.assert_exit(blob(), "a bare GSFR blob has no keys", key=0xf0)

    def test_profile_selection_preserves_temporal_spatial_and_unknown_keys(self):
        for key in (0xf0, 0x100f0, 0xdeadbeef):
            with self.subTest(key=key):
                params = self.load(profile([(key, blob())]))
                self.assertEqual(params["b3"][3], .5)
                params = self.load(profile([(key, blob())]), key)
                self.assertEqual(params["b3"][3], .5)
        spatial = bytearray(blob())
        struct.pack_into("<f", spatial, len(spatial) - 4, -.75)
        entries = [(0xf0, blob()), (0x100f0, spatial)]
        pair = profile(entries)
        self.assert_exit(pair, "pick a set with --net-key (0x0f0, 0x100f0)")
        self.assert_exit(pair, "pick a set with --net-key (0x0f0, 0x100f0)", key=17)
        for ordered in (entries, list(reversed(entries))):
            with self.subTest(order=[key for key, _ in ordered]):
                self.assertEqual(self.load(profile(ordered), 0xf0)["b3"][3], .5)
                self.assertEqual(self.load(profile(ordered), 0x100f0)["b3"][3], -.75)

    def test_profile_embedded_version_count_duplicates_and_length(self):
        self.assert_exit(profile([(0xf0, blob(2))]), "profile set has GSFR v2, want v3")
        self.assert_exit(profile([]), "unsupported GSFP v1 with 0 sets")
        self.assert_exit(profile([(i, blob()) for i in range(9)]), "unsupported GSFP v1 with 9 sets")
        self.assert_exit(profile([(0xf0, blob())], version=2), "unsupported GSFP v2 with 1 sets")
        self.assert_exit(profile([(0xf0, blob()), (0xf0, blob())]), "duplicate key 0x0f0")
        self.assert_exit(profile([(0xf0, blob())]) + b"x", "1 trailing bytes")

    def test_short_input_keeps_existing_exception_categories(self):
        self.assert_exit(b"short", "too short")
        with self.assertRaises(struct.error):
            self.load(struct.pack("<3I", 0x52465347, 3, 3))
        with self.assertRaises(struct.error):
            self.load(struct.pack("<3I", 0x50465347, 1, 1) + b"xy")
        self.assert_exit(blob()[:-1], "truncated")

    def test_nonfinite_load_rejection_and_signed_zero_transport(self):
        for bits in (0x7fc00000, 0x7f800000, 0xff800000):
            with self.subTest(bits=bits):
                raw = bytearray(blob())
                struct.pack_into("<I", raw, 48, bits)
                self.assert_exit(raw, "non-finite weights")
        raw = bytearray(blob())
        struct.pack_into("<I", raw, 48, 0x80000000)
        params = self.load(raw)
        self.assertEqual(params["W1"].view(np.uint32)[0, 0, 0, 0], 0x80000000)
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            F.export(params, self.path)
        self.assertEqual(self.path.read_bytes(), raw)

    def test_export_keeps_existing_nonfinite_admission(self):
        params = self.load(blob())
        params["b3"][3] = np.nan
        with contextlib.redirect_stdout(io.StringIO()):
            F.export(params, self.path)
        with self.assertRaises(SystemExit) as caught:
            F.load_blob(self.path)
        self.assertEqual(caught.exception.code, f"{self.path}: non-finite weights")

    def test_fresh_file_loaded_consumers_share_one_independent_format_leaf(self):
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
    import framegen_net_formats as formats
    assert "framegen_net_train" not in sys.modules
    assert "framegen_net_eval" not in sys.modules
    assert "framegen_field_reference" not in sys.modules
else:
    load("first_consumer", sys.argv[2])
    import framegen_net_formats as formats
    trainer = load("trainer_contract", "framegen-net-train.py")
    evaluator = load("eval_contract", "framegen-net-eval.py")
    hindsight = load("hindsight_contract", "framegen-hindsight-eval.py")
    field = load("field_contract", "framegen-field-gate.py")
    assert trainer.export is evaluator.T.export is field.T.export is formats.export
    assert trainer.LAYERS is evaluator.T.LAYERS is formats.LAYERS
    assert evaluator.load_blob is field.NE.load_blob is formats.load_blob
    assert evaluator._parse_gsfr is formats._parse_gsfr
    assert hindsight.T.export is formats.export
assert Path(formats.__file__).resolve() == scripts / "framegen_net_formats.py"
'''
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        for first in ("leaf", "framegen-net-train.py", "framegen-net-eval.py",
                      "framegen-hindsight-eval.py", "framegen-field-gate.py"):
            with self.subTest(first=first):
                result = subprocess.run([sys.executable, "-c", program, str(SCRIPTS), first],
                                        cwd=self.directory.name, env=env, capture_output=True,
                                        text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
