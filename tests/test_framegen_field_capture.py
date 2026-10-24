#!/usr/bin/env python3
"""GSFD plane, ownership and existing dataset-admission contracts."""
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
    print("numpy unavailable; offline field capture not exercised")
    raise SystemExit(77)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
D = importlib.import_module("framegen_field_capture")


def capture(luma_bpp=2):
    # Native-half payload matches the existing reader; the public header is LE.
    lumas = [np.arange(6, dtype=np.float16).reshape(2, 3) + offset for offset in (1, 11)]
    if luma_bpp == 8:
        lumas = [np.stack((plane, plane + 100, plane + 200, plane + 300), axis=-1).astype(np.float16)
                 for plane in lumas]
    fields = [np.arange(24, dtype=np.float16).reshape(2, 3, 4) + offset for offset in (100, 200)]
    header = struct.pack("<8I", 0x44465347, 1, 3, 2, luma_bpp, 8, 1, 0)
    return header + struct.pack("<Q", 77) + b"".join(v.tobytes() for v in (*lumas, *fields))


def header_word(raw, index, value):
    modified = bytearray(raw)
    struct.pack_into("<I", modified, index * 4, value)
    return bytes(modified)


class FieldCaptureContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.bin"

    def load(self, raw):
        self.path.write_bytes(raw)
        return D.load_sample(self.path)

    def test_luma_layouts_distinct_planes_and_storage_ownership(self):
        for luma_bpp in (2, 8):
            with self.subTest(luma_bpp=luma_bpp):
                raw = capture(luma_bpp)
                self.assertEqual(len(raw), 40 + 6 * (2 * luma_bpp + 2 * 8))
                previous, current, forward, reverse = self.load(raw)
                for plane, offset in ((previous, 1), (current, 11)):
                    expected = np.arange(6, dtype=np.float32).reshape(2, 3) + offset
                    np.testing.assert_array_equal(plane, expected)
                    self.assertEqual(plane.dtype, np.float32)
                    self.assertTrue(plane.flags.c_contiguous)
                    self.assertTrue(plane.flags.writeable)
                for plane, offset in ((forward, 100), (reverse, 200)):
                    expected = np.arange(24, dtype=np.float16).reshape(2, 3, 4) + offset
                    np.testing.assert_array_equal(plane, expected)
                    self.assertEqual(plane.dtype, np.float16)
                    self.assertTrue(plane.flags.c_contiguous)
                    self.assertFalse(plane.flags.writeable)
                    self.assertFalse(plane.flags.owndata)
                    with self.assertRaises(ValueError):
                        plane[0, 0, 0] = 0
                self.assertFalse(np.shares_memory(previous, current))
                self.assertFalse(np.shares_memory(forward, reverse))

    def test_half_edge_values_are_not_filtered_or_clamped(self):
        edge = np.array([0x8000, 0x0001, 0x7c00, 0x7e00, 0xc000, 0x4a00], np.uint16).view(np.float16)
        field = np.tile(edge, 4).reshape(2, 3, 4)
        raw = bytearray(capture())
        raw[40:52] = edge.tobytes()
        raw[64:112] = field.tobytes()
        previous, _, forward, _ = self.load(raw)
        np.testing.assert_array_equal(previous.view(np.uint32), edge.astype(np.float32).reshape(2, 3).view(np.uint32))
        np.testing.assert_array_equal(forward.view(np.uint16), field.view(np.uint16))

    def test_trailing_bytes_reserved_sequence_extra_flags_and_zero_width(self):
        raw = bytearray(header_word(header_word(capture(), 6, 0x80000001), 7, 42))
        raw[32:40] = b"\xff" * 8
        baseline = self.load(capture())
        permissive = self.load(raw + b"ignored trailing payload")
        for old, new in zip(baseline, permissive):
            np.testing.assert_array_equal(old, new)
        # No explicit dimension/luma-format policy is added by this extraction.
        planes = self.load(header_word(header_word(capture()[:40], 2, 0), 4, 6))
        self.assertEqual([p.shape for p in planes], [(2, 0), (2, 0), (2, 0, 4), (2, 0, 4)])

    def test_existing_header_layout_and_incidental_buffer_errors(self):
        for raw in (b"", capture()[:31]):
            with self.subTest(length=len(raw)), self.assertRaises(struct.error):
                self.load(raw)
        for raw in (header_word(capture(), 0, 0), header_word(capture(), 1, 2)):
            with self.assertRaises(ValueError) as caught:
                self.load(raw)
            self.assertEqual(str(caught.exception), f"{self.path}: not a GSFD v1 sample")
        for raw in (header_word(capture(), 5, 4), header_word(capture(), 6, 0)):
            with self.assertRaises(ValueError) as caught:
                self.load(raw)
            self.assertEqual(str(caught.exception), f"{self.path}: unsupported layout")
        for raw in (capture()[:32], capture()[:-1], header_word(capture(), 4, 4)):
            with self.assertRaises(ValueError):
                self.load(raw)
        with self.assertRaises(FileNotFoundError):
            D.load_sample(Path(self.directory.name) / "absent.bin")

    def test_dataset_lexical_order_skip_streams_and_mixed_dimensions(self):
        data = Path(self.directory.name)
        (data / "fg_10.bin").write_bytes(capture())
        second = bytearray(header_word(header_word(capture(), 2, 2), 3, 3))
        second[40:42] = struct.pack("=e", 17)
        (data / "fg_2.bin").write_bytes(second)
        bad = data / "fg_11_bad.bin"
        bad.write_bytes(header_word(capture(), 0, 0))
        (data / "ignored.bin").write_bytes(b"short")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            samples = D.load_dataset(data)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0][0].shape, (2, 3))
        self.assertEqual(samples[1][0].shape, (3, 2))
        self.assertEqual(samples[0][0][0, 0], 1)
        self.assertEqual(samples[1][0][0, 0], 17)
        self.assertEqual(out.getvalue(), "loaded 2 samples (3x2 field res)\n")
        self.assertEqual(err.getvalue(), f"skipping {bad}: not a GSFD v1 sample\n")

    def test_empty_all_invalid_and_struct_error_dataset_outcomes_stay_distinct(self):
        data = Path(self.directory.name)
        with self.assertRaises(SystemExit) as caught:
            D.load_dataset(data)
        self.assertEqual(caught.exception.code, f"no fg_*.bin samples under {data}")
        bad = data / "fg_bad.bin"
        bad.write_bytes(header_word(capture(), 0, 0))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertRaises(IndexError):
            D.load_dataset(data)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(err.getvalue(), f"skipping {bad}: not a GSFD v1 sample\n")
        bad.write_bytes(b"short")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), self.assertRaises(struct.error):
            D.load_dataset(data)
        self.assertEqual((out.getvalue(), err.getvalue()), ("", ""))

    def test_fresh_file_loaded_consumers_share_independent_reader(self):
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
    import framegen_field_capture as reader
    for name in ("framegen_field_reference", "framegen_net", "framegen_net_formats", "framegen_net_train"):
        assert name not in sys.modules
else:
    load("first_consumer", sys.argv[2])
    import framegen_field_capture as reader
    trainer = load("trainer_contract", "framegen-net-train.py")
    evaluator = load("eval_contract", "framegen-net-eval.py")
    hindsight = load("hindsight_contract", "framegen-hindsight-eval.py")
    field = load("field_contract", "framegen-field-gate.py")
    for consumer in (trainer, evaluator.T, hindsight.T, field.T, field.NE.T):
        assert consumer.load_sample is reader.load_sample
        assert consumer.load_dataset is reader.load_dataset
        assert consumer.MAGIC_SAMPLE == reader.MAGIC_SAMPLE
assert Path(reader.__file__).resolve() == scripts / "framegen_field_capture.py"
'''
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        for first in ("leaf", "framegen-net-train.py", "framegen-net-eval.py",
                      "framegen-hindsight-eval.py", "framegen-field-gate.py", "framegen-temporal-gate.py"):
            with self.subTest(first=first):
                result = subprocess.run([sys.executable, "-c", program, str(SCRIPTS), first],
                                        cwd=self.directory.name, env=env, capture_output=True,
                                        text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
