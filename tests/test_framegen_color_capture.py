#!/usr/bin/env python3
"""GSCF wire, plane identity and code-value color contracts, without a GPU."""
import importlib
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
    print("numpy unavailable; offline color capture not exercised")
    raise SystemExit(77)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
C = importlib.import_module("framegen_color_capture")


def capture(version=3, header=None, meta=None, floats=(.25, 0, .5, 1), payload=None):
    # Public wire fixture, independent of the reader's constants/Struct/maps.
    planes, flags = (2, 1) if version == 1 else (4, 3)
    h = [0x46435347, version, 2, 2, 0x34324241, 4, 7, flags, planes, 1, 0, 0]
    if header:
        for index, value in header.items():
            h[index] = value
    metadata = [77, 100, 104, 108, 1000000000, 1250000000, 2000000000, 0]
    if meta:
        for index, value in meta.items():
            metadata[index] = value
    if payload is None:
        payload = b"".join(bytes((10 + i, 100 + i, 200 + i, 23)) * 4
                           for i in range(planes))
    return (struct.pack("<12I", *h) + struct.pack("<8Q", *metadata)
            + struct.pack("<4f", *floats) + payload)


class ColorCaptureContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "color_fixture.gscf"

    def load(self, raw):
        self.path.write_bytes(raw)
        return C.load_sample(self.path)

    def assert_error(self, raw, message):
        with self.assertRaises(ValueError) as caught:
            self.load(raw)
        self.assertEqual(str(caught.exception), message)

    def test_integer_formats_channel_order_normalization_and_ignored_alpha(self):
        cases = []
        for text in ("AB24", "XB24", "AR24", "XR24"):
            rgb = (17, 129, 241) if text[1] == "B" else (241, 129, 17)
            cases.append((text, bytes((17, 129, 241, 0)), rgb, 255))
        for text in ("AB30", "XB30", "AR30", "XR30"):
            rgb = (17, 529, 1001) if text[1] == "B" else (1001, 529, 17)
            cases.append((text, struct.pack("<I", 17 | (529 << 10) | (1001 << 20) | (3 << 30)),
                          rgb, 1023))
        for text in ("AB48", "XB48"):
            cases.append((text, struct.pack("<4H", 1, 12345, 65535, 0), (1, 12345, 65535), 65535))
        for text, pixel, rgb, scale in cases:
            with self.subTest(format=text):
                fmt = int.from_bytes(text.encode("ascii"), "little")
                decoded = C.decode_plane(pixel * 6, 3, 2, fmt, len(pixel))
                expected = np.broadcast_to(np.array(rgb, np.float32) / scale, (2, 3, 3))
                np.testing.assert_array_equal(decoded, expected)
                self.assertEqual(decoded.dtype, np.float32)
                self.assertTrue(decoded.flags.c_contiguous)
                self.assertEqual(C.fourcc(text), fmt)
                self.assertEqual(C.fourcc_name(fmt), text)

    def test_float_formats_preserve_hdr_signed_zero_and_nonfinite_rgb(self):
        for text, packing in (("AB4H", "<4e"), ("XB4H", "<4e"), ("AB8F", "<4f")):
            with self.subTest(format=text):
                pixel = struct.pack(packing, -0., -2., 12., float("nan"))
                pixel += struct.pack(packing, float("inf"), float("nan"), -float("inf"), 0.)
                fmt = int.from_bytes(text.encode("ascii"), "little")
                decoded = C.decode_plane(pixel, 2, 1, fmt, struct.calcsize(packing))
                expected = np.array([[[-0., -2., 12.], [np.inf, np.nan, -np.inf]]], np.float32)
                np.testing.assert_array_equal(decoded, expected)
                self.assertEqual(decoded.view(np.uint32)[0, 0, 0], 0x80000000)
                self.assertTrue(np.isfinite(decoded[0, 0]).all())
                self.assertEqual(decoded.dtype, np.float32)
                self.assertTrue(decoded.flags.c_contiguous)

    def test_versions_candidate_identity_and_exact_metadata(self):
        self.assertEqual(C.HEADER.size, 128)
        for version, kind in ((1, None), (2, "occlusion"), (3, "endpoint-trace")):
            with self.subTest(version=version):
                sample = self.load(capture(version))
                self.assertIs(sample["path"], self.path)
                self.assertEqual(sample["candidate_kind"], kind)
                self.assertEqual((sample["width"], sample["height"], sample["format"], sample["eotf"]),
                                 (2, 2, 0x34324241, 7))
                self.assertEqual((sample["sequence"], sample["anchor_id"], sample["reference_id"],
                                  sample["endpoint_id"]), (77, 100, 104, 108))
                self.assertEqual((sample["anchor_ns"], sample["reference_ns"], sample["endpoint_ns"]),
                                 (1000000000, 1250000000, 2000000000))
                self.assertEqual(sample["phase"], .25)
                strengths = [None] if version == 1 else [0., .5, 1.]
                self.assertEqual([s for s, _ in sample["candidates"]], strengths)
                for i, (_, plane) in enumerate(sample["candidates"]):
                    np.testing.assert_array_equal(plane[0, 0], np.array([10+i, 100+i, 200+i], np.float32) / 255)
                last = 1 if version == 1 else 3
                np.testing.assert_array_equal(sample["reference"][0, 0],
                                              np.array([10+last, 100+last, 200+last], np.float32) / 255)
        self.assertEqual(self.load(capture(header={9: 0}))["candidate_kind"], "occlusion")

    def test_ignored_fields_extra_flags_and_metadata_tolerance_stay_accepted(self):
        sample = self.load(capture(1, header={9: 0xffffffff}, floats=(.25, np.nan, np.inf, -2)))
        self.assertIsNone(sample["candidate_kind"])
        self.assertEqual(self.load(capture(2, header={9: 0xffffffff}))["candidate_kind"], "occlusion")
        sample = self.load(capture(header={6: 0xffffffff, 7: 0x80000003, 10: 42, 11: 43},
                                   meta={7: 123}, floats=(.2500005, 0, .5, 1)))
        self.assertEqual(sample["eotf"], 0xffffffff)
        self.assertGreater(sample["phase"], .25)
        pixel = struct.pack("<4e", -2, 12, float("nan"), 1)
        sample = self.load(capture(header={4: 0x48344241, 5: 8}, payload=pixel * 16))
        np.testing.assert_array_equal(sample["reference"][0, 0], np.array([-2, 12, np.nan], np.float32))

    def test_malformed_metadata_and_guard_precedence(self):
        cases = [
            (b"short", "truncated GSCF header"),
            (capture(header={0: 0, 7: 0}), "not a supported GSCF v1/v2/v3 capture"),
            (capture(header={1: 4}), "not a supported GSCF v1/v2/v3 capture"),
            (capture(header={7: 1, 2: 0}), "capture has no supported exact held-out reference layout"),
            (capture(header={8: 2}), "capture has no supported exact held-out reference layout"),
            (capture(header={2: 0}, floats=(np.nan, 0, .5, 1)),
             "capture has invalid dimensions, format, or bytes-per-pixel"),
            (capture(header={5: 8}), "capture has invalid dimensions, format, or bytes-per-pixel"),
            (capture(header={4: 0}), "capture has invalid dimensions, format, or bytes-per-pixel"),
            (capture(meta={2: 100}, floats=(np.nan, 0, .5, 1)), "capture has an invalid held-out phase"),
            (capture(meta={2: 100}, floats=(.25, 0, 0, 1)), "capture has non-monotonic frame IDs or timestamps"),
            (capture(meta={5: 1000000000}), "capture has non-monotonic frame IDs or timestamps"),
            (capture(floats=(.75, 0, 0, 1)), "capture phase disagrees with its timestamps"),
            (capture(header={9: 99}, floats=(.25, 0, 0, 1)), "capture has invalid paired candidate strengths"),
            (capture(header={9: 99})[:-1], "capture has unknown paired candidate kind 99"),
        ]
        for raw, message in cases:
            with self.subTest(message=message):
                self.assert_error(raw, message)
        for phase in (0., 1., np.inf, -np.inf):
            self.assert_error(capture(floats=(phase, 0, .5, 1)), "capture has an invalid held-out phase")
        for strengths in ((0, np.nan, 1), (-.1, .5, 1), (0, .5, 1.1), (0, 1, .5)):
            self.assert_error(capture(floats=(.25, *strengths)), "capture has invalid paired candidate strengths")

    def test_exact_file_length_and_direct_decode_errors(self):
        raw = capture()
        for malformed in (raw[:-1], raw + b"x"):
            self.assert_error(malformed, f"file is {len(malformed)} bytes, expected {len(raw)}")
        with self.assertRaisesRegex(ValueError, "^plane is 3 bytes, expected 4$"):
            C.decode_plane(b"abc", 1, 1, C.ABGR8888, 4)
        with self.assertRaisesRegex(ValueError, "^unsupported DRM format 'ZZZZ' \\(0x5a5a5a5a\\)$"):
            C.decode_plane(b"abcd", 1, 1, 0x5a5a5a5a, 4)
        with self.assertRaises(FileNotFoundError):
            C.load_sample(Path(self.directory.name) / "absent.gscf")

    def test_fresh_file_loaded_evaluator_shares_independent_capture_module(self):
        program = r'''
import importlib.util
from pathlib import Path
import sys

scripts = Path(sys.argv[1])
if sys.argv[2] == "leaf":
    sys.path.insert(0, str(scripts))
    import framegen_color_capture as capture
else:
    spec = importlib.util.spec_from_file_location("color_eval_contract", scripts / "framegen-color-eval.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    import framegen_color_capture as capture
    for name in ("load_sample", "decode_plane", "fourcc", "fourcc_name", "HEADER", "FORMAT_BPP",
                 "CANDIDATE_KINDS", "SUPPORTED_VERSIONS"):
        assert getattr(evaluator, name) is getattr(capture, name)
assert Path(capture.__file__).resolve() == scripts / "framegen_color_capture.py"
for name in ("torch", "lpips", "framegen_net", "framegen_net_train", "framegen_field_reference"):
    assert name not in sys.modules
'''
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        for first in ("leaf", "evaluator"):
            with self.subTest(first=first):
                result = subprocess.run([sys.executable, "-c", program, str(SCRIPTS), first],
                                        cwd=self.directory.name, env=env, capture_output=True,
                                        text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
