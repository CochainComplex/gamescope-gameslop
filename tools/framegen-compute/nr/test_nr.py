#!/usr/bin/env python3
"""CPU contracts for the isolated NR probe; never loads Wine or a GPU SDK."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
SOURCE = None


def load(name):
    spec = importlib.util.spec_from_file_location('nr_test_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('run')
prepare = load('prepare')
PCI = '0000:c7:00.0'
UUID = '00000000c70000000000000000000000'
SIZE = (17, 19)


def evidence(*, disabled=False, changed=1):
    stdout = (f'NR_POC_PRESENTER uuid={UUID} pci={PCI} vendor=1002 device=150e\n'
              'NR_POC_ENCODING linear=1 encoding=1 hdr=0\n'
              f'NR_POC_RAW width=17 height=19 invalid=0 changed={changed} max_abs=0.5 disabled={int(disabled)} changed_any={changed}\n'
              f'NR_POC_COMPUTE pci={PCI} matches=1\n'
              'frames 4, results composed 3, network 16x16, last evaluation 12.5 ms\n')
    stderr = 'hipImportExternalMemory returned 0\n' * 2
    raw = struct.pack('<e', 0.5) * (17 * 19 * 4)
    return stdout, stderr, raw


def prerequisite_sentinels(root):
    """Satisfy file preflight only; none of these files is executable."""
    for relative in ['proton/files/bin/wine', 'proton/files/bin/wineserver',
                     'build/dlss5nr_harness.exe', 'build/dlss5nr_runtime.dll',
                     'build/zluda/nvcuda.dll', 'build/zluda/nvngx.dll',
                     'runtime/nvngx_dlssnr.dll', 'runtime/zluda/zluda_real.dll',
                     'runtime/zluda/nvapi64.dll', 'runtime/cache/ComputeCache/zluda2.db',
                     'runtime/cache/ComputeCache/coverage.txt', 'linux/libdlss5nr_hip_bridge.so',
                     'linux/amdhip64_7.dll', 'hip.so', 'build/nr-poc-source.json']:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'CPU-only prerequisite sentinel')
    image = root / 'input.png'
    image.write_bytes(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 17, 19))
    return argparse.Namespace(presenter_pci=PCI, presenter_uuid=UUID, image=image,
        proton=root / 'proton', runtime=root / 'runtime', build=root / 'build',
        linux=root / 'linux', hip_library=root / 'hip.so', out=root / 'output',
        timeout=1, disabled=True, baseline=None)


class Contracts(unittest.TestCase):
    def validate(self, values, disabled=False):
        return runner.validate_evidence(*values, SIZE, PCI, UUID, disabled)

    def test_exact_identity(self):
        self.assertEqual(runner.identity(PCI, UUID), (PCI, UUID))
        self.assertEqual(runner.identity('ffff:ff:1f.7', 'f' * 32), ('ffff:ff:1f.7', 'f' * 32))
        for pci, uuid in [(PCI.upper(), UUID), ('0000:c7:20.0', UUID),
                          ('0000:c7:00.8', UUID), ('c7:00.0', UUID),
                          (PCI + '\n', UUID), (PCI, '0' * 32),
                          (PCI, UUID[:-1]), (PCI, 'F' * 32), (PCI, UUID + '\n')]:
            with self.subTest(pci=pci, uuid=uuid), self.assertRaises(ValueError):
                runner.identity(pci, uuid)

    def test_real_evidence_and_raw_digest(self):
        import hashlib
        values = evidence()
        result = self.validate(values)
        self.assertEqual(result['results_composed'], 3)
        self.assertEqual(result['network_size'], [16, 16])
        self.assertEqual(result['successful_external_memory_imports'], 2)
        self.assertEqual(result['raw_sha256'], hashlib.sha256(values[2]).hexdigest())

    def test_fp16_exact_extent_and_nonfinite(self):
        stdout, stderr, raw = evidence()
        for invalid in [raw[:-1], raw[:-8], raw + b'\x00\x00']:
            with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                self.validate((stdout, stderr, invalid))
        for bits in [0x7c00, 0xfc00, 0x7e00, 0x7c01]:
            for channel in [0, 17 * 19 * 4 - 1]:
                corrupted = bytearray(raw)
                struct.pack_into('<H', corrupted, channel * 2, bits)
                with self.subTest(bits=bits, channel=channel), self.assertRaises(ValueError):
                    self.validate((stdout, stderr, corrupted))

    def test_finite_half_extrema_preserved(self):
        stdout, stderr, raw = evidence()
        raw = bytearray(raw)
        for i, bits in enumerate([0, 0x8000, 1, 0x8001, 0x7bff, 0xfbff]):
            struct.pack_into('<H', raw, i * 2, bits)
        self.validate((stdout.replace('max_abs=0.5', 'max_abs=65504'), stderr, raw))

    def test_running_or_changed_png_is_not_inference(self):
        stdout, stderr, raw = evidence()
        for fake in ['RUNNING\noutput.png changed\n', stdout.split('NR_POC_RAW')[0],
                     stdout.replace('results composed 3', 'results composed 0'),
                     stdout.replace('network 16x16', 'network 0x16'),
                     stdout.replace('changed=1 ', 'changed=0 ')]:
            with self.subTest(stdout=fake), self.assertRaises(ValueError):
                self.validate((fake, stderr, raw))
        for imports in ['', 'hipImportExternalMemory returned 0\n',
                        stderr + 'hipImportExternalMemory returned -1\n']:
            with self.subTest(imports=imports), self.assertRaises(ValueError):
                self.validate((stdout, imports, raw))

    def test_wrong_presenter_or_compute_refused(self):
        stdout, stderr, raw = evidence()
        for old, new in [(f'uuid={UUID}', 'uuid=' + 'f' * 32),
                         ('NR_POC_COMPUTE pci=' + PCI, 'NR_POC_COMPUTE pci=0000:01:00.0'),
                         ('matches=1', 'matches=0'), ('matches=1', 'matches=2'),
                         ('matches=1', 'matches=10')]:
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                self.validate((stdout.replace(old, new), stderr, raw))

    def test_raw_diagnostic_consistency(self):
        stdout, stderr, raw = evidence()
        for old, new in [('width=17', 'width=19'), ('invalid=0', 'invalid=1'),
                         ('disabled=0', 'disabled=1'), ('changed=1 ', 'changed=970 '),
                         ('changed_any=1', 'changed_any=0'), ('changed_any=1', 'changed_any=1293'),
                         ('max_abs=0.5', 'max_abs=nan'), ('max_abs=0.5', 'max_abs=inf'),
                         ('max_abs=0.5', 'max_abs=-1'), ('max_abs=0.5', 'max_abs=0.6'),
                         (' changed_any=1', '')]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.validate((stdout.replace(old, new), stderr, raw))

    def test_explicit_linear_sdr_domain_required(self):
        stdout, stderr, raw = evidence()
        for replacement in ['', 'NR_POC_ENCODING linear=0 encoding=1 hdr=0',
                            'NR_POC_ENCODING linear=1 encoding=0 hdr=0',
                            'NR_POC_ENCODING linear=1 encoding=1 hdr=1']:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                self.validate((stdout.replace('NR_POC_ENCODING linear=1 encoding=1 hdr=0', replacement), stderr, raw))

    def test_independent_baseline_numeric_rgb_changes(self):
        baseline = struct.pack('<e', .5) * (17 * 19 * 4)
        raw = bytearray(baseline)
        struct.pack_into('<e', raw, 0, 1)
        struct.pack_into('<e', raw, (17 * 19 * 4 - 2) * 2, -.25)
        struct.pack_into('<e', raw, 3 * 2, 0)  # Alpha is not an RGB improvement.
        self.assertEqual(runner.compare_baseline(raw, baseline, SIZE, 2),
                         {'independent_changed_rgb_channels': 2, 'max_rgb_delta': .75})
        for reported in [0, 1, 3, 1292]:
            with self.subTest(reported=reported), self.assertRaises(ValueError):
                runner.compare_baseline(raw, baseline, SIZE, reported)

    def test_baseline_signed_zero_and_alpha_only_are_not_changes(self):
        baseline = b'\x00\x00' * (17 * 19 * 4)
        for channel, bits in [(0, 0x8000), (3, 0x3c00)]:
            raw = bytearray(baseline)
            struct.pack_into('<H', raw, channel * 2, bits)
            with self.subTest(channel=channel), self.assertRaises(ValueError):
                runner.compare_baseline(raw, baseline, SIZE, 1)
        with self.assertRaises(ValueError):
            runner.compare_baseline(baseline, baseline, SIZE, 0)

    def test_baseline_exact_extent_and_finite_alpha_in_both_images(self):
        baseline = struct.pack('<e', .5) * (17 * 19 * 4)
        raw = bytearray(baseline)
        struct.pack_into('<e', raw, 0, 1)
        for output, reference in [(raw[:-2], baseline), (raw, baseline[:-2]),
                                  (raw + b'\0\0', baseline + b'\0\0')]:
            with self.subTest(lengths=(len(output), len(reference))), self.assertRaises(ValueError):
                runner.compare_baseline(output, reference, SIZE, 1)
        for bits in [0x7c00, 0xfc00, 0x7e00, 0x7c01]:
            for corrupt_output in [False, True]:
                output, reference = bytearray(raw), bytearray(baseline)
                struct.pack_into('<H', output if corrupt_output else reference,
                                 (17 * 19 * 4 - 1) * 2, bits)
                with self.subTest(bits=bits, output=corrupt_output), self.assertRaises(ValueError):
                    runner.compare_baseline(output, reference, SIZE, 1)

    def test_disabled_baseline_is_distinct_bit_exact_status(self):
        stdout, _, raw = evidence(disabled=True, changed=0)
        # Bit equality is enforced by the harness against its uploaded bytes,
        # including alpha; this parser does not independently decode the PNG.
        stdout = stdout.split('NR_POC_COMPUTE')[0]
        result = self.validate((stdout, '', raw), disabled=True)
        self.assertIs(result['disabled_transport_bit_exact_harness_asserted'], True)
        self.assertNotIn('results_composed', result)
        with self.assertRaises(ValueError):
            self.validate((stdout.replace('changed=0', 'changed=1'), '', raw), disabled=True)
        with self.assertRaises(ValueError):
            self.validate((stdout.replace('changed_any=0', 'changed_any=1'), '', raw), disabled=True)
        with self.assertRaises(ValueError):
            self.validate((stdout, '', raw), disabled=False)

    def test_source_replacement_refuses_missing_or_duplicate_anchor(self):
        self.assertEqual(prepare.replace_once('before ANCHOR after', 'ANCHOR', 'NEW'), 'before NEW after')
        for text in ['before after', 'ANCHOR ANCHOR']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                prepare.replace_once(text, 'ANCHOR', 'NEW')

    def test_unknown_revision_and_dirty_tree_never_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'
            source.mkdir()
            out = Path(directory) / 'out'
            for revision, dirty in [('f' * 40, ''), (prepare.REVISION, ' M runtime/core/network.cpp\n')]:
                with self.subTest(revision=revision, dirty=dirty):
                    with mock.patch.object(prepare.subprocess, 'check_output', side_effect=[revision, dirty]):
                        with self.assertRaises(ValueError):
                            prepare.prepare(source, out)
                    self.assertFalse(out.exists())

    def test_reused_output_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            out = root / 'out'
            out.mkdir()
            marker = out / 'existing'
            marker.write_bytes(b'preserve')
            with mock.patch.object(prepare.subprocess, 'check_output', side_effect=[prepare.REVISION, '']):
                with self.assertRaises(FileExistsError):
                    prepare.prepare(source, out)
            self.assertEqual(marker.read_bytes(), b'preserve')
            self.assertEqual(list(out.iterdir()), [marker])

    def test_preparation_overlap_refused_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            marker = source / 'preserve'
            marker.write_bytes(b'original')
            for out in [source, source / 'nested', root]:
                with self.subTest(out=out), self.assertRaisesRegex(ValueError, 'overlap'):
                    prepare.prepare(source, out)
            self.assertEqual(marker.read_bytes(), b'original')
            self.assertFalse((source / 'nested').exists())

    def test_run_output_overlap_refused_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = prerequisite_sentinels(root)
            for out in [root / 'runtime', root / 'runtime/nested', root]:
                args.out = out
                with self.subTest(out=out), self.assertRaisesRegex(ValueError, 'overlap'):
                    runner.run(args)
            self.assertFalse((root / 'runtime/nested').exists())
            self.assertFalse((root / 'result.json').exists())

    def test_enabled_run_requires_matched_untampered_disabled_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = prerequisite_sentinels(root)
            args.disabled = False
            with self.assertRaisesRegex(ValueError, 'baseline'):
                runner.run(args)
            args.baseline = root / 'baseline'
            args.baseline.mkdir()
            raw = evidence()[2]
            raw_path = args.baseline / 'output.png.fp16'
            raw_path.write_bytes(raw)
            valid = {'status': 'passed', 'disabled': True, 'pci': PCI, 'vulkan_uuid': UUID,
                     'input_sha256': runner.digest(args.image),
                     'source_manifest_sha256': runner.digest(args.build / 'nr-poc-source.json'),
                     'raw_sha256': runner.digest(raw_path)}
            for field, wrong in [('status', 'failed'), ('disabled', False), ('disabled', 1),
                                 ('pci', '0000:01:00.0'), ('vulkan_uuid', 'f' * 32),
                                 ('input_sha256', '0' * 64), ('source_manifest_sha256', '0' * 64),
                                 ('raw_sha256', '0' * 64)]:
                receipt = dict(valid)
                receipt[field] = wrong
                (args.baseline / 'result.json').write_text(json.dumps(receipt))
                with self.subTest(field=field, wrong=wrong), self.assertRaisesRegex(ValueError, 'baseline'):
                    runner.run(args)
                self.assertFalse(args.out.exists())

    def test_pinned_real_sources_adapt_exactly_once(self):
        if SOURCE is None:
            self.skipTest('supply --source for pinned upstream source-anchor coverage')
        for relative, adapt in [('runtime/core/network.cpp', prepare.adapt_network),
                                ('runtime/tests/harness.cpp', prepare.adapt_harness)]:
            original = (SOURCE / relative).read_text()
            patched = adapt(original)
            self.assertNotEqual(patched, original)
            with self.assertRaises(ValueError):
                adapt(patched)
            with self.assertRaises(ValueError):
                adapt(original + '\n' + original)

    @unittest.skipUnless(os.name == 'posix' and Path('/proc').exists(), 'Linux process-group cleanup contract')
    def test_owned_process_group_cleanup_on_timeout_and_nonzero_exit(self):
        for timeout_case in [True, False]:
            with self.subTest(timeout=timeout_case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                marker = root / 'owned-child.pid'
                descendant = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'
                script = ('import subprocess,sys,time,pathlib; '
                          f'p=subprocess.Popen([sys.executable,"-c",{descendant!r}]); '
                          f'pathlib.Path({str(marker)!r}).write_text(str(p.pid)); '
                          + ('time.sleep(60)' if timeout_case else 'time.sleep(.1);sys.exit(7)'))
                pid = None
                try:
                    if timeout_case:
                        with self.assertRaises(subprocess.TimeoutExpired):
                            runner.command([sys.executable, '-c', script], os.environ.copy(), root, 1)
                    else:
                        self.assertEqual(runner.command([sys.executable, '-c', script], os.environ.copy(), root, 3), 7)
                    self.assertTrue(marker.exists(), 'owned descendant must actually have started')
                    pid = int(marker.read_text())
                    deadline = time.monotonic() + 2
                    while time.monotonic() < deadline:
                        status = Path(f'/proc/{pid}/stat')
                        if not status.exists() or status.read_text().split(') ', 1)[1].startswith('Z'):
                            break
                        time.sleep(.01)
                    else:
                        self.fail('owned descendant survived helper cleanup')
                finally:
                    if pid is None and marker.exists():
                        pid = int(marker.read_text())
                    if pid is not None:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source.resolve(strict=True) if args.source else None
    unittest.main(argv=[sys.argv[0], *remaining])
