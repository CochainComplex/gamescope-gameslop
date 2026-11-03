#!/usr/bin/env python3
"""Prepare a pinned, private NR harness with exact device and raw-output checks.

Does not download, build, execute a runtime or change compositor/upstream files.
Patch anchors adapt RedDukeDev/dlss5-nr-amd-custom; see README.md and THIRD_PARTY.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REVISION = 'b5744cf03208f1442d63eeacc48eaf77a6bb4765'
HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('upstream source does not match the pinned NR adapter patch')
    return text.replace(old, new, 1)


def adapt_network(text):
    return replace_once(text,
        '            CUdevice device = 0;\n'
        '            if (!cu_ok(g.cu.cuDeviceGet(&device, 0), "cuDeviceGet", error)) return false;',
        '''            char wanted[32] = {};
            if (!GetEnvironmentVariableA("GSC_NR_PCI", wanted, sizeof wanted)) {
                error = "GSC_NR_PCI is required; this PoC never chooses device zero";
                return false;
            }
            auto pci_id = reinterpret_cast<CUresult (*)(char *, int, CUdevice)>(GetProcAddress(g.gate, "cuDeviceGetPCIBusId"));
            if (!pci_id) { error = "CUDA provider cannot establish PCI identity"; return false; }
            int count = 0, matches = 0;
            CUdevice device = 0;
            if (!cu_ok(g.cu.cuDeviceGetCount(&count), "cuDeviceGetCount", error)) return false;
            for (int ordinal = 0; ordinal < count; ++ordinal) {
                CUdevice candidate = 0;
                char pci[32] = {};
                if (!cu_ok(g.cu.cuDeviceGet(&candidate, ordinal), "cuDeviceGet", error) ||
                    !cu_ok(pci_id(pci, sizeof pci, candidate), "cuDeviceGetPCIBusId", error)) return false;
                if (!_stricmp(pci, wanted)) { device = candidate; ++matches; }
            }
            if (matches != 1) { error = "exactly one CUDA device must match the presenter PCI address"; return false; }
            log(2, "NR_POC_COMPUTE pci=%s matches=%d", wanted, matches);''')


def adapt_harness(text):
    text = replace_once(text, '#include <dxgi1_6.h>', '#include <dxgi1_6.h>\n#include "nr_poc_identity.hpp"')
    text = replace_once(text, 'if (hardware && SUCCEEDED(D3D12CreateDevice(',
                        'if (hardware && nr_poc::matches(factory, adapter) && SUCCEEDED(D3D12CreateDevice(')
    text = replace_once(text, 'bool linear = false, hdr = false, ratio = false, show_tracking = false, wait = false, bare = false;',
                        'bool linear = false, hdr = false, ratio = false, show_tracking = false, wait = false, bare = false, disabled = false;')
    text = replace_once(text, '        bare |= flag == L"--bare";',
                        '        bare |= flag == L"--bare";\n        disabled |= flag == L"--disabled";')
    text = replace_once(text, 'key == L"--wait" || key == L"--bare" ||',
                        'key == L"--disabled" || key == L"--wait" || key == L"--bare" ||')
    text = replace_once(text, '    settings.resolution_scale = scale;',
                        '    settings.enabled = disabled ? 0 : 1;\n'
                        '    settings.encoding = linear ? DLSS5NR_ENCODING_LINEAR : DLSS5NR_ENCODING_DISPLAY;\n'
                        '    printf("NR_POC_ENCODING linear=%d encoding=%d hdr=%d\\n", int(linear), settings.encoding, int(hdr));\n'
                        '    settings.resolution_scale = scale;')
    text = replace_once(text, '    for (int last = -1;;) {', '    for (int last = -1; !disabled;) {')
    text = replace_once(text, '        result.resize((size_t)width * height * 4);', '''        // Explicit diagnostic readback: inspect FP16 before PNG conversion can hide faults.
        if (!linear) { fprintf(stderr, "NR PoC requires --linear for raw diagnostics\\n"); exit(2); }
        FILE *raw = _wfopen((out + L".fp16").c_str(), L"wb");
        if (!raw) { fprintf(stderr, "could not create FP16 diagnostic\\n"); exit(2); }
        size_t invalid = 0, changed = 0, changed_any = 0;
        float max_abs = 0;
        for (UINT y = 0; y < height; ++y) {
            const uint16_t *row = reinterpret_cast<const uint16_t *>(mapped + fp.Offset + size_t(y)*fp.Footprint.RowPitch);
            const uint16_t *source = reinterpret_cast<const uint16_t *>(pixels.data() + size_t(y)*width*8);
            if (fwrite(row, 8, width, raw) != width) { fprintf(stderr, "FP16 write failed\\n"); exit(2); }
            for (UINT i = 0; i < width*4; ++i) {
                const float v = from_half(row[i]);
                if (!std::isfinite(v)) ++invalid;
                else max_abs = std::max(max_abs, std::abs(v));
                if (row[i] != source[i]) ++changed_any;
                if ((i & 3) != 3 && v != from_half(source[i])) ++changed;
            }
        }
        const bool closed = fclose(raw) == 0;
        printf("NR_POC_RAW width=%u height=%u invalid=%zu changed=%zu max_abs=%.9g disabled=%d changed_any=%zu\\n",
            width, height, invalid, changed, max_abs, int(disabled), changed_any);
        if (!closed || invalid || (disabled && changed_any)) { fprintf(stderr, "FP16 correctness gate failed\\n"); exit(2); }
        result.resize((size_t)width * height * 4);''')
    text = replace_once(text, '    if (!snapshot(out)) return 1;', '''    if (!disabled && (status.state != DLSS5NR_STATE_RUNNING || composed < evaluations || composed == 0))
        return fail("no fresh NR result: refusing to report passthrough as inference"), 1;
    if (!snapshot(out)) return 1;''')
    return text


def prepare(source, out, headers=Path('/usr/include')):
    source, out = source.resolve(strict=True), out.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('source and private output trees must not overlap')
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != REVISION or dirty:
        raise ValueError('requires a clean upstream 1.3.2 source checkout')
    if out.exists():
        raise FileExistsError('preserve prior work and use a new source output directory')
    originals = {}
    patches = {'runtime/core/network.cpp': adapt_network, 'runtime/tests/harness.cpp': adapt_harness,
               'runtime/CMakeLists.txt': lambda text: text + '\nconfigure_file("${CMAKE_CURRENT_SOURCE_DIR}/../nr-poc-source.json" '
                   '"${CMAKE_BINARY_DIR}/bin/nr-poc-source.json" COPYONLY)\n'}
    # Validate all replacements before creating an output tree.
    edited = {path: operation((source / path).read_text()) for path, operation in patches.items()}
    for header in ('vulkan/vulkan_core.h', 'vulkan/vk_platform.h', 'vk_video/vulkan_video_codec_h264std.h'):
        if not (headers / header).is_file():
            raise ValueError('Vulkan development headers are required for the DXVK identity query')
    shutil.copytree(source, out, ignore=shutil.ignore_patterns('.git', 'build', 'build-mingw', '__pycache__'))
    sdk = out / 'vulkan-headers'
    sdk.mkdir()
    for header in ('vulkan_core.h', 'vk_platform.h'):
        shutil.copy2(headers / 'vulkan' / header, sdk / header)
    shutil.copytree(headers / 'vk_video', sdk / 'vk_video')
    for path, text in edited.items():
        originals[path] = hashlib.sha256((source / path).read_bytes()).hexdigest()
        (out / path).write_text(text)
    shutil.copy2(HERE / 'presenter_identity.hpp', out / 'runtime/tests/nr_poc_identity.hpp')
    manifest = {'upstream_revision': REVISION, 'original_sha256': originals,
                'adapted_sha256': {path: hashlib.sha256((out / path).read_bytes()).hexdigest() for path in
                    (*patches, 'runtime/tests/nr_poc_identity.hpp')},
                'vulkan_headers_sha256': {str(p.relative_to(sdk)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in sdk.rglob('*') if p.is_file()},
                'scope': 'offscreen correctness only; no compositor changes or latency qualification'}
    (out / 'nr-poc-source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.out), indent=2))
