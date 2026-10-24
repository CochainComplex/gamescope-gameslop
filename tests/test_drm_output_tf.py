#!/usr/bin/env python3
"""Execute drm_prepare's output-TF policy with observed reads and writes.

This characterizes CPU decisions, not KMS requests, commits or GPU completion.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def fragment(source, begin, end):
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError('ambiguous/missing production fragment: ' + begin)
    start = source.index(begin)
    stop = source.index(end, start)
    return start, source[start:stop]


def materialize(source_path, directory):
    root = Path(__file__).resolve().parents[1]
    source = source_path.read_text()
    function_start, function = fragment(source,
        'int drm_prepare( struct drm_t *drm, bool async, const struct FrameInfo_t *frameInfo )',
        '\nbool drm_poll_state(')
    declaration = '\tconst bool bSinglePlane' if '\tconst bool bSinglePlane' in function else '\tbool bSinglePlane'
    start, body = fragment(function, declaration + ' = frameInfo->layers.count() < 2 && cv_drm_single_plane_optimizations;',
                           '\n\tuint32_t flags = DRM_MODE_ATOMIC_NONBLOCK;')
    start += function_start
    line = source.count('\n', 0, start) + 1
    (directory / 'policy.inc').write_text(f'#line {line} "{source_path}"\n' + body)
    _, enum = fragment((root / 'src/drm_include.h').read_text(),
                       'enum amdgpu_transfer_function {', '\nenum drm_panel_orientation {')
    (directory / 'transfer_function.inc').write_text(enum)
    (directory / 'fixture.cpp').write_text(FIXTURE)


def execute(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    materialize(source, directory)
    commands = []
    for name, flags in [('normal', ['-O0']), ('optimized', ['-O3', '-ffast-math'])]:
        binary = directory / name
        command = shlex.split(os.environ.get('CXX', 'c++')) + [
            '-std=c++20', '-Wall', '-Wextra', '-Werror', '-UNDEBUG', *flags,
            '-I' + str(directory), str(directory / 'fixture.cpp'), '-o', str(binary),
        ]
        subprocess.run(command, check=True, timeout=60)
        subprocess.run([str(binary)], check=True, timeout=30)
        commands.append(command)
    (directory / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')


FIXTURE = r'''
#include <cassert>
#include <cstdio>
#include <vector>
#include "transfer_function.inc"
static std::vector<unsigned> reads;
struct Observed {
    unsigned tag;
    bool value=false;
    operator bool() const { reads.push_back(tag); return value; }
};
static Observed cv_drm_single_plane_optimizations{2};
static Observed cv_drm_debug_disable_output_tf{5};
static Observed g_bOutputHDREnabled{6};
struct TransferFunction {
    amdgpu_transfer_function value=AMDGPU_TRANSFER_FUNCTION_DEFAULT;
    unsigned writes=0;
    void operator=(amdgpu_transfer_function v) { reads.push_back(7); value=v; ++writes; }
};
struct drm_t {
    struct { TransferFunction output_tf; } pending;
} g_DRM;
static bool supported;
static bool drm_supports_color_mgmt(drm_t *owner) {
    assert(owner==&g_DRM); reads.push_back(3); return supported;
}
struct Layers {
    unsigned size;
    unsigned count() const { reads.push_back(1); return size; }
};
struct FrameInfo_t {
    Layers layers;
    Observed applyOutputColorMgmt{4};
};
static void evaluate(drm_t *drm, const FrameInfo_t *frameInfo) {
#include "policy.inc"
}
int main() {
    unsigned cases=0;
    for (unsigned layers : {0u,1u,2u,3u})
        for (bool single : {false,true}) for (bool capability : {false,true})
        for (bool apply : {false,true}) for (bool disabled : {false,true})
        for (bool hdr : {false,true}) for (auto prior : {
                AMDGPU_TRANSFER_FUNCTION_DEFAULT, AMDGPU_TRANSFER_FUNCTION_SRGB_EOTF,
                AMDGPU_TRANSFER_FUNCTION_PQ_EOTF}) {
            cv_drm_single_plane_optimizations.value=single;
            cv_drm_debug_disable_output_tf.value=disabled;
            g_bOutputHDREnabled.value=hdr; supported=capability;
            FrameInfo_t frame{{layers},{4,apply}};
            // The policy observes capabilities on g_DRM, and writes through its
            // drm parameter. Distinct objects catch an accidental substitution.
            drm_t state; state.pending.output_tf.value=prior;
            g_DRM.pending.output_tf.value=AMDGPU_TRANSFER_FUNCTION_IDENTITY;
            reads.clear(); evaluate(&state,&frame);
            const bool enabled=capability && apply && !disabled && !(layers<2 && single);
            const auto expected=enabled ? (hdr ? AMDGPU_TRANSFER_FUNCTION_PQ_EOTF :
                AMDGPU_TRANSFER_FUNCTION_SRGB_EOTF) : AMDGPU_TRANSFER_FUNCTION_DEFAULT;
            assert(state.pending.output_tf.value==expected && state.pending.output_tf.writes==1);
            assert(g_DRM.pending.output_tf.value==AMDGPU_TRANSFER_FUNCTION_IDENTITY && !g_DRM.pending.output_tf.writes);
            std::vector<unsigned> order{1};
            if (layers<2) order.push_back(2);
            order.push_back(3);
            if (capability) {
                order.push_back(4);
                if (apply) order.push_back(5);
            }
            if (enabled) order.push_back(6);
            order.push_back(7);
            assert(reads==order); ++cases;
        }
    std::printf("DRM output-TF characterization: %u cases PASS\n",cases);
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = (args.source or Path(__file__).resolve().parents[1] /
              'src/Backends/DRMBackend.cpp').resolve()
    if args.output:
        execute(source, args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='gamescope-drm-output-tf-') as name:
            execute(source, Path(name))


if __name__ == '__main__':
    main()
