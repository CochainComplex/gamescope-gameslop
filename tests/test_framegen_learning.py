#!/usr/bin/env python3
"""Execute production learning transactions/recording against CPU effect fakes."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def definition(source, signature):
    # Locate balanced bodies without interpreting braces in comments/literals.
    clean = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                   lambda match: ' ' * len(match[0]), source, flags=re.S)
    begin = source.index(signature)
    brace = clean.index('{', begin)
    depth = 0
    for end in range(brace, len(clean)):
        depth += (clean[end] == '{') - (clean[end] == '}')
        if depth == 0:
            return source[begin:end+1] + (';' if source[end+1:end+2] == ';' else '') + '\n'
    raise ValueError(f'unterminated production definition: {signature}')


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / 'src/rendervulkan.cpp').read_text()
    header = (root / 'src/rendervulkan.hpp').read_text()
    signatures = {
        'busy': 'static bool framegen_work_completed()',
        'drop': 'static void framegen_net_drop_profile_readback()',
        'train': 'static bool framegen_record_net_train(',
        'upload': 'static void framegen_record_net_upload_barrier(',
        'pair': 'static bool framegen_record_net_train_pair(',
        'transaction': 'struct FramegenLearningRecording\n',
        'idle': 'void vulkan_framegen_train_idle()',
        'consume': 'static void framegen_net_profile_consume()\n{',
        'store': 'static void framegen_net_store_put(',
        'key': 'static void framegen_net_activate_key(',
        'write': 'static void framegen_net_profile_write_file(',
        'snapshot': 'static bool framegen_net_profile_snapshot(',
        'join': 'static void framegen_net_profile_join_writer()\n{',
        'flush': 'static void framegen_net_profile_flush()\n{',
        'pins': 'static void framegen_release_completed_read_pins()',
        'scene': 'void vulkan_framegen_invalidate_history(',
        'reset': 'void vulkan_framegen_reset(',
    }
    with tempfile.TemporaryDirectory(prefix='framegen-learning-') as name:
        directory = Path(name)
        for part, signature in signatures.items():
            (directory / f'learning_{part}.inc').write_text(definition(source, signature))
        constants = ('k_uFramegenNetTrainTiles', 'k_flFramegenNetEmaAlpha',
                     'k_flFramegenNetDecay', 'k_uFramegenNetProfileInterval')
        (directory / 'learning_constants.inc').write_text('\n'.join(
            next(line for line in source.splitlines() if line.startswith('static constexpr') and name in line)
            for name in constants) + '\n' + definition(header, 'static inline uint32_t div_roundup('))
        for variant, flags in (('normal', []), ('fastmath', ['-O3', '-ffast-math'])):
            binary = directory / variant
            subprocess.run(shlex.split(os.environ.get('CXX', 'c++')) + [
                '-std=c++20', '-Wall', '-Wextra', '-Werror', '-Wno-missing-field-initializers', '-pthread', *flags,
                '-DFRAMEGEN_PRODUCTION_FIXTURE', '-I', str(root / 'src'),
                '-I', str(directory), str(root / 'tests/framegen_learning.cpp'),
                str(root / 'src/framegen/atomic_file.cpp'), '-o', str(binary)], check=True, timeout=60)
            subprocess.run([str(binary), str(directory / f'{variant}-profile.bin')], check=True, timeout=15)


if __name__ == '__main__':
    main()
