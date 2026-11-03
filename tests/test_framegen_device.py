#!/usr/bin/env python3
"""Exercise production completed-timing collection with CPU Vulkan-call fakes.

No Vulkan device is created. This protects host ordering and accounting, not
GPU synchronization, cross-device ownership or physical presentation timing.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from test_framegen_learning import definition


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=Path, default=root / 'src/framegen/device.cpp')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.device.read_text()
    header = (root / 'src/rendervulkan.hpp').read_text()
    methods = ('framegenResetTrainingCost', 'framegenResetRungCosts', 'framegenGarbageCollect')
    helper = 'framegenReadCompletedTiming'
    if f'void CVulkanDevice::{helper}(' in source:
        methods = (helper, *methods)
    owner = '\n'.join(definition(source, f'void CVulkanDevice::{name}(') for name in methods)
    association = definition(header, 'struct FramegenQueryAssoc_t ')
    cache = definition(header, 'inline void cache_timeline_completion(')
    with tempfile.TemporaryDirectory(prefix='framegen-device-') as temporary:
        output = args.output or Path(temporary)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'owner.inc').write_text(owner)
        (output / 'association.inc').write_text(association)
        (output / 'cache.inc').write_text(cache)
        (output / 'helper.inc').write_text(
            next(line.strip() for line in header.splitlines() if f'void {helper}(' in line) + '\n'
            if helper in methods else '')
        for name, flags in (('normal', ['-O0']), ('fastmath', ['-O3', '-ffast-math'])):
            binary = output / name
            command = shlex.split(os.environ.get('CXX', 'c++')) + [
                '-std=c++20', '-Wall', '-Wextra', '-Werror', '-UNDEBUG', *flags,
                '-I', str(output), str(root / 'tests/framegen_device.cpp'), '-o', str(binary)]
            subprocess.run(command, check=True, timeout=60)
            subprocess.run([str(binary)], check=True, timeout=15)
            print(name + ': completed-timing contracts passed', flush=True)


if __name__ == '__main__':
    main()
