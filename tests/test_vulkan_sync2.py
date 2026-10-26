#!/usr/bin/env python3
"""Exercise production image-visibility tracking with CPU barrier recording.

Real Vulkan headers supply constants; no Vulkan device or library is needed.
External queue ownership and GPU execution are outside this contract.
"""
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory


def between(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('ambiguous/missing production seam: ' + start)
    begin = source.index(start)
    finish = source.index(end, begin)
    return source[begin:finish]


def main():
    root = Path(__file__).resolve().parents[1]
    header = (root / 'src/rendervulkan.hpp').read_text()
    source = (root / 'src/rendervulkan.cpp').read_text()
    extracts = {
        'texture_state': between(header, 'struct TextureState\n', '\nclass CVulkanCmdBuffer\n'),
        'image_types': between(header, '\tstruct ImageUse\n', '\n\tstd::pair<TextureState *, bool> trackTexture'),
    }
    seams = (
        ('void CVulkanCmdBuffer::addImageUse(', '\nvoid CVulkanCmdBuffer::emitExternalAcquireBarriers('),
        ('void CVulkanCmdBuffer::emitSync2Barriers()', '\nvoid CVulkanCmdBuffer::emitPreDispatchBarriers('),
        ('void CVulkanCmdBuffer::discardImage(', '\nvoid CVulkanCmdBuffer::markDirty('),
        ('void CVulkanCmdBuffer::markDirty(', '\nstd::pair<TextureState *, bool> CVulkanCmdBuffer::trackTexture('),
        ('std::pair<TextureState *, bool> CVulkanCmdBuffer::trackTexture(', '\nTextureState *CVulkanCmdBuffer::findTextureState('),
        ('TextureState *CVulkanCmdBuffer::findTextureState(', '\nvoid CVulkanCmdBuffer::insertBarrier('),
        ('void CVulkanCmdBuffer::insertBarrier(', '\nvoid CVulkanCmdBuffer::hostReadBarrier('),
    )
    extracts['owner'] = '\n'.join(between(source, start, end) for start, end in seams)
    with TemporaryDirectory(prefix='vulkan-sync2-') as directory:
        output = Path(directory)
        for name, body in extracts.items():
            (output / (name + '.inc')).write_text(body)
        for name, flags in (('normal', ['-O0']), ('fastmath', ['-O3', '-ffast-math'])):
            binary = output / name
            command = shlex.split(os.environ.get('CXX', 'c++')) + [
                '-std=c++20', '-Wall', '-Wextra', '-Werror',
                '-Wno-missing-field-initializers', '-UNDEBUG', *flags,
                '-I', str(output), str(root / 'tests/test_vulkan_sync2.cpp'),
                '-o', str(binary),
            ]
            subprocess.run(command, check=True, timeout=60)
            subprocess.run([str(binary)], check=True, timeout=10)
            print(name + ': 12 image-visibility contracts passed', flush=True)


if __name__ == '__main__':
    main()
