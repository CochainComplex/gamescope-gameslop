#!/usr/bin/env python3
"""Execute the compositor's real-admission owner against controlled CPU state.

No compositor, Wayland server or GPU is started. The primary coordinates this
compiler fixture with all timing campaigns through the shared build/GPU lock.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def definition(source, name, parameter=None):
    # Keep offsets while excluding comments/literals from brace matching.
    # A numeric separator (100'000) is not the start of a character literal.
    clean = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|(?<!\w)(?:u8|[uUL])?\'(?:\\.|[^\'\\\n])*\'',
                   lambda match: ' ' * len(match[0]), source, flags=re.S)
    pattern = r'(?:static\s+)?(?:bool|uint64_t|void)\s+' + re.escape(name) + r'\s*\([^;{}]*\)\s*\{'
    matches = [match for match in re.finditer(pattern, clean)
               if parameter is None or parameter in match[0]]
    if len(matches) != 1:
        raise ValueError('ambiguous/missing production definition: ' + name)
    match = matches[0]
    brace = clean.index('{', match.start())
    depth = 0
    for end in range(brace, len(clean)):
        depth += (clean[end] == '{') - (clean[end] == '}')
        if depth == 0:
            line = source.count('\n', 0, match.start()) + 1
            return line, source[match.start():end+1] + '\n'
    raise ValueError('unterminated production definition: ' + name)


def materialize(root, source_path, output):
    source = source_path.read_text()
    provenance = {}
    for part, name, parameter in (
        ('limit', 'steamcompmgr_window_should_limit_fps', None),
        ('gate', 'steamcompmgr_should_vblank_window', 'bool bShouldLimitFPS'),
        ('note', 'steamcompmgr_note_vblank_pass', None),
        ('owner', 'steamcompmgr_framegen_next_real_eligible_vblank', None),
    ):
        line, body = definition(source, name, parameter)
        (output/(part+'.inc')).write_text(f'#line {line} "{source_path}"\n'+body)
        provenance[part] = {'source': str(source_path), 'line': line,
                            'sha256': hashlib.sha256(body.encode()).hexdigest()}
    declarations = []
    for name in ('s_ulVBlankIdx', 's_ulPrevVBlankIdx', 's_ulLastPassVBlank',
                 's_ulLastPassRefreshCycle'):
        matches = re.findall(r'^static uint64_t '+name+r'\s*=\s*0\s*;', source, re.M)
        if len(matches) != 1:
            raise ValueError('ambiguous/missing owner state: ' + name)
        declarations.append(matches[0])
    for pattern in (
        r'^static constexpr uint64_t k_ulVRRScheduleFudge\s*=\s*[^;]+;',
        r'^static std::optional<uint64_t> s_oLowestFPSLimitScheduleVRR\s*;',
    ):
        matches = re.findall(pattern, source, re.M)
        if len(matches) != 1:
            raise ValueError('ambiguous/missing gate state: ' + pattern)
        declarations.append(matches[0])
    (output/'state.inc').write_text('\n'.join(declarations)+'\n')
    return provenance


def execute(root, source, output):
    output.mkdir(parents=True, exist_ok=True)
    provenance = materialize(root, source, output)
    fixture = root/'tests/framegen_real_admission.cpp'
    report = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
              'headers': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (root/'src/Utils/VBlankSchedule.h', root/'src/refresh_rate.h')},
              'seams': provenance, 'runs': [],
              'limits': ['CPU owner state and scheduling policy, with controlled external state.',
                         'No Wayland threading/lifetime, GPU execution, native display, physical latency or quality evidence.']}
    for name, flags in (('normal', ['-O0']), ('fastmath', ['-O3', '-ffast-math'])):
        binary = output/name
        command = shlex.split(os.environ.get('CXX', 'c++')) + [
            '-std=c++20', '-Wall', '-Wextra', '-Werror', '-Wno-missing-field-initializers',
            '-UNDEBUG', *flags, '-I', str(root/'src'), '-I', str(output),
            str(fixture), '-o', str(binary)]
        compiled = subprocess.run(command, text=True, capture_output=True, timeout=60)
        (output/(name+'-compile.log')).write_text(compiled.stdout+compiled.stderr)
        record = {'name': name, 'command': command, 'compile_returncode': compiled.returncode}
        if compiled.returncode == 0:
            tested = subprocess.run([str(binary)], text=True, capture_output=True, timeout=15)
            (output/(name+'-run.log')).write_text(tested.stdout+tested.stderr)
            record['test_returncode'] = tested.returncode
            print(tested.stdout+tested.stderr, end='')
        else:
            print(compiled.stdout+compiled.stderr, end='')
        report['runs'].append(record)
    report['passed'] = all(run['compile_returncode']==0 and run.get('test_returncode')==0
                           for run in report['runs'])
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['passed'] else 1


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'src/steamcompmgr.cpp')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output:
        return execute(root, args.source.resolve(), args.output.resolve())
    with tempfile.TemporaryDirectory(prefix='framegen-real-admission-') as name:
        return execute(root, args.source.resolve(), Path(name))


if __name__ == '__main__':
    raise SystemExit(main())
