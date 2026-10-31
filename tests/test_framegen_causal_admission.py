#!/usr/bin/env python3
"""Execute production causal admission, recovery and correction against CPU fakes.

Run only under the primary's coordinated build lock. No GPU/device is created.
--renderer selects baseline/candidate source; headers stay the live root headers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name('framegen_causal_admission.cpp')


def between(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('ambiguous/missing production seam: ' + start)
    begin, finish = source.index(start), source.index(end)
    if finish <= begin:
        raise ValueError('reversed production seam: ' + start)
    return begin, source[begin:finish]


def materialize(renderer, output):
    source = renderer.read_text()
    seams = {
        'history_gap': ('static constexpr uint64_t k_ulFramegenMaxRealFrameGapNs = ',
                        '// Dedicated-queue idle refill may extrapolate beyond the originally expected'),
        'config': ('static FramegenEffective_t framegen_effective_config( uint32_t nDegradeSteps )',
                   'static size_t framegen_bidir_pending_hard_capacity( uint32_t multiplier )'),
        'commit_plan': ('static gamescope::framegen::FixedRefreshCommitPlan_t\nframegen_plan_fixed_refresh_commit(',
                        'static uint64_t framegen_predicted_interval_ns()'),
        'recovery_active': ('static bool framegen_recovery_active_for_path()',
                            'static bool framegen_native_causal_deadline_path()'),
        'recovery_capacity': ('static void framegen_recovery_note_capacity_failure()',
                              'static void framegen_recovery_note_degradation()'),
        'recovery_degradation': ('static void framegen_recovery_note_degradation()',
                                 'static void framegen_log_ladder_recovery('),
        'selector': ('static gamescope::framegen::DeadlineWorkClass_t framegen_causal_work_class(',
                     'static void framegen_record_causal_anchor('),
        'caller': ('static uint64_t framegen_nested_slot_wake_ns(',
                   'static void framegen_apply_live_flip_feedback( const DisplayFeedback_t &feedback )\n{'),
        'learning_deadline': ('static uint64_t framegen_learning_deadline()\n{',
                              'struct FramegenLearningRecording\n'),
    }
    provenance = {}
    for name, (start, end) in seams.items():
        begin, body = between(source, start, end)
        line = source.count('\n', 0, begin) + 1
        (output/(name+'.inc')).write_text(f'#line {line} "{renderer}"\n'+body+'\n')
        provenance[name] = {'source': str(renderer), 'line': line,
                            'sha256': hashlib.sha256(body.encode()).hexdigest()}
    # Execute the exact erase predicate/call, without simulating feedback EMA or
    # duplicating its admission logic. The wrapper supplies a real correction.
    callback_definition = 'static void framegen_apply_live_flip_feedback( const DisplayFeedback_t &feedback )\n{'
    if source.count(callback_definition) != 1:
        raise ValueError('ambiguous/missing feedback definition')
    callback_start = source.index(callback_definition)
    callback_end = source.index('// VRR hybrid slot.', callback_start)
    callback = source[callback_start:callback_end]
    begin, body = between(callback, '\t\tstd::erase_if( g_framegenHistory.pending,',
                          '\t\tconst size_t nDiscarded =')
    line = source.count('\n', 0, callback_start+begin) + 1
    (output/'correction.inc').write_text(f'#line {line} "{renderer}"\n'+body+'\n')
    provenance['correction'] = {'source': str(renderer), 'line': line,
                                'sha256': hashlib.sha256(body.encode()).hexdigest()}
    # Reuse the existing 22 observations, not a copied replacement algorithm.
    tests = ROOT/'tests/framegen_deadline.cpp'
    test_source = tests.read_text()
    _, case = between(test_source,
        'TEST_CASE( "nested correction retains the 22 recorded irreplaceable provisional slots",',
        'TEST_CASE( "nested correction uses strict measured admission before discarding pixels",')
    _, records = between(case, '\tstruct RecordedReplacement', '\tconstexpr uint64_t intervalNs =')
    (output/'recorded.inc').write_text(records+'\n')
    provenance['recorded'] = {'source': str(tests), 'sha256': hashlib.sha256(records.encode()).hexdigest()}
    (output/'fixture.cpp').write_text(FIXTURE.read_text())
    return provenance


def execute(renderer, output):
    output.mkdir(parents=True, exist_ok=True)
    provenance = materialize(renderer, output)
    report = {'renderer': str(renderer),
              'renderer_sha256': hashlib.sha256(renderer.read_bytes()).hexdigest(),
              'fixture_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
              'seams': provenance, 'runs': [],
              'limits': ['CPU fragments with controlled external state, not the complete compositor.',
                         'Headers are live root sources; candidate renderer/header identity must be coordinated.',
                         'No GPU completion, native display, performance, physical latency or quality claim.']}
    for name, flags in (('normal', ['-O0']), ('fastmath', ['-O3', '-ffast-math'])):
        binary = output/name
        command = shlex.split(os.environ.get('CXX', 'c++')) + [
            '-std=c++20', '-Wall', '-Wextra', '-Werror',
            '-Wno-missing-field-initializers', '-UNDEBUG', *flags,
            '-I', str(ROOT/'src'), '-I', str(output),
            str(output/'fixture.cpp'), '-o', str(binary)]
        built = subprocess.run(command, text=True, capture_output=True, timeout=90)
        (output/(name+'-compile.log')).write_text(built.stdout+built.stderr)
        run = {'name': name, 'command': command, 'compile_returncode': built.returncode}
        if built.returncode == 0:
            tested = subprocess.run([str(binary)], text=True, capture_output=True, timeout=20)
            (output/(name+'-run.log')).write_text(tested.stdout+tested.stderr)
            run['test_returncode'] = tested.returncode
            print(tested.stdout, end='')
            print(tested.stderr, end='')
        else:
            print(built.stdout, end='')
            print(built.stderr, end='')
        report['runs'].append(run)
    report['passed'] = all(r['compile_returncode']==0 and r.get('test_returncode')==0 for r in report['runs'])
    (output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renderer', type=Path, default=ROOT/'src/rendervulkan.cpp')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output:
        return execute(args.renderer.resolve(), args.output.resolve())
    with tempfile.TemporaryDirectory(prefix='causal-admission-') as directory:
        return execute(args.renderer.resolve(), Path(directory))


if __name__ == '__main__':
    raise SystemExit(main())
