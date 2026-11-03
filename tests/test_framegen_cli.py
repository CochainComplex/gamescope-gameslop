#!/usr/bin/env python3
"""Exercise the framegen CLI without starting a compositor or querying a GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FAKE = r'''
import json, os, pathlib, subprocess, sys
name = pathlib.Path(sys.argv[0]).name
if name in ('gamescope', 'timeout'):
    with open(os.environ['TRACE'], 'a') as trace:
        trace.write(json.dumps([name, sys.argv[1:],
                                os.environ.get('GAMESCOPE_FRAMEGEN_DEBUG_EVERY')]) + '\n')
if name == 'gamescope':
    sys.stdout.write(os.environ['LOG_TEXT'])
    sys.stdout.flush()
    sys.stderr.write(os.environ.get('LOG_STDERR', ''))
    sys.exit(int(os.environ['RUN_STATUS']))
if name == 'tee':
    data = sys.stdin.buffer.read()
    pathlib.Path(sys.argv[1]).write_bytes(data)
    sys.stdout.buffer.write(data)
    sys.exit(int(os.environ['TEE_STATUS']))
if name == 'timeout':
    result = subprocess.run(sys.argv[2:])
    sys.exit(int(os.environ.get('TIMEOUT_STATUS', result.returncode)))
if name == 'lspci':
    print('00:01.0 VGA compatible controller: test AMD [1002:1234]')
'''


class FramegenCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='framegen-cli-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scripts = self.root / 'scripts'
        scripts.mkdir()
        self.script = scripts / 'test-framegen.sh'
        shutil.copyfile(ROOT / 'scripts/test-framegen.sh', self.script)
        shutil.copyfile(ROOT / 'scripts/env-gamescope-local.sh',
                        scripts / 'env-gamescope-local.sh')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for name in ('gamescope', 'tee', 'timeout', 'lspci', 'vkcube'):
            path = self.bin / name
            path.write_text(f'#!{sys.executable}\n' + FAKE)
            path.chmod(0o755)

    def run_cli(self, args=(), **overrides):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith('GAMESCOPE_') and key not in
               ('REFRESH', 'LIMIT', 'DEBUG_EVERY', 'PIPELINE', 'DURATION',
                'BASH_ENV', 'DISPLAY', 'WAYLAND_DISPLAY')}
        trace = self.root / 'trace.jsonl'
        trace.unlink(missing_ok=True)
        env.update(PATH=f'{self.bin}:/usr/bin:/bin', TMPDIR=str(self.root),
                   GAMESCOPE_LOCAL_PREFIX=str(self.root / 'prefix'),
                   GAMESCOPE_BIN=str(self.bin / 'gamescope'),
                   TRACE=str(trace), LOG_TEXT='', RUN_STATUS='0', TEE_STATUS='0')
        env.update(overrides)
        result = subprocess.run(['bash', str(self.script), *args], env=env,
                                capture_output=True, text=True, timeout=5)
        calls = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        return result, calls

    def test_app_arguments_and_launch_settings(self):
        cases = [([], ['vkcube']),
                 (['--', 'vkcube', 'two words', '', '*'], ['vkcube', 'two words', '', '*']),
                 (['vkcube', 'two words'], ['vkcube', 'two words']),
                 (['vkcube one two'], ['vkcube', 'one', 'two'])]
        for suffix, app in cases:
            with self.subTest(suffix=suffix):
                result, calls = self.run_cli(['run', '1002:ABCD', 'motion', '1440', *suffix],
                                            REFRESH='120', LIMIT='3', DEBUG_EVERY='7', PIPELINE='spatial')
                self.assertEqual((result.returncode, result.stderr), (0, ''))
                self.assertEqual(calls, [['gamescope', [
                    '--backend', 'wayland', '--prefer-vk-device', '1002:abcd',
                    '-W', '2560', '-H', '1440', '-r', '120', '--framerate-limit', '3',
                    '--experimental-framegen', '--framegen-mode', 'motion',
                    '--framegen-pipeline', 'spatial', '--framegen-strength', '0.5',
                    '--framegen-debug', '--', *app], '7']])
                self.assertIn('refresh=120  game=~40fps', result.stdout)

    def test_default_launch(self):
        result, calls = self.run_cli(['run'])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls[0][1][4:14],
                         ['-W', '1920', '-H', '1080', '-r', '144', '--framerate-limit', '2',
                          '--experimental-framegen', '--framegen-mode'])
        self.assertEqual(calls[0][1][-1], 'vkcube')
        self.assertIn('device=1002:1234  mode=extrapolate/warp', result.stdout)

    def test_validation_order_and_no_launch(self):
        cases = [(['run', 'bad'], {}, "unknown gpu 'bad'"),
                 (['run', 'auto', 'motion', '1080', '--'], {}, 'app command is empty'),
                 (['run', 'auto', 'bad', 'bad'], {}, 'res must be'),
                 (['run', 'auto', 'bad'], {'PIPELINE': 'bad'}, 'mode must be'),
                 (['run'], {'PIPELINE': 'bad', 'REFRESH': '0'}, 'PIPELINE must be'),
                 (['run'], {'REFRESH': '0', 'LIMIT': '0'}, 'REFRESH must be'),
                 (['run'], {'LIMIT': '0', 'DEBUG_EVERY': '0'}, 'LIMIT must be'),
                 (['run'], {'DEBUG_EVERY': '0', 'DURATION': 'bad'}, 'DEBUG_EVERY must be'),
                 (['run'], {'DURATION': 'bad'}, 'DURATION must be'),
                 (['run', 'auto', 'motion', '1080', '--', 'missing-test-app'], {}, 'not found'),
                 (['run', 'bad'], {'GAMESCOPE_BIN': '/missing/gamescope'}, 'binary not found')]
        for args, env, error in cases:
            with self.subTest(args=args, env=env):
                result, calls = self.run_cli(args, **env)
                self.assertEqual(result.returncode, 1)
                self.assertIn(error, result.stderr)
                self.assertEqual((calls, result.stdout), ([], ''))

    def test_pipeline_status_precedence_and_timeout(self):
        for duration in ('', '0', '0.25'):
            for run, tee, expected in ((0, 0, 0), (7, 0, 7), (0, 9, 9),
                                       (7, 9, 7), (124, 0, 124), (0, 124, 124)):
                with self.subTest(duration=duration, run=run, tee=tee):
                    result, calls = self.run_cli(['run'], DURATION=duration,
                                                RUN_STATUS=str(run), TEE_STATUS=str(tee))
                    self.assertEqual(result.returncode, 0 if duration and expected == 124 else expected)
                    self.assertIn('RESULT: no generated frames presented.', result.stdout)
                    self.assertEqual([call[0] for call in calls],
                                     ['timeout', 'gamescope'] if duration else ['gamescope'])
                    if duration:
                        self.assertEqual(calls[0][1][0], duration)
                        self.assertEqual(calls[0][2], '1')

    def test_exact_summary_and_retained_log(self):
        for enabled, presented in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(enabled=enabled, presented=presented):
                log = ('prefix framegen: enabled old\nprefix framegen: enabled latest\n' if enabled else '')
                log += ('vblank slot=real vblank slot=real\nvblank slot=real\n'
                        'vblank slot=generated\nvblank slot=repeat\n'
                        'framegen: generated 3\nframegen: generated\ngeneration_too_slow\n')
                log += 'framegen: presented generated 1\n' if presented else ''
                result, _ = self.run_cli(['run'], LOG_TEXT=log, LOG_STDERR='diagnostic\n')
                self.assertEqual((result.returncode, result.stderr), (0, ''))
                path = Path(next(line[7:] for line in result.stdout.splitlines() if line.startswith('# log: ')))
                self.assertEqual(path.read_text(), log + 'diagnostic\n')
                summary = '\n==================== framegen cadence summary ====================\n'
                summary += ('  framegen: enabled latest\n' if enabled else
                            "  framegen never enabled — no base frames were composited\n"
                            "  (window didn't render? try a different app, or check the log below)\n")
                for label, count in (('vblank slots: real', 2), ('vblank slots: generated', 1),
                                     ('vblank slots: repeat', 1), ('generation batches', 1),
                                     ('generated frames presented', int(presented)), ('discarded (too slow)', 1)):
                    summary += f'  {label:<26} {count}\n'
                summary += '-----------------------------------------------------------------\n'
                summary += ('  RESULT: PASS — synthetic frames were generated and presented.\n' if presented else
                            '  RESULT: no generated frames presented.\n'
                            '    - Ensure the game runs BELOW the display rate (LIMIT>1); with no gap\n'
                            '      there are no empty vblanks to fill.\n'
                            '    - Shared-queue motion mode needs a short stable-cadence warm-up.\n'
                            f'    - Full log: {path}\n')
                summary += '=================================================================\n'
                self.assertTrue(result.stdout.endswith(summary), result.stdout)


if __name__ == '__main__':
    unittest.main()
