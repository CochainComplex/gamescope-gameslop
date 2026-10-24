#!/usr/bin/env python3
"""Selection-time gap regressions against the actual smoke checker."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/framegen-nested-smoke.py'
spec = importlib.util.spec_from_file_location('nested_smoke_contract', SCRIPT)
S = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = S
spec.loader.exec_module(S)


def presented(receipt, identity, selection=None):
    suffix = '' if selection is None else f' selection_ns={selection}'
    return receipt, f'framegen: presented generated frame id={identity}.50'+suffix


class SelectionGapTests(unittest.TestCase):
    windows = [(15.0, 66.667)]

    def check_gap(self, records, since=None, last=None):
        timeline = S.bidir_timeline(records)
        return S.bidir_gap_errors(timeline, self.windows,
                                  records[0][0] if since is None else since,
                                  records[-1][0] if last is None else last)

    def test_confirmed_receipt_jitter_uses_selection_interval(self):
        records = [presented(13.279, 328, 232139325023750),
                   presented(13.365, 329, 232139410331109)]
        self.assertEqual(S.delay_bound_ns(self.windows,13.279,13.365)+S.LOG_SLACK_NS,85_333_667)
        errors, evidence = self.check_gap(records)
        self.assertEqual(errors, [])
        self.assertIn('selection_ns=1, receipt=2', evidence)

    def test_real_over_bound_selection_gap_cannot_hide_in_fast_log_pump(self):
        errors, evidence = self.check_gap([presented(13.279,1,10**12),
                                          presented(13.359,2,10**12+86_000_000)])
        self.assertTrue(errors)
        self.assertIn('86 ms', errors[0])
        self.assertIn('(selection_ns)', errors[0])

    def test_missing_or_mixed_stamps_keep_legacy_receipt_fallback(self):
        for first, second in ((None,None),(None,10**12),(10**12,None)):
            with self.subTest(first=first,second=second):
                errors,evidence = self.check_gap([presented(13.279,1,first),presented(13.365,2,second)])
                self.assertTrue(errors)
                self.assertIn('selection_ns=0, receipt=3', evidence)
                self.assertIn('(receipt)',errors[0])

    def test_legacy_in_bound_interval_still_passes(self):
        errors,_ = self.check_gap([presented(13.279,1),presented(13.359,2)])
        self.assertEqual(errors,[])

    def test_since_and_last_synthetic_boundaries_use_receipt_time(self):
        record = presented(13.4,1,232139325023750)
        for since,last in ((13.279,13.4),(13.4,13.521)):
            with self.subTest(since=since,last=last):
                errors,evidence = self.check_gap([record],since,last)
                self.assertTrue(errors)
                self.assertIn('(receipt)',errors[0])
                self.assertIn('selection_ns=0, receipt=2',evidence)

    def test_transition_membership_stays_in_receipt_time(self):
        a,b = presented(13.279,1,10**12),presented(13.365,2,10**12+160_000_000)
        for transition,passes in ((13.3,True),(13.2,False)):
            with self.subTest(transition=transition):
                records = sorted([a,b,(transition,'framegen: reset history reason=test')])
                errors,_ = self.check_gap(records,13.279,13.365)
                self.assertEqual(not errors,passes)

    def test_equal_or_reversed_stamps_are_errors(self):
        for second in (10**12,10**12-1):
            with self.subTest(second=second):
                errors,_ = self.check_gap([presented(13.279,1,10**12),presented(13.359,2,second)])
                self.assertTrue(any('non-increasing selection_ns' in error for error in errors))

    def test_malformed_stamps_are_errors_not_silent_fallback_passes(self):
        for stamp in ('oops','-1','0',str(2**64),'','123 selection_ns=456','9'*5000):
            with self.subTest(stamp=stamp[:30]):
                errors,_ = self.check_gap([presented(13.279,1,stamp),presented(13.359,2)])
                self.assertTrue(any('invalid selection_ns' in error for error in errors))

    def test_same_receipt_timestamp_does_not_alias_selected_content(self):
        records = [presented(13.279,1,10**12),presented(13.279,2,10**12+86_000_000)]
        timeline = S.bidir_timeline(records)
        self.assertEqual(len(timeline.contents),2)
        self.assertEqual(S.window_fill(timeline,13.279)[0],2)
        errors,_ = self.check_gap(records)
        self.assertTrue(errors)

    def test_fill_membership_uses_receipt_time_not_absolute_selection_clock(self):
        records = [presented(10.0,1,10**12),presented(10.001,2,10**12+1)]
        timeline = S.bidir_timeline(records)
        self.assertEqual(S.window_fill(timeline,15.0)[0],1) # lower boundary excluded


if __name__ == '__main__':
    unittest.main()
