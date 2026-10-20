"""Artifact identity and existing-tool imports; no SDK dependencies."""

import hashlib
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import signal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Terminated(BaseException):
    """A process termination request must escape library exception handlers."""


@contextmanager
def termination_signals():
    """Turn normal process termination into an exception so receipts get flushed."""
    def interrupted(signum, _frame):
        # Let receipt/cache cleanup finish even if termination is requested again.
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        raise Terminated(f'terminated by {signal.Signals(signum).name}')
    previous = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        handler = signal.getsignal(sig)
        if sig == signal.SIGHUP and handler == signal.SIG_IGN:
            continue  # preserve nohup
        previous[sig] = signal.signal(sig, interrupted)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace('-', '_'),
                                                ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result
