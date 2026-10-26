#!/usr/bin/env python3
"""Exercise production GPU diagnostics transport without Vulkan or a GPU."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
# This file does not compile wlserver.cpp: its only GPU-diagnostics call must be the tested opt-in helper.
code = re.sub(r"\s+", "", (ROOT / "src/wlserver.cpp").read_text())
assert re.findall(r"create_gpu_diagnostics_global\w*\(", code) == ["create_gpu_diagnostics_global_if_requested("]

with tempfile.TemporaryDirectory(prefix="gameslop-gpu-ipc-") as directory:
    work = Path(directory)
    protocol = ROOT / "protocol/gamescope-control.xml"
    for mode, output in (("server-header", "gamescope-control-protocol.h"),
                         ("client-header", "control-client.h"), ("private-code", "protocol.c")):
        subprocess.run(["wayland-scanner", mode, str(protocol), str(work / output)], check=True)
    flags = shlex.split(subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", "wayland-server", "wayland-client"], text=True))
    subprocess.run(["cc", "-c", str(work / "protocol.c"), "-o", str(work / "protocol.o"), *flags], check=True)
    subprocess.run(["c++", "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(work),
                    "-I", str(ROOT / "src"), str(ROOT / "tools/vapor-menu/test_gpu_diagnostics.cpp"),
                    str(work / "protocol.o"), "-o", str(work / "test"), *flags], check=True)
    # Exercise the production getenv, gate and logging through a real registry.
    base = {k: v for k, v in os.environ.items() if k != "GAMESCOPE_GPU_DIAGNOSTICS"}
    cases = [(None, "--expect-absent", "quiet")]
    cases += [(value, "--expect-absent", "quiet") for value in ("0", "false", "no", "off")]
    cases += [(value, "--expect-absent", "error")
              for value in ("", "bogus", "TRUE", "2", " 1", "1 ", "1\n")]
    cases += [(value, "--expect-present", "info") for value in ("1", "true", "yes", "on")]
    for value, expectation, log_expectation in cases:
        env = dict(base) if value is None else dict(base, GAMESCOPE_GPU_DIAGNOSTICS=value)
        subprocess.run([str(work / "test"), expectation, log_expectation], check=True, timeout=20, env=env)
