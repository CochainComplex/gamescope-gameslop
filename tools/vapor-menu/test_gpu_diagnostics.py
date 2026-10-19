#!/usr/bin/env python3
"""Exercise production GPU diagnostics transport without Vulkan or a GPU."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
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
    # Opt-in: absent by default and for off/invalid values, present when on.
    base = {k: v for k, v in os.environ.items() if k != "GAMESCOPE_GPU_DIAGNOSTICS"}
    for value, expectation in ((None, "--expect-absent"), ("0", "--expect-absent"),
                               ("bogus", "--expect-absent"), ("1", "--expect-present")):
        env = dict(base) if value is None else dict(base, GAMESCOPE_GPU_DIAGNOSTICS=value)
        subprocess.run([str(work / "test"), expectation], check=True, timeout=20, env=env)
