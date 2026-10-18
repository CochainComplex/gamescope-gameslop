#!/usr/bin/env python3
"""Exercise the native menu in a private headless gamescope, then reap it."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import sys

repo = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix="vapor-smoke-", dir="/tmp") as runtime:
    env = os.environ.copy()
    env.update(XDG_RUNTIME_DIR=runtime, GAMESCOPE_ZENITY_DISABLE="1")
    env["XDG_CONFIG_HOME"] = str(Path(runtime) / "config")
    env.pop("VAPOR_PROFILE_ROOT", None)
    env.pop("VAPOR_GAME", None)
    env.pop("VAPOR_PROFILE", None)
    env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    build_dir = Path(env.get("GAMESCOPE_BUILD_DIR", "build"))
    binary = Path(env.get("GAMESCOPE_MENU_TEST_BINARY", str(repo / build_dir / "src/gamescope"))).resolve()
    # RUNPATH selects the matching runtime. Never inject the legacy prefix
    # into a rebase build. Put its matching reaper on PATH as well.
    env["PATH"] = str(binary.parent) + os.pathsep + env.get("PATH", "")
    env.pop("LD_LIBRARY_PATH", None)
    cmd = [str(binary),
           "--backend", "headless", "--expose-wayland", "-W", "1280", "-H", "720",
           "--", "build-vapor-menu/vapor-menu", "--smoke"]
    if "--gdb" in sys.argv:
        cmd = ["gdb", "--batch", "-ex", "run", "-ex", "thread apply all bt 8", "--args"] + cmd
    process = subprocess.Popen(cmd, cwd=repo, env=env, start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        try:
            output, _ = process.communicate(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
        log = repo / "build-vapor-menu/smoke.log"
        log.write_text(output)
        print(output[-7000:])
        print(f"Exit: {process.returncode}; log: {log}")
        if process.returncode or "configure/render/navigate/hide/reopen PASS" not in output:
            raise SystemExit(1)
    finally:
        # Kill only the process group this test created, including Xwayland.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
