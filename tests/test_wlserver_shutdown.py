#!/usr/bin/env python3
"""Run the production Wayland loop/stop functions against CPU-only event fixtures."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def function(source, signature):
    start = source.index(signature + "\n{")
    return source[start:source.index("\n}\n", start) + 3]


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/wlserver.cpp").read_text()
    compositor = function((root / "src/steamcompmgr.cpp").read_text(),
                          "static void\nsteamcompmgr_exit(void)")
    order = ["vulkan_framegen_flush_profile();", "wlserver_shutdown_input();",
             "vulkan_framegen_shutdown();", "wlserver_shutdown();",
             "gamescope::IBackend::Set( nullptr );"]
    positions = [compositor.index(item) for item in order]
    assert positions == sorted(positions), "backend users must retire before deletion"
    loop = function(source, "void wlserver_run(void)")
    assert (loop.index("NotifyPhysicalInput") < loop.index("wlserver.xdg_wins.clear();")
            < loop.index("g_bWaylandDispatchStopped = true;")
            < loop.index("xwayland_server_guard")), "acknowledge before Xwayland mutex wait"
    declarations = [line for line in source.splitlines()
                    if line.startswith(("static std::atomic<bool> g_bShutdownWLServer",
                                        "static std::atomic<bool> g_bWaylandDispatchStopped",
                                        "static int g_wlserverNudgePipe"))]
    with tempfile.TemporaryDirectory(prefix="gamescope-wl-shutdown-") as directory:
        directory = Path(directory)
        (directory / "wlserver_loop.inc").write_text(
            "\n".join(declarations) + "\n" + loop
            + function(source, "void wlserver_shutdown()"))
        binary = directory / "test"
        subprocess.run(shlex.split(os.environ.get("CXX", "c++"))
                       + ["-std=c++20", "-pthread", "-Wall", "-Wextra", "-Wno-missing-field-initializers",
                          "-I", str(directory), str(root / "tests/test_wlserver_shutdown.cpp"),
                          "-o", str(binary)], check=True, timeout=60)
        for case in ("startup", "idle", "dispatch", "notification", "error"):
            subprocess.run([str(binary), case], check=True, timeout=10)
            print(f"Wayland shutdown {case}: PASS", flush=True)


if __name__ == "__main__":
    main()
