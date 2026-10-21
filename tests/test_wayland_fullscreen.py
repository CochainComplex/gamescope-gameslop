#!/usr/bin/env python3
"""Exercise production Wayland focus/hotkey code with deterministic owner removal."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def function(source, signature, indent="    "):
    start = source.index(signature + "\n" + indent + "{")
    return source[start:source.index("\n" + indent + "}\n", start) + len(indent) + 3]


def declaration(source, name):
    return next(line for line in source.splitlines() if name in line and line.rstrip().endswith(";")) + "\n"


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/Backends/WaylandBackend.cpp").read_text()
    backend = source[source.index("class CWaylandBackend :"):]
    backend = backend[:backend.index("\n    };")]
    assert backend.rstrip().endswith("CWaylandInputThread m_InputThread;"), \
        "input must stop and join before backend members are destroyed"
    constructor = function(source, "CWaylandInputThread::CWaylandInputThread()")
    assert "m_Thread = std::thread" in constructor, "start the worker after its init gate is constructed"

    methods = [
        "CWaylandInputThread::CWaylandInputThread()",
        "CWaylandInputThread::~CWaylandInputThread()",
        "std::weak_ptr<std::atomic<bool>> CWaylandConnector::GetFullscreenState() const",
        "CWaylandConnector::~CWaylandConnector()",
        "std::shared_ptr<IBackendConnector> CWaylandBackend::CreateVirtualConnector( uint64_t ulVirtualConnectorKey )",
        "void CWaylandInputThread::SetFullscreenState( std::weak_ptr<std::atomic<bool>> pState )",
        "void CWaylandInputThread::RunHotkey( uint32_t uKey )",
    ]
    # Execute the production worker through its first backend-read statement.
    # A scope-exit marker and the atomic gate fixture force the old destructor's
    # wake-before-stop interleaving without copying its branch or waiter logic.
    worker = function(source, "void CWaylandInputThread::ThreadFunc()")
    first_read = "int nFD = wl_display_get_fd( m_pBackend->GetDisplay() );"
    worker = worker[:worker.index(first_read) + len(first_read)] + "\n        (void)nFD;\n    }\n"
    worker = worker.replace("\n    {", "\n    {\n        WorkerFinished finished;", 1)
    waiter = (root / "src/waitable.h").read_text()
    waiter = waiter[:waiter.index("    // A raw pointer class")] + "}\n"
    waiter = waiter.replace('#include "log.hpp"', "")
    with tempfile.TemporaryDirectory(prefix="gamescope-wayland-fullscreen-") as directory:
        directory = Path(directory)
        (directory / "wayland_fullscreen_waiter.inc").write_text(waiter)
        (directory / "wayland_fullscreen_methods.inc").write_text(
            "\n".join(function(source, signature) for signature in methods) + worker)
        for filename, member in (
            ("connector_state", "m_pDesiredFullscreenState ="),
            ("input_state", "m_pFullscreenState;"),
            ("focus", "m_pFocusConnector;"),
        ):
            (directory / f"wayland_fullscreen_{filename}.inc").write_text(declaration(source, member))
        (directory / "wayland_fullscreen_connector_destroyed.inc").write_text(
            function(source, "void OnConnectorDestroyed( CWaylandConnector *pConnector )", indent="        "))

        binary = directory / "test"
        subprocess.run(shlex.split(os.environ.get("CXX", "c++"))
                       + ["-std=c++20", "-pthread", "-Wall", "-Wextra", "-I", str(directory),
                          str(root / "tests/test_wayland_fullscreen.cpp"), "-o", str(binary)],
                       check=True, timeout=60)
        for case in ("startup", "current", "expired", "pinned", "replacement", "failed-replacement", "old-destroy"):
            subprocess.run([str(binary), case], check=True, timeout=10)
            print(f"Wayland fullscreen {case}: PASS", flush=True)


if __name__ == "__main__":
    main()
