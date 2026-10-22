#!/usr/bin/env python3
"""Check local layer discovery overrides without invoking a Vulkan loader."""
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    script = Path(__file__).resolve().parents[1] / "scripts/env-gamescope-local.sh"
    with tempfile.TemporaryDirectory(prefix="gamescope-env-") as directory:
        repo = Path(directory)
        local = repo / "build/layer"
        local.mkdir(parents=True)
        for override in (None, "", "/chosen/layers"):
            env = {key: value for key, value in os.environ.items()
                   if key not in ("VK_LAYER_PATH", "VK_ADD_LAYER_PATH")}
            env.update(GAMESCOPE_REPO=str(repo), GAMESCOPE_BUILD_DIR="build",
                       GAMESCOPE_LOCAL_PREFIX=str(repo / "prefix"),
                       VK_ADD_LAYER_PATH="/extra/layers")
            if override is not None:
                env["VK_LAYER_PATH"] = override
            result = subprocess.run(["bash", "-c", 'source "$1"; source "$1"; env -0',
                                     "env-fixture", str(script)], env=env,
                                    capture_output=True, check=True, timeout=5)
            observed = dict(item.split(b"=", 1) for item in result.stdout.split(b"\0") if item)
            if override is None:
                assert b"VK_LAYER_PATH" not in observed
                assert observed[b"VK_ADD_LAYER_PATH"] == f"{local}:/extra/layers".encode()
            else:
                expected = str(local) + (":" + override if override else "")
                assert observed[b"VK_LAYER_PATH"] == expected.encode()
                assert observed[b"VK_ADD_LAYER_PATH"] == b"/extra/layers"
    print("Local environment preserves default and explicit layer discovery: PASS")


if __name__ == "__main__":
    main()
