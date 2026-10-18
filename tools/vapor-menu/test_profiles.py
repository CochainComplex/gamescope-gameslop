#!/usr/bin/env python3
"""Profile storage and launch contracts; no GPU or game is started."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("launcher", ROOT / "scripts/vapor-profile.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class Profiles(unittest.TestCase):
    def test_cpp_storage_and_python_reader_agree(self):
        with tempfile.TemporaryDirectory(prefix="vapor-profiles-", dir="/tmp") as work:
            binary = Path(work) / "test"
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++20", "-Wall", "-Wextra", "-Werror",
                            str(ROOT / "tools/vapor-menu/test_profiles.cpp"), "-o", str(binary)], check=True)
            subprocess.run([str(binary), work], check=True)
            loaded = launcher.load(Path(work) / "roundtrip.ini")
            self.assertEqual((loaded["multiplier"], loaded["pipeline"]), ("4", "guided"))
            self.assertEqual(launcher.load(Path(work) / "test-game/default.ini")["pipeline"], "spatial")

    def test_launch_uses_profile_without_shell_or_library_override(self):
        with tempfile.TemporaryDirectory(prefix="vapor-launch-", dir="/tmp") as work:
            folder = Path(work) / "game"
            folder.mkdir()
            (folder / "test.ini").write_text("version=1\npipeline=warp\nmultiplier=3\nhud=0\nonline=off\n")
            args = argparse.Namespace(game="game", profile="test", profile_root=work,
                                      net_profile="/tmp/weights with spaces.bin", gamescope="/host/gamescope",
                                      host_arg=["-f"], command=["game", "literal $(not a shell)"])
            command, env, *_ = launcher.launch_plan(args, {"LD_LIBRARY_PATH": "/old/prefix"})
            self.assertEqual(command[command.index("--framegen-pipeline")+1], "warp")
            self.assertEqual(command[command.index("--framegen-multiplier")+1], "3")
            self.assertEqual(command[-2:], args.command)
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_NET_PROFILE"], args.net_profile)
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_HUD"], "0")
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_NET_ONLINE"], "0")
            self.assertNotIn("LD_LIBRARY_PATH", env)
            self.assertIn("LD_LIBRARY_PATH=/old/prefix", command)  # Restored only for the game.
            self.assertFalse(Path(args.net_profile).exists())
            (folder / ".active").write_text("test\n")
            args.profile = None
            self.assertEqual(launcher.launch_plan(args, {})[1]["VAPOR_PROFILE"], "test")

    def test_spatial_uses_the_same_training_container_and_local_host(self):
        with tempfile.TemporaryDirectory(prefix="vapor-spatial-", dir="/tmp") as work:
            folder = Path(work) / "game"
            folder.mkdir()
            weights = folder / "learned.bin"
            weights.write_bytes(b"private existing training data")
            (folder / "default.ini").write_text(
                f"version=1\npipeline=spatial\nnet_profile={weights}\n")
            args = argparse.Namespace(game="game", profile="default", profile_root=work,
                                      net_profile=None, gamescope="/host/gamescope",
                                      host_arg=[], command=["game"])
            command, env, *_ = launcher.launch_plan(args, {})
            self.assertEqual(command[command.index("--framegen-pipeline")+1], "spatial")
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_NET_PROFILE"], str(weights))
            self.assertEqual(weights.read_bytes(), b"private existing training data")
            self.assertEqual(launcher.default_gamescope({"GAMESCOPE_BINARY": "/explicit/host"}),
                             "/explicit/host")
            build = Path(work) / "build/src"
            build.mkdir(parents=True)
            host = build / "gamescope"
            host.touch()
            host.chmod(0o700)
            self.assertEqual(launcher.default_gamescope({"GAMESCOPE_BUILD_DIR": str(build.parent)}),
                             str(host))

    def test_reject_invalid_presets(self):
        for invalid in ["..", "../game", "with space", "", "a"*65]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                launcher.identifier(invalid)
        with tempfile.TemporaryDirectory(dir="/tmp") as work:
            path = Path(work) / "bad.ini"
            for content in ["version=2\n", "pipeline=warp\n", "version=1\nmultiplier=99\n",
                            "version=1\nhud=1\nhud=0\n", "version=1\nunknown=x\n"]:
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    launcher.load(path)


if __name__ == "__main__":
    unittest.main()
