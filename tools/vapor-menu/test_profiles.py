#!/usr/bin/env python3
"""Profile storage and launch contracts; no GPU or game is started."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import struct
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
                            "-I", str(ROOT / "src"),
                            str(ROOT / "tools/vapor-menu/test_profiles.cpp"), "-o", str(binary)], check=True)
            subprocess.run([str(binary), work], check=True)
            loaded = launcher.load(Path(work) / "roundtrip.ini")
            self.assertEqual(loaded["style"], "cga")
            self.assertEqual(loaded["hud_opacity"], "70")
            self.assertEqual(loaded["hud_scale"], "3")
            self.assertEqual((loaded["multiplier"], loaded["pipeline"]), ("4", "guided"))
            self.assertEqual(launcher.load(Path(work) / "test-game/default.ini")["pipeline"], "spatial")
            launcher.validate_net_profile(Path(work) / "keep.bin")  # Real C++ encoder.
            cpp_choices = subprocess.check_output([str(binary), "--choices"], text=True)
            self.assertEqual({line.split()[0]: tuple(line.split()[1:]) for line in cpp_choices.splitlines()},
                             launcher.CHOICES)
            matrix = subprocess.check_output([str(binary), "--matrix"], text=True).splitlines()
            self.assertEqual(len(matrix), 1440)
            for row in matrix:
                fg, mode, pipeline, bidir, backend, base, vrr, valid, net = row.split()
                settings = dict(launcher.DEFAULTS, framegen=fg, mode=mode, pipeline=pipeline, bidir=bidir)
                context = dict(backend=backend, base=base=="1", vrr_hybrid=vrr=="1")
                with self.subTest(config=row):
                    if valid=="1":
                        launcher.validate(settings, context)
                    else:
                        with self.assertRaises(ValueError):
                            launcher.validate(settings, context)
                    self.assertEqual(launcher.uses_net(settings), net=="1")

    def test_launch_uses_profile_without_shell_or_library_override(self):
        with tempfile.TemporaryDirectory(prefix="vapor-launch-", dir="/tmp") as work:
            folder = Path(work) / "game"
            folder.mkdir()
            (folder / "test.ini").write_text("version=1\npipeline=warp\nmultiplier=3\nhud=0\nonline=off\nstyle=cga\nhud_opacity=70\nhud_scale=3\n")
            args = argparse.Namespace(game="game", profile="test", profile_root=work,
                                      net_profile="/tmp/weights with spaces.bin", gamescope="/host/gamescope",
                                      host_arg=["-f"], command=["game", "literal $(not a shell)"])
            command, env, *_ = launcher.launch_plan(args, {"LD_LIBRARY_PATH": "/old/prefix"})
            self.assertEqual(command[command.index("--framegen-pipeline")+1], "warp")
            self.assertEqual(command[command.index("--framegen-multiplier")+1], "3")
            self.assertEqual(command[-2:], args.command)
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_NET_PROFILE"], args.net_profile)
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_HUD"], "0")
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_HUD_STYLE"], "cga")
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_HUD_OPACITY"], "70")
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_HUD_SCALE"], "3")
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_NET_ONLINE"], "0")
            self.assertEqual(env["GAMESCOPE_GPU_DIAGNOSTICS"], "1")  # GPU DETAILS page.
            # A user's explicit opt-out is respected, not overridden.
            self.assertEqual(launcher.launch_plan(args, {"GAMESCOPE_GPU_DIAGNOSTICS": "0"})[1]["GAMESCOPE_GPU_DIAGNOSTICS"], "0")
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
            metadata = struct.pack("<12I", 0x52465347, 3, 3, 12, 16, 3, 16, 16, 3, 16, 4, 3)
            original = struct.pack("<4I", 0x50465347, 1, 1, 0x100f0) + metadata + bytes(4644*4)
            weights.write_bytes(original)
            (folder / "default.ini").write_text(
                f"version=1\npipeline=spatial\nnet_profile={weights}\n")
            args = argparse.Namespace(game="game", profile="default", profile_root=work,
                                      net_profile=None, gamescope="/host/gamescope",
                                      host_arg=[], command=["game"])
            command, env, *_ = launcher.launch_plan(args, {})
            self.assertEqual(command[command.index("--framegen-pipeline")+1], "spatial")
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_NET_PROFILE"], str(weights))
            self.assertEqual(weights.read_bytes(), original)
            self.assertEqual(launcher.default_gamescope({"GAMESCOPE_BINARY": "/explicit/host"}),
                             "/explicit/host")
            build = Path(work) / "build/src"
            build.mkdir(parents=True)
            host = build / "gamescope"
            host.touch()
            host.chmod(0o700)
            self.assertEqual(launcher.default_gamescope({"GAMESCOPE_BUILD_DIR": str(build.parent)}),
                             str(host))

    def test_auto_hud_appearance_clears_inherited_values(self):
        with tempfile.TemporaryDirectory(prefix="vapor-auto-", dir="/tmp") as work:
            args = argparse.Namespace(game="game", profile="default", profile_root=work,
                                      net_profile=None, gamescope="/host/gamescope",
                                      host_arg=[], command=["game"])
            inherited = {"GAMESCOPE_FRAMEGEN_HUD_OPACITY": "100", "GAMESCOPE_FRAMEGEN_HUD_SCALE": "6"}
            _, env, *_ = launcher.launch_plan(args, inherited)
            self.assertNotIn("GAMESCOPE_FRAMEGEN_HUD_OPACITY", env)
            self.assertNotIn("GAMESCOPE_FRAMEGEN_HUD_SCALE", env)

    def test_experimental_preset_and_unavailable_npu_are_truthful(self):
        with tempfile.TemporaryDirectory(prefix="vapor-experiments-", dir="/tmp") as work:
            folder = Path(work) / "game"
            self.assertEqual(launcher.load(folder / "spatial-experimental.ini")["pipeline"], "spatial")
            args = argparse.Namespace(game="game", profile="spatial-experimental", profile_root=work,
                                      net_profile=None, gamescope="/host/gamescope", host_arg=[], command=["game"])
            command, env, *_ = launcher.launch_plan(args, {})
            self.assertEqual(command[command.index("--framegen-pipeline")+1], "spatial")
            self.assertEqual(env["GAMESCOPE_FRAMEGEN_NET_ONLINE"], "1")
            self.assertFalse(folder.exists())  # Virtual preset never rewrites weights/settings.
            folder.mkdir()
            path = folder / "spatial-experimental.ini"
            path.write_text("version=1\npipeline=guided\n")
            self.assertEqual(launcher.load(path)["pipeline"], "guided")  # User override wins.
            path.write_text("version=1\ntraining_device=npu\n")
            with self.assertRaisesRegex(ValueError, "NPU training unavailable"):
                launcher.launch_plan(args, {})
            self.assertFalse((folder / "learned.bin").exists())

    def test_spatial_builtin_refuses_older_host_without_writes(self):
        with tempfile.TemporaryDirectory(prefix="vapor-old-host-", dir="/tmp") as work:
            host = Path(work) / "gamescope"
            host.write_text("#!/bin/sh\n[ \"$1\" = --help ] || exit 91\necho \"--framegen-pipeline NAME 'warp', 'learned'\"\n")
            host.chmod(0o700)
            root = Path(work) / "profiles"
            command = ["python3", str(ROOT / "scripts/vapor-profile.py"), "--game", "game",
                       "--profile", "spatial-experimental", "--profile-root", str(root),
                       "--gamescope", str(host), "--dry-run", "--", "game"]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("Spatial" if ROOT.name == "gamescope" else "does not support", result.stderr)
            self.assertFalse(root.exists())

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

    def test_launch_rejects_ignored_or_overridden_configurations(self):
        with tempfile.TemporaryDirectory(prefix="vapor-config-", dir="/tmp") as work:
            folder = Path(work) / "game"
            folder.mkdir()
            preset = folder / "default.ini"
            args = argparse.Namespace(game="game", profile="default", profile_root=work,
                                      net_profile=None, gamescope="/host/gamescope",
                                      host_arg=[], command=["game"])
            cases = [
                ("pipeline=spatial\nbidir=on\n", [], {}),
                ("mode=blend\nbidir=on\n", [], {}),
                ("mode=extrapolate\nbidir=on\n", [], {}),
                ("bidir=on\n", ["--backend=sdl"], {}),
                ("bidir=on\n", [], {"DISPLAY": ":99"}),
                ("bidir=on\n", [], {"GAMESCOPE_FRAMEGEN_BASE": "yes"}),
                ("bidir=on\n", [], {"GAMESCOPE_FRAMEGEN_VRR_HYBRID": "on"}),
                ("", ["--backend", "headless"], {}),
                ("", ["--backend=openvr"], {}),
                ("", ["--force-orientation=left"], {}),
                ("", ["--virtual-connector-strategy=PerWindow"], {}),
                ("", ["--framegen-pipeline=warp"], {}),
                ("", ["--framegen-p", "warp"], {}),
                ("framegen=off\n", ["--experimental-framegen"], {}),
                ("", ["--"], {}),
                ("", ["--back=sdl"], {}),
                ("", ["--backend=wayland", "--backend=drm"], {}),
                ("", [], {"GAMESCOPE_FRAMEGEN_BASE": "perhaps"}),
            ]
            for content, host_args, env in cases:
                preset.write_text("version=1\n" + content)
                args.host_arg = host_args
                with self.subTest(content=content, host_args=host_args, env=env), self.assertRaises(ValueError):
                    launcher.launch_plan(args, env)
            # Both output-ring backends retain bidir; auto follows host env.
            preset.write_text("version=1\nbidir=on\n")
            for host_args, env in [(["--backend=drm"], {}),
                                   (["--backend", "wayland"], {}),
                                   ([], {"WAYLAND_DISPLAY": "wayland-outer", "DISPLAY": ":0"})]:
                args.host_arg = host_args
                self.assertEqual(launcher.launch_plan(args, env)[1]["GAMESCOPE_FRAMEGEN_BIDIR"], "1")

    def test_inactive_preferences_are_not_enabled_in_the_renderer(self):
        with tempfile.TemporaryDirectory(prefix="vapor-inactive-", dir="/tmp") as work:
            folder = Path(work) / "game"
            folder.mkdir()
            preset = folder / "default.ini"
            args = argparse.Namespace(game="game", profile="default", profile_root=work,
                                      net_profile=None, gamescope="/host/gamescope",
                                      host_arg=[], command=["game"])
            for content in ["mode=blend\n", "mode=extrapolate\n", "pipeline=warp\n", "pipeline=checked\n"]:
                preset.write_text("version=1\n" + content)
                command, env, *_ = launcher.launch_plan(args, {"GAMESCOPE_FRAMEGEN_NET_ONLINE": "1"})
                self.assertEqual(env["GAMESCOPE_FRAMEGEN_NET_ONLINE"], "0")
                if "mode=" in content:
                    self.assertNotIn("--framegen-pipeline", command)
            preset.write_text("version=1\nframegen=off\nbidir=on\nonline=on\nhud=2\n")
            args.host_arg = ["--backend=headless"]
            command, env, *_ = launcher.launch_plan(args, {})
            self.assertNotIn("--experimental-framegen", command)
            for key in ("NET_ONLINE", "BIDIR", "HUD"):
                self.assertEqual(env["GAMESCOPE_FRAMEGEN_"+key], "0")

    def test_rejected_training_files_are_never_replaced(self):
        with tempfile.TemporaryDirectory(prefix="vapor-invalid-net-", dir="/tmp") as work:
            folder = Path(work) / "game"
            folder.mkdir()
            weights = folder / "learned.bin"
            (folder / "default.ini").write_text(f"version=1\npipeline=spatial\nnet_profile={weights}\n")
            args = argparse.Namespace(game="game", profile="default", profile_root=work,
                                      net_profile=None, gamescope="/host/gamescope",
                                      host_arg=[], command=["game"])
            metadata = struct.pack("<12I", 0x52465347, 3, 3, 12, 16, 3, 16, 16, 3, 16, 4, 3)
            blob = metadata + bytes(4644*4)
            valid = struct.pack("<4I", 0x50465347, 1, 1, 0x100f0) + blob
            nonfinite = valid[:-4] + struct.pack("<f", float("nan"))
            duplicate = struct.pack("<3I", 0x50465347, 1, 2) + (struct.pack("<I", 0xf0) + blob)*2
            for data in [b"", b"legacy weights", blob, valid[:-1], nonfinite, duplicate]:
                weights.write_bytes(data)
                with self.subTest(size=len(data)), self.assertRaises(ValueError):
                    launcher.launch_plan(args, {})
                self.assertEqual(weights.read_bytes(), data)
            weights.write_bytes(valid)
            launcher.launch_plan(args, {})
            self.assertEqual(weights.read_bytes(), valid)

    def test_host_probe_and_dry_run_reject_unsupported_pipeline(self):
        with tempfile.TemporaryDirectory(prefix="vapor-host-", dir="/tmp") as work:
            root = Path(work)
            host = root / "gamescope"
            host.write_text("#!/bin/sh\n[ \"$1\" = --help ] || exit 91\n"
                            "echo \"  --framegen-pipeline NAME 'warp', 'learned'\"\n")
            host.chmod(0o700)
            self.assertEqual(launcher.host_pipelines(str(host), os.environ), ["warp", "learned"])
            command = ["python3", str(ROOT / "scripts/vapor-profile.py"), "--game", "game",
                       "--profile-root", str(root / "profiles"), "--gamescope", str(host),
                       "--dry-run", "--", "game", "--framegen-pipeline", "game-own-option"]
            # Existing supported Learned default; dry-run writes nothing.
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "profiles").exists())
            folder = root / "profiles/game"
            folder.mkdir(parents=True)
            (folder / "default.ini").write_text("version=1\npipeline=spatial\n")
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("does not support", result.stderr)
            self.assertFalse((folder / "learned.bin").exists())


if __name__ == "__main__":
    unittest.main()
