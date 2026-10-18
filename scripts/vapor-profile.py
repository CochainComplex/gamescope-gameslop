#!/usr/bin/env python3
"""Launch Gamescope with a saved per-game preset and its native menu."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

CHOICES = {
    "framegen": ("on", "off"), "mode": ("motion", "extrapolate", "blend"),
    "pipeline": ("warp", "checked", "learned", "predict", "guided", "spatial"),
    "multiplier": ("2", "3", "4"), "online": ("on", "off"),
    "bidir": ("off", "on"), "hud": ("1", "2", "0"),
    "corner": ("br", "bl", "tr", "tl", "bc"),
}
DEFAULTS = {key: values[0] for key, values in CHOICES.items()}
DEFAULTS["pipeline"] = "learned"
DEFAULTS["net_profile"] = ""


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", value) or value in (".", ".."):
        raise ValueError("Game and preset names must use letters, numbers, '.', '_' or '-'")
    return value


def load(path):
    settings = DEFAULTS.copy()
    if not path.exists():
        return settings
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8192:
        raise ValueError("Preset must be a small regular file")
    seen = set()
    for line in path.read_bytes().decode("utf-8").split("\n"):
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key in seen:
            raise ValueError("Invalid or duplicate preset field")
        seen.add(key)
        if key == "version":
            if value != "1":
                raise ValueError("Unsupported preset version")
        elif key == "net_profile":
            if len(value.encode()) > 4096 or any(c in value for c in "\r\n\0"):
                raise ValueError("Invalid learned-profile path")
            settings[key] = value
        elif key not in CHOICES or value not in CHOICES[key]:
            raise ValueError(f"Invalid preset setting: {key}")
        else:
            settings[key] = value
    if "version" not in seen:
        raise ValueError("Missing preset version")
    return settings


def launch_plan(args, environment):
    game = identifier(args.game)
    config = Path(environment.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    root = Path(args.profile_root or config / "gamescope/profiles").expanduser().resolve()
    active = root / game / ".active"
    profile = args.profile
    if profile is None and active.exists():
        if active.is_symlink() or not active.is_file() or active.stat().st_size > 65:
            raise ValueError("Invalid active preset")
        profile = active.read_text().removesuffix("\n")
    profile = identifier(profile or "default")
    settings = load(root / game / f"{profile}.ini")
    env = environment.copy()
    net = args.net_profile or settings["net_profile"] or env.get("GAMESCOPE_FRAMEGEN_NET_PROFILE")
    if not net:
        net = str(root / game / "learned.bin")
    net = str(Path(net).expanduser().resolve())
    env.update(VAPOR_GAME=game, VAPOR_PROFILE=profile, VAPOR_PROFILE_ROOT=str(root),
               GAMESCOPE_FRAMEGEN_NET_PROFILE=net,
               GAMESCOPE_FRAMEGEN_NET_ONLINE=str(int(settings["online"] == "on")),
               GAMESCOPE_FRAMEGEN_BIDIR=str(int(settings["bidir"] == "on")),
               GAMESCOPE_FRAMEGEN_HUD=settings["hud"],
               GAMESCOPE_FRAMEGEN_HUD_CORNER=settings["corner"])
    # This launcher trusts the host's RUNPATH. An inherited Lutris runtime or
    # legacy dependency prefix must not override a rebased compositor's libs.
    game_library_path = env.pop("LD_LIBRARY_PATH", None)
    if os.path.sep in args.gamescope:
        env["PATH"] = str(Path(args.gamescope).resolve().parent) + os.pathsep + env.get("PATH", os.defpath)
    command = [args.gamescope, "--expose-wayland"]
    if settings["framegen"] == "on":
        command += ["--experimental-framegen", "--framegen-mode", settings["mode"],
                    "--framegen-pipeline", settings["pipeline"], "--framegen-multiplier", settings["multiplier"]]
    command += args.host_arg
    wrapper = Path(__file__).resolve().parent / "vapor-menu.sh"
    game_command = args.command
    if game_library_path is not None:
        game_command = ["env", "LD_LIBRARY_PATH=" + game_library_path, *game_command]
    command += ["--", "bash", str(wrapper), "--", *game_command]
    return command, env, root / game, Path(net).parent


def default_gamescope(environment):
    if environment.get("GAMESCOPE_BINARY"):
        return environment["GAMESCOPE_BINARY"]
    root = Path(__file__).resolve().parents[1]
    build = Path(environment.get("GAMESCOPE_BUILD_DIR", "build"))
    candidate = root / build / "src/gamescope"
    return str(candidate) if os.access(candidate, os.X_OK) else "gamescope"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, help="Stable per-game name, e.g. farcry6")
    parser.add_argument("--profile", help="Preset name; defaults to the last one saved for this game")
    parser.add_argument("--profile-root", help="Override the XDG settings directory")
    parser.add_argument("--net-profile", help="Existing learned .bin file to use")
    parser.add_argument("--gamescope", default=default_gamescope(os.environ))
    parser.add_argument("--host-arg", action="append", default=[], help="Extra Gamescope argument; use --host-arg=-f")
    parser.add_argument("--dry-run", action="store_true", help="Print the launch plan without starting or writing files")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    if not args.command:
        parser.error("Provide a game command after --")
    try:
        command, env, settings_dir, learned_dir = launch_plan(args, os.environ)
        if args.dry_run:
            print(json.dumps({"command": command, "settings": {k: v for k, v in env.items()
                  if k.startswith(("VAPOR_", "GAMESCOPE_FRAMEGEN_"))}}, indent=2))
            return 0
        # Build before starting the compositor so opening the menu is immediate.
        subprocess.run(["bash", str(Path(__file__).with_name("vapor-menu.sh")), "--build"], check=True)
        settings_dir.mkdir(parents=True, exist_ok=True)
        learned_dir.mkdir(parents=True, exist_ok=True)
        os.execvpe(command[0], command, env)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"vapor-profile: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
