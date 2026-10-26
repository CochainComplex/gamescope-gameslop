#!/usr/bin/env python3
"""Launch Gamescope with a saved per-game preset and its native menu."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import struct
import sys

CHOICES = {
    "framegen": ("on", "off"), "mode": ("motion", "extrapolate", "blend"),
    "pipeline": ("warp", "checked", "learned", "predict", "guided", "spatial"),
    "multiplier": ("2", "3", "4"), "online": ("on", "off"),
    "training_device": ("gpu", "npu"),
    "bidir": ("off", "on"), "hud": ("1", "2", "0"),
    "corner": ("br", "bl", "tr", "tl", "bc"),
    "style": ("quiet", "mallsoft", "cga"),
    "hud_opacity": ("auto", "0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"),
    "hud_scale": ("auto", "1", "2", "3", "4", "5", "6"),
}
DEFAULTS = {key: values[0] for key, values in CHOICES.items()}
DEFAULTS["pipeline"] = "learned"
DEFAULTS["net_profile"] = ""


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", value) or value in (".", ".."):
        raise ValueError("Game and preset names must use letters, numbers, '.', '_' or '-'")
    return value


def uses_net(settings):
    return (settings["framegen"] == "on" and settings["mode"] == "motion"
            and settings["pipeline"] in ("learned", "predict", "guided", "spatial"))


def validate(settings, context=None):
    context = context or {}
    if settings["framegen"] == "off":
        return  # Inactive preferences are remembered, never enabled in the host.
    if context.get("backend") in ("headless", "openvr"):
        raise ValueError("This backend cannot present generated frames")
    if context.get("orientation", "normal") != "normal":
        raise ValueError("Output rotation makes frame generation dormant")
    if context.get("outputs", "SingleApplication") not in ("SingleApplication", "SteamControlled"):
        raise ValueError("Frame generation requires a single-output connector strategy")
    if settings["bidir"] == "on":
        if settings["mode"] != "motion":
            raise ValueError("Bidirectional needs active motion mode")
        if settings["pipeline"] == "spatial":
            raise ValueError("Spatial is causal only")
        if context.get("backend") == "sdl":
            raise ValueError("SDL cannot present bidirectional frames")
        if context.get("base"):
            raise ValueError("Base-layer scenes use causal generation; disable BASE for bidirectional")
        if context.get("vrr_hybrid"):
            raise ValueError("VRR hybrid requires causal generation")


def environment_bool(environment, key):
    value = environment.get(key, "0")
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean setting: {key}")


def launch_context(host_args, environment):
    managed = ("--experimental-framegen", "--framegen-mode", "--framegen-pipeline",
               "--framegen-multiplier")
    observed = {}
    valued = ("--backend", "--force-orientation", "--virtual-connector-strategy")
    index = 0
    while index < len(host_args):
        arg = host_args[index]
        option, equal, value = arg.partition("=")
        if arg == "--" or (option.startswith("--") and
                           any(name.startswith(option) for name in managed)):
            raise ValueError(f"{option} cannot override the preset through --host-arg")
        if option.startswith("--") and any(name.startswith(option) for name in valued):
            if option not in valued:
                raise ValueError(f"Use the full host option name: {option}")
            if not equal:
                index += 1
                if index >= len(host_args):
                    raise ValueError(f"Missing value for {option}")
                value = host_args[index]
            if option in observed:
                raise ValueError(f"Duplicate host option: {option}")
            observed[option] = value
        index += 1
    backend = observed.get("--backend", "auto")
    if backend not in ("auto", "drm", "wayland", "sdl", "headless", "openvr"):
        raise ValueError(f"Invalid backend: {backend}")
    if backend == "auto":
        backend = "wayland" if "WAYLAND_DISPLAY" in environment else "sdl" if "DISPLAY" in environment else "drm"
    return {"backend": backend,
            "base": environment_bool(environment, "GAMESCOPE_FRAMEGEN_BASE"),
            "vrr_hybrid": environment_bool(environment, "GAMESCOPE_FRAMEGEN_VRR_HYBRID"),
            "orientation": observed.get("--force-orientation", "normal"),
            "outputs": observed.get("--virtual-connector-strategy", "SingleApplication")}


def validate_net_profile(path):
    """Refuse files the renderer would reject and replace on its first save.

    Mirrors decode_net_profile; cross-reader tests use actual C++-encoded files.
    Missing files are valid new profiles. Existing files are never migrated.
    """
    if not path.exists():
        return
    error = f"{path}: incompatible training profile; choose a valid GSFP file or a new path"
    if not path.is_file() or path.stat().st_size > 149036:
        raise ValueError(error)
    with path.open("rb") as source:
        data = source.read(149037)
    if len(data) < 12:
        raise ValueError(error)
    magic, version, count = struct.unpack_from("<3I", data)
    if (magic, version) != (0x50465347, 1) or not 1 <= count <= 8 or len(data) != 12 + count * 18628:
        raise ValueError(error)
    keys = set()
    metadata = (0x52465347, 3, 3, 12, 16, 3, 16, 16, 3, 16, 4, 3)
    for index in range(count):
        offset = 12 + index * 18628
        key = struct.unpack_from("<I", data, offset)[0]
        if key in keys or struct.unpack_from("<12I", data, offset + 4) != metadata:
            raise ValueError(error)
        keys.add(key)
        if not all(math.isfinite(weight) for weight in struct.unpack_from("<4644f", data, offset + 52)):
            raise ValueError(error)


def load(path):
    settings = DEFAULTS.copy()
    if not path.exists():
        if path.stem == "spatial-experimental":
            settings["pipeline"] = "spatial"
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
    validate(settings)
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
    if settings["training_device"] != "gpu":
        raise ValueError("NPU training unavailable: current companion supports offline inference only")
    context = launch_context(args.host_arg, environment)
    validate(settings, context)
    env = environment.copy()
    net = args.net_profile or settings["net_profile"] or env.get("GAMESCOPE_FRAMEGEN_NET_PROFILE")
    if not net:
        net = str(root / game / "learned.bin")
    net = str(Path(net).expanduser().resolve())
    if uses_net(settings):
        validate_net_profile(Path(net))
    env.update(VAPOR_GAME=game, VAPOR_PROFILE=profile, VAPOR_PROFILE_ROOT=str(root),
               VAPOR_BACKEND=context["backend"], VAPOR_BASE=str(int(context["base"])),
               VAPOR_VRR_HYBRID=str(int(context["vrr_hybrid"])),
               VAPOR_ORIENTATION=context["orientation"], VAPOR_OUTPUTS=context["outputs"],
               GAMESCOPE_FRAMEGEN_NET_PROFILE=net,
               GAMESCOPE_FRAMEGEN_NET_ONLINE=str(int(uses_net(settings) and settings["online"] == "on")),
               GAMESCOPE_FRAMEGEN_BIDIR=str(int(settings["framegen"] == "on" and settings["bidir"] == "on")),
               GAMESCOPE_FRAMEGEN_HUD=settings["hud"] if settings["framegen"] == "on" else "0",
               GAMESCOPE_FRAMEGEN_HUD_STYLE=settings["style"],
               GAMESCOPE_FRAMEGEN_HUD_CORNER=settings["corner"])
    # The menu's GPU DETAILS page reads the opt-in diagnostics global, so the
    # launcher enables it by default. While it is on, the game can bind it too;
    # an explicit GAMESCOPE_GPU_DIAGNOSTICS from the user (e.g. 0) is kept.
    env.setdefault("GAMESCOPE_GPU_DIAGNOSTICS", "1")
    for field, variable in (("hud_opacity", "GAMESCOPE_FRAMEGEN_HUD_OPACITY"),
                            ("hud_scale", "GAMESCOPE_FRAMEGEN_HUD_SCALE")):
        if settings[field] == "auto":
            env.pop(variable, None)
        else:
            env[variable] = settings[field]
    # This launcher trusts the host's RUNPATH. An inherited Lutris runtime or
    # legacy dependency prefix must not override a rebased compositor's libs.
    game_library_path = env.pop("LD_LIBRARY_PATH", None)
    if os.path.sep in args.gamescope:
        env["PATH"] = str(Path(args.gamescope).resolve().parent) + os.pathsep + env.get("PATH", os.defpath)
    command = [args.gamescope, "--expose-wayland"]
    if settings["framegen"] == "on":
        command += ["--experimental-framegen", "--framegen-mode", settings["mode"],
                    "--framegen-multiplier", settings["multiplier"]]
        if settings["mode"] == "motion":
            command += ["--framegen-pipeline", settings["pipeline"]]
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


def host_pipelines(binary, environment):
    # --help exits before Vulkan/Wayland initialization. Old builds must not
    # receive newly added pipelines, nor advertise them in their companion.
    result = subprocess.run([binary, "--help"], env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=5, check=True)
    row = next((line for line in result.stdout.splitlines() if "--framegen-pipeline " in line), "")
    supported = [name for name in re.findall(r"'([a-z]+)'", row) if name in CHOICES["pipeline"]]
    if not supported:
        raise ValueError("Selected host does not advertise supported framegen pipelines")
    return supported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, help="Stable per-game name, e.g. farcry6")
    parser.add_argument("--profile", help="Preset name; defaults to the last one saved for this game")
    parser.add_argument("--profile-root", help="Override the XDG settings directory")
    parser.add_argument("--net-profile", help="Existing learned .bin file to use")
    parser.add_argument("--gamescope", default=default_gamescope(os.environ))
    parser.add_argument("--host-arg", action="append", default=[], help="Extra Gamescope argument; use --host-arg=-f")
    parser.add_argument("--dry-run", action="store_true", help="Check host support and print the plan without starting a compositor/game or writing files")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command.pop(0)
    if not args.command:
        parser.error("Provide a game command after --")
    try:
        command, env, settings_dir, learned_dir = launch_plan(args, os.environ)
        supported = host_pipelines(command[0], env)
        host_command = command[:command.index("--")]
        if "--framegen-pipeline" in host_command and host_command[host_command.index("--framegen-pipeline")+1] not in supported:
            raise ValueError("Selected host does not support the preset pipeline")
        env["VAPOR_PIPELINES"] = ",".join(supported)
        if args.dry_run:
            print(json.dumps({"command": command, "settings": {k: v for k, v in env.items()
                  if k.startswith(("VAPOR_", "GAMESCOPE_FRAMEGEN_"))}}, indent=2))
            return 0
        # Build before starting the compositor so opening the menu is immediate.
        subprocess.run(["bash", str(Path(__file__).with_name("vapor-menu.sh")), "--build"], check=True)
        settings_dir.mkdir(parents=True, exist_ok=True)
        learned_dir.mkdir(parents=True, exist_ok=True)
        os.execvpe(command[0], command, env)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"vapor-profile: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
