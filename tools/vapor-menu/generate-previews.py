#!/usr/bin/env python3
"""Render every selectable menu style and HUD layout into the previews folder."""
from pathlib import Path
import subprocess
import shutil
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "previews"


def main():
    subprocess.run(["bash", str(ROOT / "scripts/vapor-menu.sh"), "--build"], check=True)
    binary = ROOT / "build-vapor-menu/vapor-menu"
    OUTPUT.mkdir(exist_ok=True)
    modes = {
        "quiet": "--preview-ppm",
        "mallsoft": "--preview-mallsoft-ppm",
        "cga": "--preview-cga-ppm",
        "experiments": "--preview-experimental-ppm",
        "gpu-details": "--preview-gpu-details-ppm",
        "gpu-details-cga": "--preview-gpu-details-cga-ppm",
        "gpu-details-mallsoft": "--preview-gpu-details-mallsoft-ppm",
        "gpu-unknown": "--preview-gpu-unknown-ppm",
        "hud-cga-compact": "--preview-hud-cga-ppm",
        "hud-cga-detailed": "--preview-hud-cga-details-ppm",
        "hud-cga-clear": "--preview-hud-cga-clear-ppm",
        "hud-cga-opaque": "--preview-hud-cga-opaque-ppm",
        "hud-cga-large": "--preview-hud-cga-large-ppm",
        "hud-compact": "--preview-hud-ppm",
        "hud-detailed": "--preview-hud-details-ppm",
        "hud-off": "--preview-hud-off-ppm",
    }
    with tempfile.TemporaryDirectory(prefix="vapor-previews-") as temporary:
        for name, flag in modes.items():
            source = Path(temporary) / f"{name}.ppm"
            subprocess.run([str(binary), flag, str(source)], check=True)
            with Image.open(source) as rendered:
                rendered.save(OUTPUT / f"{name}.png", optimize=True)
            print(OUTPUT / f"{name}.png")
    # Keep links from earlier reviews pointing at the latest rendered artwork.
    for filename, preview in {
        "preview.png": "quiet.png", "settings-preview.png": "quiet.png",
        "hud-preview.png": "hud-compact.png", "cga-preview.png": "cga.png",
        "hud-cga-preview.png": "hud-cga-compact.png",
    }.items():
        shutil.copyfile(OUTPUT / preview, ROOT / "build-vapor-menu" / filename)


if __name__ == "__main__":
    main()
