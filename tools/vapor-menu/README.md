# Vapor menu

An optional native Wayland menu for Gamescope: cyan and magenta pixel text,
a striped sunset, transparent panel, and static grid. It runs as a separate
process using Gamescope's layer-shell, action-binding and control protocols.
Rendering uses Wayland EGL buffers, with no X11 or shared-memory requirement.

Build independently (C++20 compiler, pkg-config, wayland-scanner,
wayland-protocols, Wayland client/EGL, EGL/GLES2, xkbcommon headers):

```sh
bash scripts/vapor-menu.sh --build
```

For per-game presets, launch the game through the profile launcher:

```sh
python3 scripts/vapor-profile.py --game farcry6 \
  --gamescope /path/to/patched/build/src/gamescope \
  --net-profile "$HOME/Devstuff/lutris-local/profiles/farcry6.bin" \
  --host-arg=-f -- your-game its-arguments
```

`--game` is a stable name you choose, not a process-name guess. Settings live
under `$XDG_CONFIG_HOME/gamescope/profiles/<game>/` (default
`~/.config/gamescope/profiles/<game>/`). Each named `.ini` preset references a
learned `.bin` file; presets can share it. Cloning a preset keeps that reference. If none is supplied, the default learned file is
`<game>/learned.bin`. Existing learned files are selectable from that file's
directory. The menu never edits, deletes, converts, or clones model weights.

Pipeline (Warp/Checked/Learned/Predict/Guided/Spatial), multiplier x2–x4, motion mode,
framegen, online training, bidirectional mode, learned profile, HUD level and
HUD position are editable. **Save Preset** writes atomically and remembers
that preset for the next launch. **Save As New Preset** creates `custom-N`;
use `--profile custom-N` to select a specific saved preset explicitly.
Unsaved drafts survive hiding/reopening; switching presets is blocked until
you save. `*` marks an unsaved preset. No running renderer settings change.

Spatial keeps its own trained-weight set **inside the selected `.bin` file**,
separate from the other pipelines' set. Selecting Spatial does not create a
second file or copy Learned's weights. Spatial is causal; a bidir request is
ignored by the renderer with a diagnostic.

GPU buffer sizes and learned state are initialized at launch. Keeping profile
edits for the next launch avoids a new renderer control protocol and unsafe
mid-frame buffer/profile replacement. The menu displays this directly.

Use `--dry-run` to inspect arguments/environment without starting or writing
anything. `--host-arg=VALUE` repeats for additional Gamescope arguments. The launcher prefers this checkout's `build/src/gamescope` when present (or
`GAMESCOPE_BUILD_DIR`), unless `--gamescope` / `GAMESCOPE_BINARY` selects a host.
The launcher removes inherited `LD_LIBRARY_PATH` and uses the host RUNPATH.

For a standalone companion, launch the wrapper **inside** Gamescope:

```sh
gamescope --expose-wayland -- bash /absolute/path/to/gamescope/scripts/vapor-menu.sh -- your-game its-arguments
```

The wrapper starts the menu, runs the game, and stops the menu when the game
exits. Alternatively, run `bash scripts/vapor-menu.sh` from an environment
whose `WAYLAND_DISPLAY` names Gamescope's own compositor socket. Connecting
to the outer desktop compositor is not supported.

- **Ctrl+Shift+F12** toggles the menu (either left or right modifiers).
- **Up/Down** selects; **Left/Right** adjusts; **Enter** activates.
- **Escape** resumes. Resume hides the panel; it does not pause/unpause games.
- Screenshot requests a base-plane PNG in `/tmp`, excluding the menu. The
  process prints the request and the compositor's completion event.
- Preset settings are scrolled with Up/Down; Left/Right or Enter edits a value.
- Save/clone and screenshot actions require Enter.

The compact HUD uses the same palette with thin edges and four small status
lines, positioned bottom right by default. Full diagnostics remain available
as HUD level 2. HDR presentation uses a linear-light blend in Rec.2020/PQ.
The menu does not expose HDR or refresh-rate controls. Add settings in the
validated profile schema and launcher instead of reaching into renderer state.
Navigation is keyboard-only. Other game
inputs continue to work while the menu is open; the game is not paused.

When hidden, the surface and EGL resources are destroyed and the event loop
blocks on Wayland input. Only the toggle binding remains armed. There is no
animation/render timer. The 250 ms one-shot rearm debounce is active only
after an action. The companion currently accounts for this checkout's inverted
`NO_BLOCK` implementation in `GamescopeActionBinding.h`; check that assumption
when porting to a newer Gamescope implementation.

The accompanying small native-focus fix in `steamcompmgr.cpp` clears stale
external-overlay pointers and ignores unmapped native overlays; the rebased
`wlserver.cpp` already marks the native window list dirty on destruction. **Build these fixes into
the host before using the companion**: an unpatched host can crash when a
layer is destroyed. Do not replace the known playable baseline to try this.

Preview without a compositor:

```sh
bash scripts/vapor-menu.sh --preview-ppm /tmp/vapor-menu.ppm
```

Headless lifecycle check with an explicitly selected host:

```sh
GAMESCOPE_MENU_TEST_BINARY=/path/to/patched/build/src/gamescope python3 tools/vapor-menu/smoke.py
```

The check creates a private runtime directory, opens/renders/navigates/hides/
reopens the panel, and reaps its process group. Add `--gdb` for a host backtrace.
The compositor must have GPU access. This check does not inject real keyboard
events and does not establish native-game, HDR, or framegen visual correctness.

Storage/launcher checks run without a compositor:

```sh
python3 tools/vapor-menu/test_profiles.py
```

Build the rebased host with `~/Devstuff/prefix-upstream` first in
`PKG_CONFIG_PATH`, as documented by its `rb.sh`/`build-deps.sh`. At runtime use
its RUNPATH; do not inject the old `~/Devstuff/prefix` with `LD_LIBRARY_PATH` or
`GAMESCOPE_LOCAL_PREFIX`. The menu build writes only `build-vapor-menu/` and
does not modify either dependency prefix or Gamescope's `build/`.
