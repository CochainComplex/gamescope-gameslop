# gamescope-gameslop settings menu

An optional native Wayland menu for Gamescope: faded cyan and mauve pixel text,
a spacious header, thin edges and a translucent panel. It runs as a separate
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
`<game>/learned.bin`. Existing compatible GSFP learned files are selectable from that file's
directory. Legacy, malformed and non-finite profiles are refused before launch
or save; they are never converted or replaced by the menu. The menu never edits, deletes, converts, or clones model weights.

Pipeline (Warp/Checked/Learned/Predict/Guided/Spatial), multiplier x2–x4, motion mode,
framegen, online training, bidirectional mode, learned profile, HUD level and
HUD position are editable. **Save Preset** writes atomically and remembers
that preset for the next launch. **Save As New Preset** creates `custom-N`;
use `--profile custom-N` to select a specific saved preset explicitly.
Unsaved drafts survive hiding/reopening; switching presets is blocked until
you save. `*` marks an unsaved preset. No running renderer settings change.

Spatial keeps its own trained-weight set **inside the selected `.bin` file**,
separate from the other pipelines' set. Selecting Spatial does not create a
second file or copy Learned's weights. Spatial is causal; the menu and launcher block bidir for it. A raw renderer
command still ignores such a request with a diagnostic.

GPU buffer sizes and learned state are initialized at launch. Keeping profile
edits for the next launch avoids a new renderer control protocol and unsafe
mid-frame buffer/profile replacement. The menu displays this directly.

Use `--dry-run` to check the selected host and inspect arguments/environment
without starting a compositor/game or writing files. It runs the host with
`--help` to discover its advertised pipelines. `--host-arg=VALUE` repeats for additional Gamescope arguments. The launcher prefers this checkout's `build/src/gamescope` when present (or
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
lines plus the original small deadline-hit and jitter graphs, positioned bottom right by default. Full diagnostics remain available
as HUD level 2. HDR presentation uses a linear-light blend in Rec.2020/PQ.
The menu does not expose HDR or refresh-rate controls. Add settings in the
validated profile schema and launcher instead of reaching into renderer state.
**Menu Style** selects Quiet (the current dark default), Mallsoft (washed
stone/teal/rose), or CGA (cyan-to-magenta ordered pixel dithering). The companion
previews immediately; Save Preset remembers it. HUD style is applied on the next
launch through `GAMESCOPE_FRAMEGEN_HUD_STYLE`; CGA preserves the small graphs
and uses white lettering with a crisp black outline and wider row spacing on a
translucent dithered background. Quiet and Mallsoft retain the current muted HUD. The HUD reuses
its existing uniform buffer and pass; no new runtime control protocol is added. See [all style/HUD previews](previews/README.md)
and the short research references there.

**HUD BG Opacity** controls the background independently of the lettering: Auto
uses 14% for Quiet/Mallsoft and 36% for CGA; explicit values range from 0%
(clear) to 100% (solid) in 10% steps. **HUD Text Size** offers Auto or 1–6×.
Both rows, like HUD Position, are inactive while the HUD is off. CGA lettering
always has a black pixel outline; Quiet/Mallsoft lettering gets the same outline
at 0% (below 10% backing), so a clear HUD stays readable over bright scenes.
Save Preset remembers both for the next launch. The launcher exports
`GAMESCOPE_FRAMEGEN_HUD_OPACITY` and `GAMESCOPE_FRAMEGEN_HUD_SCALE`; Auto
clears those overrides.

All eighteen controls are visible together in a two-column layout: frame
generation on the left, HUD/appearance on the right, preset above and actions
below. Integer-scaled text fits a 608×340 logical panel (2× at 1280×720).
Up/Down follows the visual order; Left/Right edits; Enter activates actions;
Tab jumps between preset, frame generation, HUD and actions. No scrolling.
CGA retains its dithered margins with dark text panels for contrast. The compact
HUD retains its graphs and shows the build checkout hash, including dirty state.

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

## Configuration dependencies

Unavailable controls show **inactive** and explain the reason when selected.
The launcher independently validates hand-edited presets before building the
menu, starting the compositor, or writing learned data.

| Setting | Requirement |
|---|---|
| Pipeline | Frame generation on, Motion mode, and a pipeline advertised by the selected host |
| Online training / learned-file selection | Motion with Learned, Predict, Guided or Spatial |
| Bidirectional | Motion, any pipeline except Spatial, DRM or nested Wayland; no inherited BASE or VRR_HYBRID request |
| HUD position | Frame generation and the HUD enabled |
| Frame generation | A presenting backend (DRM, Wayland or SDL), normal output orientation and SingleApplication/SteamControlled output strategy |

Changing mode/pipeline so bidir becomes incompatible turns bidir off with a
message. Parent settings that are inactive retain their preferences, but the
launcher sends training=0/bidir=0/HUD=0 while frame generation is off. Non-motion
modes omit the motion-pipeline argument; Warp/Checked never enable training.
Switching back restores applicable preferences. Selecting an older host skips
its unsupported pipelines; a saved unsupported active pipeline is rejected at
launch rather than silently changed.

The launcher derives backend context before Gamescope rewrites display
variables and passes it to the companion. SDL supports causal frame generation
but cannot present bidirectional output. Raw renderer commands can request a
mixed BASE/bidir policy (causal in base-layer scenes); this simpler menu refuses
that mixed policy so an enabled bidir switch means the requested presentation
variant can actually run. Extra host arguments cannot override preset-owned
framegen flags, abbreviate checked host options, or inject a command separator.

These are configuration checks, not a hardware guarantee. Startup scripts can
override the backend; actual display rotation/output state, available compute
memory/queues, buffer formats, allocation failures and timing pressure still
require the renderer's runtime checks and may cause dormancy or fallback. Use
the HUD/logs to inspect the effective mode. A missing frozen weight set leaves
the optional refiner inactive; it does not make the underlying pipeline invalid.

Experimental settings: **spatial-experimental** is a built-in next-launch preset
(Motion, Spatial, ×2, GPU online training, bidirectional off). It is offered
alongside saved presets; an existing file with that name takes precedence.
The preset is virtual until saved and reuses the game's existing learned-file
reference. Spatial keeps its own role/encoding weight set inside that file.
The integrated launcher probes host support; the legacy launcher also refuses
a Spatial launch on a host that does not advertise it. No old host is upgraded.
The pipeline selector marks Spatial **[EXPERIMENTAL]**.

**NPU TRAINING [EXPERIMENTAL]** shows a disabled switch beside GPU training.
Its explanation is “NPU training unavailable / offline inference only”. The
current [NPU companion](../../doc/framegen-npu-experiment.md) performs offline
inference and separate CPU training; it cannot replace GPU online gradients.
The schema reserves `training_device=gpu|npu` (default `gpu`), but saving or
launching an NPU selection is refused. No SDK probe, compilation or helper is
run at startup. This control is a visible availability indicator, not live NPU
offload. Training remains on the GPU.

**GPU DETAILS** opens a read-only page: the compositor/present GPU and the
inferred game-render GPU (name, PCI address, vendor:device, DRM render node,
Vulkan driver), the evidence for the inference, the buffer path and the build.
It reads the `gamescope_gpu_diagnostics_v1` Wayland global, which is opt-in:
gamescope advertises it only with `GAMESCOPE_GPU_DIAGNOSTICS=1`, because every
Wayland client, the game included, can bind it. `scripts/vapor-profile.py` sets
the variable for the gamescope it starts, so the launched game can read the same
data while that session runs. Without it the page shows "GPU diagnostics
disabled" and the menu keeps working.

HUD labels use muted cyan (canonical cyan in CGA); measured values use bright
white. Colons and pipe-delimited fields separate the readings, for example
`SOURCE: 40 fps | GENERATED: 78 fps | FILL: 98%`. Sparklines, checkout hash and
independent backing opacity remain. Label/value attributes fit inside the
existing packed character bytes; the uniform is 1648 bytes (eight lines).
