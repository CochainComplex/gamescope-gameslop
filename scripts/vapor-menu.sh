#!/usr/bin/env bash
# Build and run the optional native Wayland companion independently of gamescope.
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
menu_build="$repo_dir/build-vapor-menu"
mkdir -p "$menu_build"
build_menu() {
    local spec stem
    local -a wayland_flags link_flags
    read -r -a wayland_flags <<< "$(pkg-config --cflags wayland-client)"
    read -r -a link_flags <<< "$(pkg-config --cflags --libs wayland-client wayland-egl egl glesv2 xkbcommon)"
    for spec in 'wlr-layer-shell-unstable-v1.xml:layer-shell' 'gamescope-action-binding.xml:action-binding' 'gamescope-control.xml:control'; do
        stem=${spec#*:}
        wayland-scanner client-header "$repo_dir/protocol/${spec%:*}" "$menu_build/$stem-client.h"
        wayland-scanner private-code "$repo_dir/protocol/${spec%:*}" "$menu_build/$stem-protocol.c"
        "${CC:-cc}" -c "$menu_build/$stem-protocol.c" -o "$menu_build/$stem-protocol.o" "${wayland_flags[@]}"
    done
    wayland-scanner private-code "$(pkg-config --variable=pkgdatadir wayland-protocols)/stable/xdg-shell/xdg-shell.xml" "$menu_build/xdg-shell-protocol.c"
    "${CC:-cc}" -c "$menu_build/xdg-shell-protocol.c" -o "$menu_build/xdg-shell-protocol.o" "${wayland_flags[@]}"
    "${CXX:-c++}" -std=c++20 -O2 -Wall -Wextra -Wpedantic \
        -I"$menu_build" -I"$repo_dir/src" "$repo_dir/tools/vapor-menu/main.cpp" \
        "$menu_build/"*-protocol.o -o "$menu_build/vapor-menu" \
        "${link_flags[@]}"
}
if [[ ! -x "$menu_build/vapor-menu" || "$repo_dir/tools/vapor-menu/main.cpp" -nt "$menu_build/vapor-menu" || "$repo_dir/tools/vapor-menu/profiles.hpp" -nt "$menu_build/vapor-menu" || "$repo_dir/src/framegen/hud.hpp" -nt "$menu_build/vapor-menu" || "${BASH_SOURCE[0]}" -nt "$menu_build/vapor-menu" || "$repo_dir/protocol/gamescope-action-binding.xml" -nt "$menu_build/vapor-menu" || "$repo_dir/protocol/gamescope-control.xml" -nt "$menu_build/vapor-menu" || "$repo_dir/protocol/wlr-layer-shell-unstable-v1.xml" -nt "$menu_build/vapor-menu" || "${1:-}" == --build ]]; then
    build_menu
fi
if [[ "${1:-}" == --build ]]; then exit 0; fi
if [[ "${1:-}" == -- ]]; then
    shift
    if (($# == 0)); then echo 'Usage: vapor-menu.sh -- GAME [ARGS...]' >&2; exit 2; fi
    "$menu_build/vapor-menu" &
    menu_pid=$!
    trap 'kill "$menu_pid" 2>/dev/null || true; wait "$menu_pid" 2>/dev/null || true' EXIT
    "$@"
else
    exec "$menu_build/vapor-menu" "$@"
fi
