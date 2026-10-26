#!/usr/bin/env python3
"""Execute the compositor's initial paint decision with controlled CPU inputs.

The extracted production switch precedes framegen arbitration. This fixture
checks decisions, throttle state and lazy calls, not GPU or display behavior.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def materialize(source_path, directory):
    source = source_path.read_text()
    for name, begin, end in (
        ('flip_type', 'enum class FlipType\n', 'FlipType eFlipType ='),
        ('decision', 'bool bShouldPaint = false;',
         '// State processed above can invalidate or replace the queue'),
    ):
        if source.count(begin) != 1 or source.count(end) != 1:
            raise ValueError('ambiguous/missing production fragment: ' + name)
        start, stop = source.index(begin), source.index(end)
        if stop <= start:
            raise ValueError('reversed production fragment: ' + name)
        line = source.count('\n', 0, start) + 1
        (directory / (name + '.inc')).write_text(
            f'#line {line} "{source_path}"\n' + source[start:stop] + '\n')
    (directory / 'fixture.cpp').write_text(FIXTURE)


def execute(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    materialize(source, directory)
    commands = []
    for name, flags in (('normal', ['-O0']), ('optimized', ['-O3', '-ffast-math'])):
        binary = directory / name
        command = shlex.split(os.environ.get('CXX', 'c++')) + [
            '-std=c++20', '-Wall', '-Wextra', '-Werror', '-UNDEBUG', *flags,
            '-I', str(directory), str(directory / 'fixture.cpp'), '-o', str(binary),
        ]
        subprocess.run(command, check=True, timeout=60)
        subprocess.run([str(binary)], check=True, timeout=30)
        commands.append(command)
    (directory / 'commands.json').write_text(json.dumps(commands, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = (args.source or Path(__file__).resolve().parents[1] /
              'src/steamcompmgr.cpp').resolve()
    if args.output:
        execute(source, args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='steamcompmgr-paint-') as name:
            execute(source, Path(name))


FIXTURE = r'''
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "flip_type.inc"

struct Input {
    FlipType type = FlipType::Normal;
    bool timer = false, vblank = false, real = false, overlay = false;
    bool canFlip = false, pending = false, exhausted = false, ignore = false;
    bool connected = false, force = false;
    int ignored = 0, cycles = 0;
    unsigned inFlight = 0;
    bool operator==(const Input &) const = default;
};
static Input input;
static uint64_t trace;
static void note(unsigned call) { trace = trace * 10 + call; }
struct Cycles {
    operator int() const { note(7); return input.cycles; }
} cv_adaptive_sync_overlay_cycles;
struct Ignore {
    operator bool() const { note(5); return input.ignore; }
} cv_adaptive_sync_ignore_overlay;
static bool vulkan_framegen_has_pending_generated_frame() {
    note(1); return input.pending;
}
struct Feedback {
    unsigned CurrentPresentsInFlight() const { note(6); return input.inFlight; }
};
struct Connector {
    Feedback PresentationFeedback() const { note(4); return {}; }
};
struct Backend {
    Connector connector;
    Connector *GetCurrentConnector() {
        note(3); return input.connected ? &connector : nullptr;
    }
} backend;
static Backend *GetBackend() { note(2); return &backend; }

struct Result {
    bool paint, generated;
    int ignored;
    uint64_t calls;
    bool operator==(const Result &) const = default;
};

static Result observe(const Input &value) {
    input = value;
    trace = 0;
    const auto eFlipType = value.type;
    const bool vblank = value.vblank, bOverlayRepaint = value.overlay;
    const bool bForceSyncFlip = value.force, bVRRCanFlip = value.canFlip;
    const bool bIsVBlankFromTimer = value.timer;
    const std::atomic<bool> hasRepaint{value.real};
    int nIgnoredOverlayRepaints = value.ignored;
    struct { bool exhausted; } repaintPlan{value.exhausted};
#include "decision.inc"
    assert(input == value);
    return {bShouldPaint, bFramegenGeneratedPaint, nIgnoredOverlayRepaints, trace};
}

// Independent contracts: normal paints need a display slot; async real paints
// do not. VRR admits real work or an eligible timer-driven overlay, then the
// connector's in-flight presents veto it. Only deferred overlays age here.
static Result expected(const Input &v) {
    if (v.type == FlipType::Normal)
        return {v.vblank && (v.real || v.overlay || v.force), false, v.ignored, 0};
    if (v.type == FlipType::Async)
        return {v.real, false, v.ignored + int(v.vblank && !v.real && v.overlay), 0};
    const bool realEligible = v.real && v.canFlip;
    const bool overlayCheck = v.timer && v.overlay;
    const bool overlayEligible = overlayCheck &&
        (v.pending || v.ignored >= v.cycles || v.exhausted);
    const bool age = overlayCheck && !overlayEligible && !realEligible && !v.ignore;
    uint64_t calls = 0;
    const auto append = [&calls](unsigned call) { calls = calls * 10 + call; };
    if (overlayCheck) {
        append(1);
        if (!v.pending) append(7);
        if (!overlayEligible && !realEligible) append(5);
    }
    append(2); append(3);
    if (v.connected) { append(2); append(3); append(4); append(6); }
    return {(realEligible || overlayEligible) && !(v.connected && v.inFlight),
        false, v.ignored + int(age), calls};
}

static uint64_t digest = 14695981039346656037ull;
static unsigned cases;
static void check(const Input &v) {
    const auto result = observe(v);
    assert(result == expected(v));
    // Stable value digest for differential replay, independent of struct padding.
    for (uint64_t value : {uint64_t(result.paint), uint64_t(result.generated),
            uint64_t(result.ignored), result.calls}) {
        digest ^= value; digest *= 1099511628211ull;
    }
    ++cases;
}

static void characterization() {
    // Includes negative/zero throttles and increment-safe integer boundaries.
    // The superset also covers flag combinations not produced by today's caller.
    for (auto type : {FlipType::Normal, FlipType::Async, FlipType::VRR})
        for (unsigned bits = 0; bits < 1024; ++bits)
            for (int ignored : {-1, 0, 1, std::numeric_limits<int>::max()-1})
                for (int cycles : {-1, 0, 2, std::numeric_limits<int>::max()})
                    for (unsigned flight : {0u, 1u, std::numeric_limits<unsigned>::max()})
                        check({type, bool(bits&1), bool(bits&2), bool(bits&4),
                            bool(bits&8), bool(bits&16), bool(bits&32), bool(bits&64),
                            bool(bits&128), bool(bits&256), bool(bits&512),
                            ignored, cycles, flight});

    Input v{.type=FlipType::VRR, .timer=true, .vblank=true, .overlay=true,
        .connected=true, .cycles=2};
    for (int step = 0; step < 4; ++step) {
        check(v); v.ignored = observe(v).ignored;
    }
    // A pending presentation veto occurs after overlay aging, even when no timer
    // fired. Clearing it permits recovery; ignored overlays keep their age.
    v.ignored = 0; v.inFlight = 1;
    check(v); v.ignored = observe(v).ignored;
    v.timer = false; check(v);
    v.timer = true; v.inFlight = 0; check(v);
    v.pending = true; check(v);
    v.pending = false; v.exhausted = true; check(v);
    v.exhausted = false; v.ignore = true; check(v);
    v.real = true; v.canFlip = true; check(v);
    v.type = FlipType::Async; check(v);
    v.type = FlipType::Normal; check(v);
}

static void benchmark() {
    std::array<Input, 6> inputs{{
        {.type=FlipType::Normal, .vblank=true, .real=true},
        {.type=FlipType::Async, .vblank=true, .overlay=true},
        {.type=FlipType::VRR, .real=true, .canFlip=true, .connected=true},
        {.type=FlipType::VRR, .timer=true, .overlay=true, .connected=true, .cycles=2},
        {.type=FlipType::VRR, .timer=true, .overlay=true, .pending=true, .connected=true},
        {.type=FlipType::VRR, .timer=true, .overlay=true, .connected=true, .inFlight=1},
    }};
    uint64_t sink = 0;
    for (unsigned path = 0; path < inputs.size(); ++path) {
        for (unsigned i = 0; i < 2048; ++i) sink ^= observe(inputs[path]).calls;
        for (unsigned batch = 0; batch < 101; ++batch) {
            auto start = std::chrono::steady_clock::now();
            for (unsigned i = 0; i < 4096; ++i) {
                auto result = observe(inputs[path]);
                asm volatile("" : : "g"(&result) : "memory");
                sink ^= result.calls;
            }
            double ns = std::chrono::duration<double, std::nano>(
                std::chrono::steady_clock::now()-start).count()/4096;
            std::printf("%u,%u,%.6f\n", path, batch, ns);
        }
    }
    asm volatile("" : : "g"(sink) : "memory");
}

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
        benchmark(); return 0;
    }
    characterization();
    std::printf("cases=%u digest=%016llx\n", cases, (unsigned long long)digest);
}
'''


if __name__ == '__main__':
    main()
