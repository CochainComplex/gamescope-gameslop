#!/usr/bin/env python3
"""Run the production SDL event loop against a scripted queue and action trace.

SDL types and scancode translation are real; window/Wayland effects are recorded.
This checks dispatch, ordering and cancellation, not a display or GPU runtime.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def definition(source, signature):
    if source.count(signature) != 1:
        raise ValueError('missing/ambiguous production definition: ' + signature)
    start = source.index(signature)
    begin = source.index('{', start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return start, source[start:end]


def materialize(source_path, directory):
    source = source_path.read_text()
    methods = ['void CSDLBackend::SDLEventLoop(',
               'uint32_t CSDLBackend::GetUserEventIndex(',
               'void CSDLBackend::PushUserEvent(']
    fragments = []
    for signature in methods:
        start, body = definition(source, signature)
        line = source.count('\n', 0, start) + 1
        fragments.append(f'#line {line} "{source_path}"\n' + body)
    (directory / 'production.inc').write_text('\n'.join(fragments) + '\n')
    (directory / 'fixture.cpp').write_text(FIXTURE)


def execute(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    materialize(source, directory)
    root = Path(__file__).resolve().parents[1]
    includes = shlex.split(subprocess.check_output(
        ['pkg-config', '--cflags', 'sdl2'], text=True))
    commands = []
    for name, flags in [('normal', ['-O0']), ('optimized', ['-O3', '-ffast-math'])]:
        binary = directory / name
        command = shlex.split(os.environ.get('CXX', 'c++')) + [
            '-std=c++20', '-Wall', '-Wextra', '-Werror',
            '-Wno-missing-field-initializers', '-UNDEBUG', *flags,
            *includes, '-I' + str(root / 'src'), '-I' + str(directory),
            str(directory / 'fixture.cpp'), '-o', str(binary),
        ]
        subprocess.run(command, check=True, timeout=60)
        subprocess.run([str(binary)], check=True, timeout=30)
        commands.append(command)
    (directory / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')


FIXTURE = r'''
#include <SDL.h>
#include <linux/input-event-codes.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <signal.h>
#include <stop_token>
#include <string>
#include <vector>
#include "sdlscancodetable.hpp"
#include "Utils/FaultInjection.h"

using std::string;
using std::vector;
static vector<string> actions;
static unsigned lockDepth, timestamp, expectedTimestamp;
static unsigned tests;
static void note(const string &value) { actions.push_back(value); }
static void wlserver_lock() { assert(lockDepth++ == 0); note("lock"); }
static void wlserver_unlock() { assert(--lockDepth == 0); note("unlock"); }
static void input(const char *kind, uint32_t time) {
    assert(lockDepth == 1 && time == timestamp); note(kind);
}
static void wlserver_key(uint32_t key, bool down, uint32_t time) {
    input(("key:" + std::to_string(key) + ":" + std::to_string(down)).c_str(), time);
}
static void wlserver_mousemotion(double, double, uint32_t t) { input("motion",t); }
static void wlserver_touchmotion(double, double, long long, uint32_t t) { input("touch",t); }
static void wlserver_touchdown(double, double, long long, uint32_t t) { input("down",t); }
static void wlserver_touchup(long long, uint32_t t) { input("up",t); }
static void wlserver_mousebutton(uint32_t, bool, uint32_t t) { input("button",t); }
static void wlserver_mousewheel(double, double, uint32_t t) { input("wheel",t); }
static void force_repaint() { assert(!lockDepth); note("paint"); }
enum Selection { GAMESCOPE_SELECTION_CLIPBOARD, GAMESCOPE_SELECTION_PRIMARY };
static void gamescope_set_selection(char *, Selection) { note("selection"); }
static bool g_bFullscreen, g_bGrabbed, g_bWindowFocused = true;
static int g_nOutputWidthPts=100, g_nOutputHeightPts=100, g_nOutputWidth=100, g_nOutputHeight=100;
static int g_nOutputRefresh=60000, g_nNestedRefresh=60000, g_nNestedUnfocusedRefresh=30000, g_nOldNestedRefresh=60000;
enum class GamescopeUpscaleFilter { PIXEL, LINEAR, FSR, NIS };
static GamescopeUpscaleFilter g_wantedUpscaleFilter;
static int g_upscaleFilterSharpness;
namespace gamescope {
struct CScreenshotManager {
    static CScreenshotManager &Get() { static CScreenshotManager value; return value; }
    void TakeScreenshot(bool value) { assert(value && !lockDepth); note("screenshot"); }
};
static int ConvertHztomHz(int hz) { return hz*1000; }
}
struct Logger {
    template<class... T> void infof(const char *, T...) { note("info"); }
    template<class... T> void warnf(const char *, T...) { note("warn"); }
    template<class... T> void errorf(const char *, T...) { note("error"); }
} sdl_log;
static uint64_t get_time_in_nanos() { return 2'000'000'000; }
static constexpr int k_nSDLRetryMs=100;
enum SDLCustomEvents { GAMESCOPE_SDL_EVENT_TITLE, GAMESCOPE_SDL_EVENT_ICON,
    GAMESCOPE_SDL_EVENT_VISIBLE, GAMESCOPE_SDL_EVENT_GRAB, GAMESCOPE_SDL_EVENT_CURSOR,
    GAMESCOPE_SDL_EVENT_WAKE };
struct Connector { SDL_Window *GetSDLWindow() { return nullptr; } };
struct CursorInfo { vector<uint32_t> pPixels; unsigned uWidth=1,uHeight=1,uXHotspot=0,uYHotspot=0; };
namespace INestedHints { using CursorInfo=::CursorInfo; }
static bool visibilityBusy;
struct CSDLBackend {
    void SDLEventLoop(std::stop_token);
    uint32_t GetUserEventIndex(SDLCustomEvents) const;
    void PushUserEvent(SDLCustomEvents);
    void UpdateWindowVisibility() { note("visible"); m_bVisibilityPending=visibilityBusy; }
    Connector m_Connector;
    bool m_bVisibilityPending=false, m_bShown=true, m_bApplicationGrabbed=false;
    std::atomic<bool> m_bWindowMapped{true};
    std::shared_ptr<string> m_pApplicationTitle;
    std::shared_ptr<vector<uint32_t>> m_pApplicationIcon;
    std::shared_ptr<CursorInfo> m_pApplicationCursor;
    SDL_Surface *m_pIconSurface=nullptr, *m_pCursorSurface=nullptr;
    SDL_Cursor *m_pCursor=nullptr;
    uint32_t m_uUserEventIdBase=SDL_USEREVENT;
};
static vector<SDL_Event> queue;
static size_t nextEvent;
static std::stop_source *stopSource;
static bool stopOnWait, failWait;
static unsigned failures;
static bool stopOnFailure;
static Uint32 eventSubsystem;
static string waitError, failureError;
extern "C" {
int SDL_WaitEvent(SDL_Event *event) {
    if (stopOnWait) { stopSource->request_stop(); stopOnWait=false; }
    if (failWait || failures) {
        failWait=false; if (failures) --failures;
        waitError=failureError;
        if (stopOnFailure) stopSource->request_stop();
        return 0;
    }
    assert(nextEvent<queue.size()); *event=queue[nextEvent++];
    if (!stopSource->stop_requested()) ++timestamp;
    return 1;
}
int SDL_WaitEventTimeout(SDL_Event *event, int ms) { assert(ms==100); note("timed"); return SDL_WaitEvent(event); }
void SDL_ClearError() { waitError.clear(); }
const char *SDL_GetError() { return waitError.c_str(); }
int SDL_SetError(const char *, ...) { waitError="injected"; return -1; }
Uint32 SDL_WasInit(Uint32 flags) { assert(flags==SDL_INIT_EVENTS); return eventSubsystem; }
void SDL_PumpEvents() { note("pump"); }
void SDL_Delay(Uint32 ms) { assert(ms==100); note("delay"); }
int SDL_PushEvent(SDL_Event *event) { assert(!lockDepth); note("push:"+std::to_string(event->type)); return 1; }
int SDL_SetWindowFullscreen(SDL_Window *, Uint32 flags) { assert(!lockDepth); note("fullscreen:"+std::to_string(flags)); return 0; }
void SDL_SetWindowKeyboardGrab(SDL_Window *, SDL_bool enabled) { assert(!lockDepth); note("grab:"+std::to_string(enabled)); }
char *SDL_GetClipboardText() { return nullptr; }
char *SDL_GetPrimarySelectionText() { return nullptr; }
void SDL_free(void *) {}
void SDL_GetWindowSize(SDL_Window *, int *w, int *h) { *w=200; *h=150; note("points"); }
void SDL_GetWindowSizeInPixels(SDL_Window *, int *w, int *h) { *w=400; *h=300; note("pixels"); }
int SDL_GetWindowDisplayIndex(SDL_Window *) { note("display"); return 0; }
int SDL_GetDesktopDisplayMode(int, SDL_DisplayMode *mode) { mode->refresh_rate=75; note("refresh"); return 0; }
void SDL_SetWindowTitle(SDL_Window *, const char *) { note("title"); }
SDL_bool SDL_SetHint(const char *, const char *) { note("hint"); return SDL_TRUE; }
void SDL_DisableScreenSaver() { note("screensaver"); }
void SDL_FreeSurface(SDL_Surface *) { note("free-surface"); }
SDL_Surface *SDL_CreateRGBSurfaceFrom(void *, int,int,int,int,Uint32,Uint32,Uint32,Uint32) { note("surface"); return nullptr; }
void SDL_SetWindowIcon(SDL_Window *, SDL_Surface *) { note("icon"); }
int SDL_SetRelativeMouseMode(SDL_bool) { note("relative"); return 0; }
void SDL_FreeCursor(SDL_Cursor *) { note("free-cursor"); }
SDL_Cursor *SDL_CreateColorCursor(SDL_Surface *,int,int) { note("cursor-create"); return nullptr; }
void SDL_SetCursor(SDL_Cursor *) { note("cursor"); }
int raise(int sig) noexcept { assert(sig==SIGTERM); note("term"); return 0; }
}
using namespace gamescope;
#include "production.inc"
static SDL_Event key(SDL_Scancode code, Uint32 type, Uint16 mod=0, Uint8 repeat=0) {
    SDL_Event e{}; e.type=type; e.key.keysym.scancode=code; e.key.keysym.mod=mod; e.key.repeat=repeat; return e;
}
static SDL_Event window(Uint8 kind) { SDL_Event e{}; e.type=SDL_WINDOWEVENT; e.window.event=kind; return e; }
static SDL_Event type(Uint32 kind) { SDL_Event e{}; e.type=kind; return e; }
static void run(CSDLBackend &backend, vector<SDL_Event> events, vector<string> expected) {
    const unsigned count=events.size();
    events.push_back(type(SDL_QUIT));
    queue=std::move(events); nextEvent=0; actions.clear();
    std::stop_source stop; stopSource=&stop;
    expectedTimestamp+=count+1;
    backend.SDLEventLoop(stop.get_token());
    expected.push_back("info");
    assert(actions==expected && lockDepth==0 && nextEvent==queue.size());
    assert(timestamp==expectedTimestamp); ++tests;
}
static vector<string> forwarded(unsigned code, bool down) {
    return {"lock", "key:"+std::to_string(code)+":"+std::to_string(down), "unlock"};
}
int main() {
    unsetenv("GAMESCOPE_DEBUG_SDL_WAIT_INJECT");
    CSDLBackend backend;
    const SDL_Scancode shortcuts[]={SDL_SCANCODE_F,SDL_SCANCODE_N,SDL_SCANCODE_B,
        SDL_SCANCODE_U,SDL_SCANCODE_Y,SDL_SCANCODE_I,SDL_SCANCODE_O,SDL_SCANCODE_S,SDL_SCANCODE_G};
    for (SDL_Scancode code : shortcuts) for (Uint16 mod : {Uint16(0),Uint16(KMOD_LGUI),Uint16(KMOD_RGUI),Uint16(KMOD_LGUI|KMOD_SHIFT)})
        for (Uint32 kind : {SDL_KEYDOWN,SDL_KEYUP}) for (Uint8 repeat : {Uint8(0),Uint8(1)}) {
            g_bFullscreen=false; g_bGrabbed=false;
            g_wantedUpscaleFilter=GamescopeUpscaleFilter::LINEAR; g_upscaleFilterSharpness=10;
            vector<string> expected;
            if (mod&KMOD_LGUI) {
                if (kind==SDL_KEYUP) {
                    if (code==SDL_SCANCODE_F) expected={"fullscreen:"+std::to_string(SDL_WINDOW_FULLSCREEN_DESKTOP)};
                    if (code==SDL_SCANCODE_S) expected={"screenshot"};
                    if (code==SDL_SCANCODE_G) expected={"grab:1","push:"+std::to_string(SDL_USEREVENT)};
                }
            } else if (!repeat) expected=forwarded(SDLScancodeToLinuxKey(code),kind==SDL_KEYDOWN);
            run(backend,{key(code,kind,mod,repeat)},expected);
            const bool action=(mod&KMOD_LGUI) && kind==SDL_KEYUP;
            assert(g_bFullscreen==(action && code==SDL_SCANCODE_F));
            assert(g_bGrabbed==(action && code==SDL_SCANCODE_G));
            const auto filter=code==SDL_SCANCODE_N ? GamescopeUpscaleFilter::PIXEL :
                code==SDL_SCANCODE_U ? GamescopeUpscaleFilter::FSR :
                code==SDL_SCANCODE_Y ? GamescopeUpscaleFilter::NIS : GamescopeUpscaleFilter::LINEAR;
            assert(g_wantedUpscaleFilter==(action ? filter : GamescopeUpscaleFilter::LINEAR));
            assert(g_upscaleFilterSharpness==10+int(action && code==SDL_SCANCODE_I)-int(action && code==SDL_SCANCODE_O));
            assert(!backend.m_bApplicationGrabbed);
        }
    for (Uint16 mod : {Uint16(0),Uint16(KMOD_LGUI),Uint16(KMOD_RGUI)})
        for (Uint32 kind : {SDL_KEYDOWN,SDL_KEYUP}) for (Uint8 repeat : {Uint8(0),Uint8(1)})
            run(backend,{key(SDL_SCANCODE_A,kind,mod,repeat)},repeat ? vector<string>{} : forwarded(KEY_A,kind==SDL_KEYDOWN));
    for (int sharpness : {-1,0,19,20,21}) for (auto code : {SDL_SCANCODE_I,SDL_SCANCODE_O}) {
        g_upscaleFilterSharpness=sharpness;
        run(backend,{key(code,SDL_KEYUP,KMOD_LGUI)},{});
        assert(g_upscaleFilterSharpness==(code==SDL_SCANCODE_I ? std::min(20,sharpness+1) : std::max(0,sharpness-1)));
    }
    for (auto code : {SDL_SCANCODE_U,SDL_SCANCODE_Y}) {
        g_wantedUpscaleFilter=code==SDL_SCANCODE_U ? GamescopeUpscaleFilter::FSR : GamescopeUpscaleFilter::NIS;
        run(backend,{key(code,SDL_KEYUP,KMOD_LGUI)},{});
        assert(g_wantedUpscaleFilter==GamescopeUpscaleFilter::LINEAR);
    }
    auto expected=forwarded(KEY_A,true); auto up=forwarded(KEY_A,false);
    expected.insert(expected.end(),up.begin(),up.end());
    run(backend,{type(SDL_USEREVENT+99),key(SDL_SCANCODE_A,SDL_KEYDOWN),key(SDL_SCANCODE_F,SDL_KEYDOWN,KMOD_LGUI),key(SDL_SCANCODE_A,SDL_KEYUP)},expected);
#if SDL_VERSION_ATLEAST(2,26,0)
    run(backend,{window(SDL_WINDOWEVENT_SIZE_CHANGED)},{"points","pixels","paint","display","refresh"});
    assert(g_nOutputWidth==400 && g_nOutputHeight==300);
#else
    run(backend,{window(SDL_WINDOWEVENT_SIZE_CHANGED)},{"points","paint","display","refresh"});
    assert(g_nOutputWidth==200 && g_nOutputHeight==150);
#endif
    assert(g_nOutputWidthPts==200 && g_nOutputHeightPts==150 && g_nOutputRefresh==75000);
    run(backend,{window(SDL_WINDOWEVENT_FOCUS_LOST),key(SDL_SCANCODE_A,SDL_KEYDOWN),window(SDL_WINDOWEVENT_FOCUS_GAINED)},forwarded(KEY_A,true));
    assert(g_bWindowFocused && g_nNestedRefresh==g_nOldNestedRefresh);
    backend.m_bShown=true; run(backend,{window(SDL_WINDOWEVENT_HIDDEN)},{}); assert(backend.m_bWindowMapped);
    backend.m_bShown=false; run(backend,{window(SDL_WINDOWEVENT_HIDDEN)},{}); assert(!backend.m_bWindowMapped);
    run(backend,{window(SDL_WINDOWEVENT_CLOSE),key(SDL_SCANCODE_A,SDL_KEYUP)},{"term","lock","key:"+std::to_string(KEY_A)+":0","unlock"});
    // QUIT leaves already queued keys untouched; a successful cancelled wait also
    // leaves the event undispatched and does not advance the input timestamp.
    queue={type(SDL_QUIT),key(SDL_SCANCODE_A,SDL_KEYDOWN)}; nextEvent=0; actions.clear();
    std::stop_source stop; stopSource=&stop; ++expectedTimestamp;
    backend.SDLEventLoop(stop.get_token()); assert(nextEvent==1 && actions==vector<string>{"info"});
    actions.clear(); nextEvent=0; stopOnWait=true; const auto previous=timestamp;
    backend.SDLEventLoop(stop.get_token()); assert(timestamp==previous && nextEvent==1 && actions==vector<string>{"push:"+std::to_string(Uint32(SDL_USEREVENT)+GAMESCOPE_SDL_EVENT_WAKE)});
    run(backend,{key(SDL_SCANCODE_A,SDL_KEYDOWN)},forwarded(KEY_A,true));
    stop=std::stop_source{}; stopSource=&stop; actions.clear(); nextEvent=0; failWait=true;
    backend.SDLEventLoop(stop.get_token()); assert(nextEvent==0 && timestamp==expectedTimestamp && actions==vector<string>({"error","error"}));
    run(backend,{key(SDL_SCANCODE_A,SDL_KEYUP)},forwarded(KEY_A,false));
    // Empty errors count as timeouts only for a pending visibility retry.
    // Repeated errors back off each time, but identical text is log-throttled.
    visibilityBusy=true; backend.m_bVisibilityPending=true; failures=1;
    run(backend,{}, {"visible","timed","visible","timed"});
    assert(!failures);
    failureError="display lost"; eventSubsystem=SDL_INIT_EVENTS; failures=2;
    run(backend,{}, {"visible","timed","error","delay","visible","timed","delay","visible","timed"});
    visibilityBusy=false; backend.m_bVisibilityPending=false; failureError.clear(); failures=1;
    run(backend,{}, {"error","delay"});
    failureError="cancelled"; failures=1; stopOnFailure=true;
    stop=std::stop_source{}; stopSource=&stop; actions.clear(); nextEvent=0;
    backend.SDLEventLoop(stop.get_token());
    assert(nextEvent==0 && actions==vector<string>({"push:"+std::to_string(Uint32(SDL_USEREVENT)+GAMESCOPE_SDL_EVENT_WAKE),"error"}));
    stopOnFailure=false; failureError.clear(); eventSubsystem=0;
    run(backend,{key(SDL_SCANCODE_A,SDL_KEYUP)},forwarded(KEY_A,false));
    g_bFullscreen=false;
    run(backend,{key(SDL_SCANCODE_F,SDL_KEYUP,KMOD_LGUI),key(SDL_SCANCODE_F,SDL_KEYUP,KMOD_LGUI)},
        {"fullscreen:"+std::to_string(SDL_WINDOW_FULLSCREEN_DESKTOP),"fullscreen:0"});
    assert(!g_bFullscreen);
    g_bGrabbed=false; backend.m_bApplicationGrabbed=true;
    run(backend,{key(SDL_SCANCODE_G,SDL_KEYUP,KMOD_LGUI),key(SDL_SCANCODE_G,SDL_KEYUP,KMOD_LGUI)},
        {"grab:1","push:"+std::to_string(SDL_USEREVENT),"grab:0","push:"+std::to_string(SDL_USEREVENT)});
    assert(!g_bGrabbed && backend.m_bApplicationGrabbed);
    assert(timestamp==expectedTimestamp && !lockDepth);
    std::printf("SDL event characterization: %u sequences plus quit/cancel/error PASS\n",tests);
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = (args.source or Path(__file__).resolve().parents[1] /
              'src/Backends/SDLBackend.cpp').resolve()
    if args.output:
        execute(source, args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='gamescope-sdl-events-') as name:
            execute(source, Path(name))


if __name__ == '__main__':
    main()
