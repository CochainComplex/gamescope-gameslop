#!/usr/bin/env python3
"""Characterize production OpenVR modifier discovery and keyboard interpretation.

The SDK types are real; the two IPC queries and input-method effects are fakes.
This does not initialize SteamVR, create overlays, or establish GPU completion.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def fragment(source, begin, end):
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError('ambiguous/missing production fragment: ' + begin)
    start = source.index(begin)
    stop = source.index(end, start)
    return start, source[start:stop]


def materialize(source_path, directory):
    source = source_path.read_text()
    if 'void InitFormatModifiers()' in source:
        start, modifiers = fragment(source, '        void InitFormatModifiers()',
                                    '        void WaitUntilVisible()')
        line = source.count('\n', 0, start) + 1
        modifiers = f'#line {line} "{source_path}"\n' + modifiers
    else:
        start, body = fragment(source, '            if ( !g_bAllowDeferredBackend )',
                               '            if ( UsesModifiers() )')
        line = source.count('\n', 0, start) + 1
        modifiers = ('void InitFormatModifiers() {\n' +
                     f'#line {line} "{source_path}"\n' + body + '\n}\n')
    (directory / 'modifiers.inc').write_text(modifiers)
    start, body = fragment(source, '                case vr::VREvent_KeyboardCharInput:',
                           '                case vr::VREvent_MouseMove:')
    line = source.count('\n', 0, start) + 1
    (directory / 'keyboard.inc').write_text(f'#line {line} "{source_path}"\n' + body)
    (directory / 'fixture.cpp').write_text(FIXTURE)


def execute(source, directory, build_dir=None):
    directory.mkdir(parents=True, exist_ok=True)
    materialize(source, directory)
    root = Path(__file__).resolve().parents[1]
    includes = shlex.split(subprocess.check_output(
        ['pkg-config', '--cflags', 'libdrm'], text=True))
    if build_dir:
        dependencies = json.loads((build_dir / 'meson-info/intro-dependencies.json').read_text())
        sdk = next(d for d in dependencies if 'openvr_dep' in d['meson_variables'])
        includes += sdk['compile_args'] + ['-I' + p for p in sdk['include_directories']]
    else:
        includes += ['-I' + str(root / 'subprojects/openvr/headers')]
    commands = []
    for name, flags in [('normal', ['-O0']), ('optimized', ['-O3', '-ffast-math'])]:
        binary = directory / name
        command = shlex.split(os.environ.get('CXX', 'c++')) + [
            '-std=c++20', '-Wall', '-Wextra', '-Werror', '-UNDEBUG', *flags,
            *includes,
            '-I' + str(directory), str(directory / 'fixture.cpp'), '-o', str(binary),
        ]
        subprocess.run(command, check=True, timeout=60)
        subprocess.run([str(binary)], check=True, timeout=30)
        commands.append(command)
    (directory / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')


FIXTURE = r'''
#include <openvr.h>
#include <drm_fourcc.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
using std::string;
using std::vector;
using Table=std::unordered_map<uint32_t,vector<uint64_t>>;
static bool g_bAllowDeferredBackend;
static vector<string> calls;
static string number(uint64_t value) { return std::to_string(value); }
struct IPC {
    vector<uint32_t> formats;
    Table modifiers;
    bool failFormatCount=false, failFormatData=false, failModifierCount=false, failModifierData=false;
    uint32_t extraFormats=0, extraModifiers=0;
    bool GetDmabufFormats(uint32_t *count, uint32_t *data) {
        calls.push_back(data ? "formats:data:"+number(*count) : "formats:count");
        if ((!data && failFormatCount) || (data && failFormatData)) { *count=0; return false; }
        if (!data) { *count=formats.size()+extraFormats; return true; }
        assert(*count>=formats.size());
        std::copy(formats.begin(),formats.end(),data); *count=formats.size(); return true;
    }
    bool GetDmabufModifiers(vr::EVRApplicationType app, uint32_t format, uint32_t *count, uint64_t *data) {
        assert(app==vr::VRApplication_Overlay);
        calls.push_back(data ? "modifiers:data:"+number(format)+":"+number(*count) : "modifiers:count:"+number(format));
        if ((!data && failModifierCount) || (data && failModifierData)) { *count=0; return false; }
        const auto &values=modifiers[format];
        if (!data) { *count=values.size()+extraModifiers; return true; }
        assert(*count>=values.size());
        std::copy(values.begin(),values.end(),data); *count=values.size(); return true;
    }
};
enum gamescope_input_method_action { GAMESCOPE_INPUT_METHOD_ACTION_NONE,
    GAMESCOPE_INPUT_METHOD_ACTION_MOVE_LEFT, GAMESCOPE_INPUT_METHOD_ACTION_MOVE_RIGHT,
    GAMESCOPE_INPUT_METHOD_ACTION_MOVE_UP, GAMESCOPE_INPUT_METHOD_ACTION_MOVE_DOWN,
    GAMESCOPE_INPUT_METHOD_ACTION_SUBMIT, GAMESCOPE_INPUT_METHOD_ACTION_DELETE_LEFT };
struct InputMethod {} method;
static void perform_action(InputMethod *ime, gamescope_input_method_action action) {
    assert(ime==&method); calls.push_back("action:"+number(action));
}
static void type_text(InputMethod *ime, const char *text) {
    assert(ime==&method); calls.push_back("text:"+string(text));
}
struct Backend {
    IPC *m_pIPCResourceManager=nullptr;
    Table m_FormatModifiers;
    InputMethod *m_pIME=nullptr;
#include "modifiers.inc"
    void KeyboardEvent(const vr::VREvent_t &vrEvent) {
        switch (vrEvent.eventType) {
#include "keyboard.inc"
        default: break;
        }
    }
};
static unsigned cases;
static void query(IPC ipc, const Table &expected, const vector<string> &trace) {
    for (bool deferred : {false,true}) for (bool present : {false,true}) {
        g_bAllowDeferredBackend=deferred;
        Backend backend; backend.m_pIPCResourceManager=present ? &ipc : nullptr;
        calls.clear(); backend.InitFormatModifiers();
        const bool queried=!deferred && present;
        assert(backend.m_FormatModifiers==(queried ? expected : Table{}));
        assert(calls==(queried ? trace : vector<string>{})); ++cases;
    }
}
static void characterizations() {
    constexpr uint32_t a=DRM_FORMAT_XRGB8888, b=DRM_FORMAT_ARGB2101010;
    const string ac="modifiers:count:"+number(a), bc="modifiers:count:"+number(b);
    const auto ad=[](unsigned count) { return "modifiers:data:"+number(a)+":"+number(count); };
    query({}, {}, {"formats:count"});
    IPC base; base.formats={a,b}; base.modifiers={{a,{DRM_FORMAT_MOD_INVALID,0,17,0}},{b,{}}};
    query(base,{{a,{0,17,0}}},{"formats:count","formats:data:2",ac,ad(4),bc});
    IPC later=base; later.modifiers={{a,{}},{b,{17}}};
    query(later,{{b,{17}}},{"formats:count","formats:data:2",ac,bc,"modifiers:data:"+number(b)+":1"});
    IPC invalid; invalid.formats={a}; invalid.modifiers={{a,{DRM_FORMAT_MOD_INVALID}}};
    query(invalid,{}, {"formats:count","formats:data:1",ac,ad(1)});
    IPC duplicate; duplicate.formats={a,a}; duplicate.modifiers={{a,{17,0}}};
    query(duplicate,{{a,{17,0,17,0}}},{"formats:count","formats:data:2",ac,ad(2),ac,ad(2)});
    IPC failure=base; failure.failFormatCount=true;
    query(failure,{}, {"formats:count"});
    failure=base; failure.failFormatData=true;
    query(failure,{}, {"formats:count","formats:data:2"});
    failure=invalid; failure.failModifierCount=true;
    query(failure,{}, {"formats:count","formats:data:1",ac});
    // Existing behavior traverses the allocated modifier vector even if the
    // second query fails or writes fewer entries. Preserve it in this structural
    // change; this is characterization, not a claim that fallback policy is ideal.
    failure=invalid; failure.failModifierData=true;
    query(failure,{{a,{0}}}, {"formats:count","formats:data:1",ac,ad(1)});
    IPC shrinking; shrinking.formats={a}; shrinking.modifiers={{a,{17}}};
    shrinking.extraFormats=2; shrinking.extraModifiers=2;
    query(shrinking,{{a,{17,0,0}}},{"formats:count","formats:data:3",ac,ad(3)});
    struct Text { const char *value; vector<string> expected; };
    for (const Text &test : vector<Text>{
            {"",{"text:"}}, {"hello",{"text:hello"}}, {"\n",{"action:5"}},
            {"\b",{"action:6"}}, {"\nX",{"text:\nX"}}, {"\x1b",{"text:\x1b"}},
            {"\x1b[",{"text:\x1b["}}, {"\x1b[D",{"action:1"}},
            {"\x1b[C",{"action:2"}}, {"\x1b[A",{"action:3"}}, {"\x1b[B",{"action:4"}},
            {"\x1b[Z",{}}, {"\x1bXD",{"text:\x1bXD"}},
            {"\x1b[Dx",{"action:1"}}, {"\xc3\xa9",{"text:\xc3\xa9"}}}) {
        for (bool available : {false,true}) {
            vr::VREvent_t event{}; event.eventType=vr::VREvent_KeyboardCharInput;
            assert(std::strlen(test.value)<sizeof(event.data.keyboard.cNewInput));
            std::strcpy(event.data.keyboard.cNewInput,test.value);
            Backend backend; backend.m_pIME=available ? &method : nullptr;
            calls.clear(); backend.KeyboardEvent(event);
            assert(calls==(available ? test.expected : vector<string>{})); ++cases;
        }
    }
    Backend backend; backend.m_pIME=&method;
    vr::VREvent_t event{}; event.eventType=vr::VREvent_None;
    calls.clear(); backend.KeyboardEvent(event); assert(calls.empty()); ++cases;
    event.eventType=vr::VREvent_KeyboardCharInput; std::strcpy(event.data.keyboard.cNewInput,"a");
    backend.KeyboardEvent(event); backend.KeyboardEvent(event);
    assert(calls==vector<string>({"text:a","text:a"})); ++cases;
}
int main() {
    characterizations();
    std::printf("OpenVR initialization/input characterization: %u cases PASS\n",cases);
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--build-dir', type=Path)
    args = parser.parse_args()
    source = (args.source or Path(__file__).resolve().parents[1] /
              'src/Backends/OpenVRBackend.cpp').resolve()
    if args.output:
        execute(source, args.output.resolve(), args.build_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='gamescope-openvr-init-input-') as name:
            execute(source, Path(name), args.build_dir)


if __name__ == '__main__':
    main()
