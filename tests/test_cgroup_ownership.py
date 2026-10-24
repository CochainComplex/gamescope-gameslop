#!/usr/bin/env python3
"""Exercise actual cgroup lookup/focus code with private proc-file inputs.

Tracks strdup results through their last comparison/write use. It neither
changes real cgroups nor models compositor window or GPU resource lifetimes.
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
    signature = 'cgroupPathFromPID(pid_t pid)'
    if source.count(signature) != 1:
        raise ValueError('ambiguous/missing cgroup helper')
    begin = source.rfind('\n', 0, source.index(signature)) + 1
    end = source.index('\n}', begin) + 2
    fragments = {'lookup': (begin, end)}
    first = 'pid_t newFocusedWindowPID = '
    last = 'if ( pFocus->pVirtualConnector && pFocus->inputFocusWindow )'
    if source.count(first) != 1 or source.count(last) != 1:
        raise ValueError('ambiguous/missing focus cgroup block')
    begin, end = source.index(first), source.index(last)
    if end <= begin:
        raise ValueError('reversed focus cgroup block')
    fragments['focus'] = begin, end
    for name, (begin, end) in fragments.items():
        line = source.count('\n', 0, begin) + 1
        (directory / (name + '.inc')).write_text(
            f'#line {line} "{source_path}"\n' + source[begin:end] + '\n')
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
        commands.append(command)
        (directory / 'commands.json').write_text(json.dumps(commands, indent=2))
        subprocess.run([str(binary)], check=True, timeout=30)


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
        with tempfile.TemporaryDirectory(prefix='cgroup-ownership-') as name:
            execute(source, Path(name))


FIXTURE = r'''
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <sys/types.h>

struct Proc {
    std::string text;
    std::optional<std::string> path;
    bool openFail = false, formatFail = false, duplicateFail = false;
    int readLimit = -1;
};
struct File { pid_t pid; int reads = 0; };
static std::map<pid_t, Proc> procs;
static std::map<FILE *, File> files;
static std::map<void *, std::string> paths;
static std::vector<std::string> events;
static pid_t currentPid;
static unsigned allocations, releases, cases, writeFailures;
static uint64_t digest = 14695981039346656037ull;

static int proc_snprintf(char *out, size_t size, const char *format, long pid) {
    currentPid = pid;
    if (procs.at(pid).formatFail) return -1;
    return std::snprintf(out, size, format, pid);
}
static FILE *proc_fopen(const char *path, const char *mode) {
    const auto &proc = procs.at(currentPid);
    assert(std::string(path) == "/proc/" + std::to_string(currentPid) + "/cgroup");
    assert(std::strcmp(mode, "r") == 0);
    if (proc.openFail) return nullptr;
    FILE *file = std::tmpfile();
    assert(file);
    assert(std::fwrite(proc.text.data(), 1, proc.text.size(), file) == proc.text.size());
    std::rewind(file);
    files.emplace(file, File{currentPid});
    return file;
}
static ssize_t proc_getline(char **line, size_t *size, FILE *file) {
    auto &state = files.at(file);
    const auto &proc = procs.at(state.pid);
    if (proc.readLimit >= 0 && state.reads >= proc.readLimit) return -1;
    ++state.reads;
    return ::getline(line, size, file);
}
static int proc_fclose(FILE *file) {
    assert(files.erase(file) == 1);
    return std::fclose(file);
}
static char *owned_strdup(const char *value) {
    if (procs.at(currentPid).duplicateFail) return nullptr;
    char *copy = ::strdup(value);
    assert(copy && paths.emplace(copy, value).second);
    ++allocations;
    return copy;
}
static void owned_free(void *pointer) {
    if (paths.erase(pointer)) {
        events.push_back("release");
        ++releases;
    }
    std::free(pointer);
}

#define snprintf proc_snprintf
#define fopen proc_fopen
#define getline proc_getline
#define fclose proc_fclose
#define strdup owned_strdup
#define free owned_free
#define cgroupPathFromPID production_cgroup_path
#include "lookup.inc"
#undef cgroupPathFromPID
#undef free
#undef strdup
#undef fclose
#undef getline
#undef fopen
#undef snprintf

static auto cgroupPathFromPID(pid_t pid) {
    events.push_back("lookup:" + std::to_string(pid));
    return production_cgroup_path(pid);
}
static int checked_strcmp(const char *a, const char *b) {
    assert(paths.count(const_cast<char *>(a)) && paths.count(const_cast<char *>(b)));
    events.push_back("compare");
    return std::strcmp(a, b);
}
static int setDMemMemoryLow(const char *path, bool focused) {
    assert(paths.count(const_cast<char *>(path)));
    events.push_back(std::string(focused ? "focus:" : "unfocus:") + path);
    return writeFailures & (focused ? 2 : 1) ? -1 : 0;
}
struct Window { pid_t pid; };
struct Focus { Window *focusWindow; };
static pid_t sdFocusWindow_pid;
static void apply(Focus *pFocus) {
#define strcmp checked_strcmp
#define free owned_free
#include "focus.inc"
#undef free
#undef strcmp
}

static void check(const Proc &oldProc, const Proc &newProc, unsigned failures,
                  pid_t oldPid = 11, pid_t newPid = 22, bool nullFocus = false) {
    procs = {{oldPid, oldProc}, {newPid, newProc}};
    writeFailures = failures;
    sdFocusWindow_pid = oldPid;
    Window window{newPid};
    Focus focus{nullFocus ? nullptr : &window};
    events.clear();
    apply(&focus);
    assert(sdFocusWindow_pid == oldPid && window.pid == newPid);
    assert(focus.focusWindow == (nullFocus ? nullptr : &window));
    assert(files.empty());

    std::vector<std::string> expected;
    pid_t target = nullFocus ? 0 : newPid;
    if (oldPid != target) {
        expected = {"lookup:" + std::to_string(oldPid), "lookup:" + std::to_string(target)};
        auto oldPath = oldPid ? oldProc.path : std::nullopt;
        auto newPath = target ? newProc.path : std::nullopt;
        if (oldPath && newPath) expected.push_back("compare");
        if (oldPath != newPath) {
            if (oldPath) expected.push_back("unfocus:" + *oldPath);
            if (newPath) expected.push_back("focus:" + *newPath);
        }
    }
    std::vector<std::string> observed;
    bool released = false;
    for (const auto &event : events) {
        if (event == "release") { released = true; continue; }
        assert(!released); // Both lookup/comparison/write lifetimes end first.
        observed.push_back(event);
        for (unsigned char c : event) { digest ^= c; digest *= 1099511628211ull; }
        digest ^= 0xff; digest *= 1099511628211ull;
    }
    assert(observed == expected);
    ++cases;
}

int main() {
    const std::vector<Proc> inputs{
        {"0::/a\n", "/a"},
        {"0::/a\n", "/a"}, // Equal strings must still own separate allocations.
        {"0::/b", "/b"},
        {"", std::nullopt},
        {"1:memory:/v1\n", std::nullopt},
        {"0::/gone (deleted)\n", std::nullopt},
        {"0::/a\n", std::nullopt, true},
        {"0::/a\n", std::nullopt, false, true},
        {"0::/a\n", std::nullopt, false, false, true},
        {"0::/a\n", std::nullopt, false, false, false, 0},
        {"1:memory:/v1\n0::/a\n", std::nullopt, false, false, false, 1},
        {"1:memory:/v1\n0::/a\n", "/a"},
        {"0::\n", ""},
        {"0::/" + std::string(8192, 'x') + "\n", "/" + std::string(8192, 'x')},
    };
    for (const auto &a : inputs)
        for (const auto &b : inputs)
            for (unsigned fail = 0; fail < 4; ++fail) check(a, b, fail);
    for (const auto &p : inputs) {
        check(p, p, 0, 11, 11); // Unchanged PID must perform no lookup.
        check(p, p, 0, 0, 22);
        check(p, p, 0, 11, 0);
        check(p, p, 0, 11, 22, true); // Null focus selects PID0 without mutation.
        check(p, p, 0, 0, 22, true);
    }
    for (unsigned i = 0; i < 1000; ++i) {
        check(inputs[0], inputs[2], i % 4, 11, 22);
        check(inputs[2], inputs[0], i % 4, 22, 11);
        check(inputs[0], inputs[1], 0, 11, 22);
    }
    const size_t outstanding = paths.size();
    std::printf("cases=%u allocations=%u releases=%u outstanding=%zu semantic_digest=%016llx\n",
        cases, allocations, releases, outstanding, (unsigned long long)digest);
    // Baseline red runs report the actual leak, then reclaim only fixture-owned
    // allocations so the regression itself leaves no temporary leak or core dump.
    for (const auto &[pointer, value] : paths) { (void)value; std::free(pointer); }
    return outstanding == 0 && allocations == releases ? 0 : 1;
}
'''


if __name__ == '__main__':
    main()
