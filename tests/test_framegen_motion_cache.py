#!/usr/bin/env python3
"""Characterize exact production cache decisions with CPU scalar/handle fakes.

Actual forward/reverse getter definitions select raw or refined handle stubs.
No Vulkan, reference-count lifetime, resource readiness or GPU execution is modeled.
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
        raise ValueError('ambiguous/missing production definition: ' + signature)
    begin = source.index(signature)
    return begin, source[begin:source.index('\n}', begin) + 2]


def declaration(source, signature):
    if source.count(signature) != 1:
        raise ValueError('ambiguous/missing production declaration: ' + signature)
    begin = source.index(signature)
    return begin, source[begin:source.index(';', begin) + 1]


def materialize(root, source_path, directory):
    source = source_path.read_text()
    fragments = {}
    for name, signature in (
        ('field', 'static const gamescope::OwningRc<CVulkanTexture> &framegen_motion_field()'),
        ('reverse', 'static const gamescope::OwningRc<CVulkanTexture> &framegen_motion_field_rev()'),
        ('causal', 'static gamescope::framegen::DeadlineWorkClass_t framegen_causal_work_class('),
    ):
        fragments[name] = definition(source, signature)
    for name, signature in (
        ('requested', 'const bool bMotionRequested = eff.mode'),
        ('reuse', 'const bool bReuseMotion = pColorProbe'),
        ('work', 'const gamescope::framegen::DeadlineWorkClass_t eDeadlineWorkClass = bReuseMotion'),
    ):
        fragments[name] = declaration(source, signature)
    for name, (offset, fragment) in fragments.items():
        line = source.count('\n', 0, offset) + 1
        (directory / (name + '.inc')).write_text(
            f'#line {line} "{source_path}"\n' + fragment + '\n')
    (directory / 'fixture.cpp').write_text(FIXTURE)


def compile_fixture(root, directory, binary, flags):
    command = shlex.split(os.environ.get('CXX', 'c++')) + [
        '-std=c++20', '-Wall', '-Wextra', '-Werror', '-Wno-missing-field-initializers',
        '-UNDEBUG', *flags, '-I', str(root / 'src'), '-I', str(directory),
        str(directory / 'fixture.cpp'), '-o', str(binary),
    ]
    subprocess.run(command, check=True, timeout=60)
    return command


def execute(root, source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    materialize(root, source, directory)
    commands = []
    for variant, flags in (('normal', ['-O0']), ('fastmath', ['-O3', '-ffast-math'])):
        binary = directory / variant
        commands.append(compile_fixture(root, directory, binary, flags))
        subprocess.run([str(binary)], check=True, timeout=15)
    (directory / 'commands.json').write_text(json.dumps(commands, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (args.source or root / 'src/rendervulkan.cpp').resolve()
    if args.output:
        execute(root, source, args.output.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='framegen-motion-cache-') as name:
            execute(root, source, Path(name))


FIXTURE = r'''
#include "framegen/policy.hpp"
#include "framegen/scheduling.hpp"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
using FramegenEffective_t = gamescope::framegen::EffectiveConfig;
using Work = gamescope::framegen::DeadlineWorkClass_t;
struct CVulkanTexture {};
namespace gamescope {
template<class T> struct OwningRc {
    bool present = true;
    unsigned identity = 0;
    bool operator!=(std::nullptr_t) const { return present; }
    bool operator==(const OwningRc &) const = default;
};
}
struct Motion {
    uint64_t uMotionFieldFrameId = 7;
    GamescopeFramegenPipeline eMotionFieldPipeline = GamescopeFramegenPipeline::Warp;
    bool bMotionFieldBidir = false, bNetActive = false;
    gamescope::OwningRc<CVulkanTexture> mvField{true,1}, mvFieldRevChk{true,2};
    gamescope::OwningRc<CVulkanTexture> mvFieldNet{true,3}, mvFieldRevNet{true,4};
    bool operator==(const Motion &) const = default;
} g_framegenMotion;
struct History {
    uint64_t currentFrameId = 7; uint32_t drmFormat = 0;
    bool operator==(const History &) const = default;
} g_framegenHistory;
struct FramegenDispatch_t {
    bool motionSupported = true;
    bool operator==(const FramegenDispatch_t &) const = default;
} g_dispatch;
static uint64_t trace = 0;
static const FramegenDispatch_t &framegen_dispatch_for_format(uint32_t) { trace = trace*10+9; return g_dispatch; }
// Compile the actual getter bodies under private names; the wrappers record the
// selected handle identity without changing the production predicate fragments.
#define framegen_motion_field original_motion_field
#include "field.inc"
#undef framegen_motion_field
#define framegen_motion_field_rev original_motion_field_rev
#include "reverse.inc"
#undef framegen_motion_field_rev
static const gamescope::OwningRc<CVulkanTexture> &framegen_motion_field() {
    const auto &field = original_motion_field(); trace = trace*10+field.identity; return field;
}
[[maybe_unused]] static const gamescope::OwningRc<CVulkanTexture> &framegen_motion_field_rev() {
    const auto &field = original_motion_field_rev(); trace = trace*10+field.identity; return field;
}
#include "causal.inc"
struct Result { bool reuse; Work work; uint64_t calls; Work causal; uint64_t causalCalls; };
static Result submit(const FramegenEffective_t &eff, bool bBidir, bool probe) {
    trace = 0;
    const auto &dispatch = framegen_dispatch_for_format(g_framegenHistory.drmFormat);
    const void *pColorProbe = probe ? &g_dispatch : nullptr;
#include "requested.inc"
#include "reuse.inc"
#include "work.inc"
    Result result{bReuseMotion,eDeadlineWorkClass,trace};
    trace = 0;
    result.causal = framegen_causal_work_class(eff); result.causalCalls = trace;
    return result;
}
struct Case {
    uint64_t cacheId = 7, currentId = 7;
    GamescopeFramegenPipeline pipeline = GamescopeFramegenPipeline::Warp;
    GamescopeFramegenPipeline cachedPipeline = GamescopeFramegenPipeline::Warp;
    GamescopeFramegenMode mode = GamescopeFramegenMode::Motion;
    bool cachedBidir = false, bidir = false, supported = true, probe = false, net = false;
    unsigned fields = 15;
};
static Result observe(const Case &c) {
    g_framegenMotion.uMotionFieldFrameId = c.cacheId;
    g_framegenHistory.currentFrameId = c.currentId;
    g_framegenMotion.eMotionFieldPipeline = c.cachedPipeline;
    g_framegenMotion.bMotionFieldBidir = c.cachedBidir;
    g_framegenMotion.bNetActive = c.net;
    g_framegenMotion.mvField.present = c.fields&1;
    g_framegenMotion.mvFieldRevChk.present = c.fields&2;
    g_framegenMotion.mvFieldNet.present = c.fields&4;
    g_framegenMotion.mvFieldRevNet.present = c.fields&8;
    g_dispatch.motionSupported = c.supported;
    FramegenEffective_t eff; eff.mode = c.mode; eff.pipeline = c.pipeline;
    const auto motion = g_framegenMotion;
    const auto history = g_framegenHistory;
    const auto dispatch = g_dispatch;
    const auto result = submit(eff,c.bidir,c.probe);
    assert(g_framegenMotion==motion && g_framegenHistory==history && g_dispatch==dispatch);
    return result;
}
static uint64_t rows = 0;
static bool dump = false;
static void check(const Case &c, bool reuse, uint64_t calls, bool cachedCausal, uint64_t causalCalls) {
    const Result r = observe(c);
    assert(r.reuse==reuse && (r.work==Work::CachedWarp)==reuse && r.calls==calls);
    assert((r.causal==Work::CachedWarp)==cachedCausal && r.causalCalls==causalCalls);
    if(dump) std::printf("%llu,%u,%u,%llu,%u,%llu\n",(unsigned long long)rows,r.reuse,unsigned(r.work),(unsigned long long)r.calls,unsigned(r.causal),(unsigned long long)r.causalCalls);
    ++rows;
}
static void contracts() {
    Case c;
    check(c,true,91,true,91);
    // Every scalar gate rejects before a field getter; causal always obtains
    // dispatch facts even when the submit-only probe gate rejects the cache.
    c.cacheId=0; check(c,false,9,false,9);
    c.currentId=0; check(c,false,9,false,9);
    c.cacheId=UINT64_MAX; check(c,false,9,false,9);
    c.currentId=UINT64_MAX; check(c,true,91,true,91);
    c.currentId=UINT64_MAX-1; check(c,false,9,false,9);
    c=Case{}; c.probe=true; check(c,false,9,true,91);
    c.probe=false; c.supported=false; check(c,false,9,false,9);
    for(auto mode:{GamescopeFramegenMode::Extrapolate,GamescopeFramegenMode::Blend,GamescopeFramegenMode::Motion}) {
        c=Case{};c.mode=mode;
        const bool motion=mode==GamescopeFramegenMode::Motion;
        check(c,motion,motion?91:9,motion,motion?91:9);
    }
    constexpr std::array pipelines={GamescopeFramegenPipeline::Warp,GamescopeFramegenPipeline::Checked,GamescopeFramegenPipeline::Learned,GamescopeFramegenPipeline::Predict,GamescopeFramegenPipeline::Guided,GamescopeFramegenPipeline::Spatial};
    for(auto requested:pipelines) for(auto cached:pipelines) {
        c=Case{};c.pipeline=requested;c.cachedPipeline=cached;
        const bool same=requested==cached;check(c,same,same?91:9,same,same?91:9);
    }
    for(bool net:{false,true}) for(unsigned mask=0;mask<16;++mask) {
        c=Case{};c.net=net;c.fields=mask;
        const unsigned forward=net?4:1,reverse=net?8:2;
        const uint64_t fwdTrace=net?93:91,revTrace=net?934:912;
        const bool fwd=mask&forward,rev=mask&reverse;
        check(c,fwd,fwdTrace,fwd,fwdTrace);
        c.cachedBidir=true;check(c,false,9,false,9);
        c.bidir=true;check(c,fwd&&rev,fwd?revTrace:fwdTrace,false,9);
        c.cachedBidir=false;check(c,false,9,fwd,fwdTrace);
        c.probe=true;check(c,false,9,fwd,fwdTrace);
        c.supported=false;check(c,false,9,false,9);
    }
    // Consecutive observations: same interval reuses; a new real misses until
    // finalized; changing the pipeline misses until finalized for that key.
    c=Case{};check(c,true,91,true,91);
    ++c.currentId;check(c,false,9,false,9);
    c.cacheId=c.currentId;check(c,true,91,true,91);
    c.pipeline=GamescopeFramegenPipeline::Spatial;check(c,false,9,false,9);
    c.cachedPipeline=c.pipeline;check(c,true,91,true,91);
}
template<class T> static void escape(const T &value) { asm volatile("" : : "g"(&value) : "memory"); }
static void benchmark() {
    std::array<Case,10> cases;
    cases[1].bidir=cases[1].cachedBidir=true;
    cases[2].cacheId=0; cases[3].cachedPipeline=GamescopeFramegenPipeline::Predict;
    cases[4].cachedBidir=true; cases[5].fields=14;
    cases[6].bidir=cases[6].cachedBidir=true;cases[6].fields=13;
    cases[7].probe=true;cases[8].supported=false;cases[9].net=true;
    const char *names[]={"causal-hit","bidir-hit","id-miss","pipeline-miss","direction-miss","forward-missing","reverse-missing","color-probe","unsupported","refined-hit"};
    uint64_t checksum=0;
    for(size_t path=0;path<cases.size();++path) {
        for(unsigned i=0;i<2048;++i) { auto r=observe(cases[path]);escape(r);checksum+=r.calls+r.causalCalls; }
        for(unsigned batch=0;batch<101;++batch) {
            const auto start=std::chrono::steady_clock::now();
            for(unsigned i=0;i<4096;++i) { auto r=observe(cases[path]);escape(r);checksum+=r.reuse+unsigned(r.work)+r.calls+unsigned(r.causal)+r.causalCalls; }
            const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
            std::printf("%s,%u,%lld,%llu\n",names[path],batch,(long long)ns,(unsigned long long)checksum);
        }
    }
}
int main(int argc,char **argv) {
    if(argc>1 && std::strcmp(argv[1],"--bench")==0) { benchmark();return 0; }
    dump=argc>1 && std::strcmp(argv[1],"--dump")==0;
    contracts();
    if(!dump)std::printf("%llu exact production cache/trace contracts passed\n",(unsigned long long)rows);
}
'''


if __name__ == '__main__':
    main()
