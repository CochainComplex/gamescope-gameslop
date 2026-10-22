#!/usr/bin/env python3
"""Exercise the production import_commit cache boundary without Vulkan/Wayland."""
import hashlib
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import tempfile


FIXTURE = r'''
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace gamescope {
template<class T> class Rc {
    std::shared_ptr<T> value;
public:
    Rc() = default;
    Rc(std::nullptr_t) {}
    Rc(T *p) : value(p) {}
    T *get() const { return value.get(); }
    T *operator->() const { return value.get(); }
    explicit operator bool() const { return bool(value); }
    bool operator==(std::nullptr_t) const { return !value; }
};
template<class T> using OwningRc = Rc<T>;
struct PresentTiming {
    uint64_t id = 0, desired = 0;
    bool operator==(const PresentTiming &) const = default;
};
struct IBackendFb {};
}
struct wlr_surface {};
struct wlr_buffer {
    int width = 2, height = 2;
    bool dma = false, fail = false;
    std::shared_ptr<std::vector<uint8_t>> storage = std::make_shared<std::vector<uint8_t>>();
};
struct wlr_dmabuf_attributes {
    int width, height;
    uint32_t format;
    uint64_t modifier;
    int n_planes;
};
struct wlserver_vk_swapchain_feedback { int token = 0; };
struct wlserver_presentation_feedback_ref {
    int token;
    bool operator==(const wlserver_presentation_feedback_ref &) const = default;
};
struct steamcompmgr_win_t {
    uint64_t seq = 77;
    uint32_t appID = 123;
    bool steam = true, vr_scene = false, vr_app = false;
};
static bool window_is_steam(steamcompmgr_win_t *w) { return w->steam; }
static bool window_is_vr_scene_app(steamcompmgr_win_t *w) { return w->vr_scene; }
static bool window_is_vr_app(steamcompmgr_win_t *w) { return w->vr_app; }
struct CVulkanTexture {
    std::vector<uint8_t> snapshot;
    std::shared_ptr<std::vector<uint8_t>> shared_storage;
    const std::vector<uint8_t> &pixels() const { return shared_storage ? *shared_storage : snapshot; }
};
struct commit_t {
    uint64_t win_seq = 0;
    wlr_surface *surf = nullptr;
    wlr_buffer *buf = nullptr;
    bool async = false, fifo = false, is_steam = false;
    uint32_t appID = 0;
    std::vector<wlserver_presentation_feedback_ref> presentation_feedbacks;
    wlserver_vk_swapchain_feedback feedback;
    gamescope::PresentTiming present_timing;
    gamescope::Rc<CVulkanTexture> vulkanTex;
};
static unsigned queries = 0, cpu_uploads = 0, dma_imports = 0, backend_imports = 0;
static bool wlr_buffer_get_dmabuf(wlr_buffer *buf, wlr_dmabuf_attributes *attributes) {
    queries++;
    if (!buf->dma) return false;
    *attributes = {buf->width, buf->height, 0x34325241u, 0x42u, 1};
    return true;
}
namespace gamescope {
struct CBufferMemoizer {
    std::map<wlr_buffer *, OwningRc<CVulkanTexture>> entries;
    unsigned lookups = 0, stores = 0;
    OwningRc<CVulkanTexture> LookupVulkanTexture(wlr_buffer *buffer) {
        lookups++;
        auto entry = entries.find(buffer);
        return entry == entries.end() ? OwningRc<CVulkanTexture>{} : entry->second;
    }
    void MemoizeBuffer(wlr_buffer *buffer, OwningRc<CVulkanTexture> texture) {
        assert(texture);
        stores++;
        entries[buffer] = std::move(texture);
    }
};
}
static gamescope::CBufferMemoizer s_BufferMemos;
struct Backend {
    gamescope::OwningRc<gamescope::IBackendFb> ImportDmabufToBackend(wlr_dmabuf_attributes *attributes) {
        backend_imports++;
        assert(attributes->width == 2 && attributes->height == 2);
        assert(attributes->format == 0x34325241u && attributes->modifier == 0x42u && attributes->n_planes == 1);
        return new gamescope::IBackendFb;
    }
} backend;
static Backend *GetBackend() { return &backend; }
static gamescope::OwningRc<CVulkanTexture> vulkan_create_texture_from_wlr_buffer(
    wlr_buffer *buffer, gamescope::OwningRc<gamescope::IBackendFb> fb) {
    if (buffer->dma) {
        dma_imports++;
        assert(fb);
        if (buffer->fail) return nullptr;
        auto *texture = new CVulkanTexture;
        texture->shared_storage = buffer->storage;
        return texture;
    }
    assert(!fb);
    cpu_uploads++;
    auto *texture = new CVulkanTexture;
    texture->snapshot = *buffer->storage;
    return texture;
}
static bool g_bDebugDualGpuRoute = true;
struct Log {
    template<class... Args> void infof(const char *, Args...) {}
    template<class... Args> void errorf(const char *, Args...) {}
} xwm_log;

#include "import_commit.inc"

static gamescope::Rc<commit_t> submit(wlr_buffer &buffer, steamcompmgr_win_t &window, wlr_surface &surface,
                                     uint64_t id = 44, bool async = false, bool fifo = true) {
    const unsigned previous_queries = queries;
    auto feedback = std::make_shared<wlserver_vk_swapchain_feedback>();
    feedback->token = int(id + 10);
    auto commit = import_commit(&window, &surface, &buffer, async, feedback,
                               {{17}, {23}}, {id, id + 1000}, fifo);
    assert(queries == previous_queries + 1);
    if (commit) {
        assert(commit->win_seq == window.seq && commit->appID == window.appID);
        assert(commit->surf == &surface && commit->buf == &buffer && commit->is_steam == window.steam);
        assert(commit->feedback.token == int(id + 10));
        assert((commit->presentation_feedbacks == std::vector<wlserver_presentation_feedback_ref>{{17}, {23}}));
        assert((commit->present_timing == gamescope::PresentTiming{id, id + 1000}));
        assert(commit->async == (window.vr_scene || async));
        assert(commit->fifo == (window.vr_scene || window.vr_app ? false : fifo));
    }
    return commit;
}

int main() {
    steamcompmgr_win_t window;
    wlr_surface surface;
    wlr_buffer a, b;
    *a.storage = {10, 11, 12, 13};
    *b.storage = {20, 21, 22, 23};
    auto a1 = submit(a, window, surface);
    auto b1 = submit(b, window, surface);
    *a.storage = {30, 31, 32, 33};
    *b.storage = {40, 41, 42, 43};
    auto a2 = submit(a, window, surface, 45, true, false);
    auto b2 = submit(b, window, surface, 46);
    assert(a1->vulkanTex.get() != a2->vulkanTex.get());
    assert(b1->vulkanTex.get() != b2->vulkanTex.get());
    assert((a1->vulkanTex->pixels() == std::vector<uint8_t>{10,11,12,13}));
    assert((b1->vulkanTex->pixels() == std::vector<uint8_t>{20,21,22,23}));
    assert(a2->vulkanTex->pixels() == *a.storage && b2->vulkanTex->pixels() == *b.storage);
    assert(cpu_uploads == 4 && dma_imports == 0 && backend_imports == 0);
    assert(s_BufferMemos.entries.empty() && s_BufferMemos.lookups == 0 && s_BufferMemos.stores == 0);

    wlr_buffer dma;
    dma.dma = true;
    *dma.storage = {50,51,52,53};
    auto d1 = submit(dma, window, surface, 47);
    *dma.storage = {60,61,62,63};
    auto d2 = submit(dma, window, surface, 48, true, false);
    assert(d1.get() != d2.get() && d1->vulkanTex.get() == d2->vulkanTex.get());
    assert(d1->vulkanTex->pixels() == *dma.storage && d2->vulkanTex->pixels() == *dma.storage);
    assert(cpu_uploads == 4 && dma_imports == 1 && backend_imports == 1);
    assert(s_BufferMemos.entries.size() == 1 && s_BufferMemos.stores == 1 && s_BufferMemos.lookups == 2);

    wlr_buffer retry;
    retry.dma = true;
    retry.fail = true;
    *retry.storage = {70,71,72,73};
    auto failed = submit(retry, window, surface, 49);
    assert(failed == nullptr && s_BufferMemos.entries.count(&retry) == 0);
    assert(cpu_uploads == 4 && dma_imports == 2 && backend_imports == 2);
    retry.fail = false;
    auto retried = submit(retry, window, surface, 50);
    assert(retried && retried->vulkanTex->pixels() == *retry.storage);
    assert(cpu_uploads == 4 && dma_imports == 3 && backend_imports == 3);
    assert(s_BufferMemos.entries.size() == 2 && s_BufferMemos.stores == 2);
    window.vr_scene = true;
    auto vr_scene = submit(dma, window, surface, 51);
    window.vr_scene = false;
    window.vr_app = true;
    auto vr_app = submit(dma, window, surface, 52, false, true);
    assert(vr_scene && vr_app && dma_imports == 3 && backend_imports == 3);
}
'''


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/steamcompmgr.cpp").read_text()
    begin = source.index("static gamescope::Rc<commit_t>\nimport_commit (")
    function = source[begin:source.index("\nstatic int32_t\nwindow_last_done_commit_index(", begin)]
    print(f"source SHA256: {hashlib.sha256(source.encode()).hexdigest()}", flush=True)
    print(f"import_commit SHA256: {hashlib.sha256(function.encode()).hexdigest()}", flush=True)
    variants = [("production", function, True)]
    if sys.argv[1:] == ["--mutants"]:
        lookup = "\tif ( is_dmabuf )\n\t{\n\t\tif ( gamescope::OwningRc<CVulkanTexture>"
        store = "\tif ( is_dmabuf )\n\t\ts_BufferMemos.MemoizeBuffer("
        assert function.count(lookup) == function.count(store) == 1
        mutant = function.replace(lookup, lookup.replace("is_dmabuf", "true"), 1)
        mutant = mutant.replace(store, store.replace("is_dmabuf", "true"), 1)
        variants.append(("unconditional-cache-mutant", mutant, False))
    with tempfile.TemporaryDirectory(prefix="gamescope-buffer-import-") as directory:
        directory = Path(directory)
        fixture = directory / "test.cpp"
        fixture.write_text(FIXTURE)
        for name, body, expected in variants:
            (directory / "import_commit.inc").write_text(body)
            binary = directory / name
            subprocess.run(shlex.split(os.environ.get("CXX", "c++"))
                           + ["-std=c++20", "-Wall", "-Wextra", "-Werror", "-Wno-missing-field-initializers",
                              "-I", str(directory), str(fixture), "-o", str(binary)],
                           check=True, timeout=60)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10,
                                    preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
            if (result.returncode == 0) != expected:
                raise RuntimeError(f"{name}: unexpected exit {result.returncode}\n{result.stderr}")
            if not expected:
                assert result.returncode == -6 and "Assertion" in result.stderr, result.stderr
            print(f"{name}: {'PASS' if expected else 'caught at runtime'} (exit {result.returncode})")
            if not expected:
                print(result.stderr.strip())


if __name__ == "__main__":
    main()
