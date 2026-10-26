#!/usr/bin/env python3
"""Execute production backend export normalization and commit readiness on CPU."""
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile


def between(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


def capability(source):
    matches = re.findall(r"virtual bool UsesDmabufForPresentation\(\) const(?: override)?\s*\{[^}]*\}", source)
    assert len(matches) == 1, "expected one production capability definition"
    return matches[0]


def fixture(root):
    source = (root / "src/rendervulkan.cpp").read_text()
    compositor = (root / "src/steamcompmgr.cpp").read_text()
    init = between(source, "bool CVulkanTexture::BInit(", "bool CVulkanTexture::BInitFromSwapchain(")
    # Execute the backend normalization/export prefix. The independent
    # host-upload ownership guard has its own mapped-memory allocation fixture.
    prefix = between(init, "\tflags.bFlippable =", "\n\t// Possible extensions for below")
    private_types = between(source, "#define VK_STRUCTURE_TYPE_WSI_IMAGE_CREATE_INFO_MESA", "\n// DRM doesn't")
    image_hints = between(init, "\tif ( flags.bFlippable == true && tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT )", "\n\tm_width = width;")
    memory_hint = between(init, "\t\tif ( flags.bFlippable == true )", "\n\t\tif ( flags.bExportable == true || pDMA != nullptr )")
    readiness = between(compositor, "\tif ( eventFd != gamescope::CAcquireTimelinePoint::k_InvalidEvent )", "\n\tgpuvis_trace_printf( \"pushing wait")
    methods = {name: capability((root / path).read_text()) for name, path in {
        "Backend": "src/backend.h", "SDL": "src/Backends/SDLBackend.cpp",
        "Headless": "src/Backends/HeadlessBackend.cpp", "Deferred": "src/Backends/DeferredBackend.h"}.items()}
    for name in ("DRM", "Wayland", "OpenVR"):
        assert "UsesDmabufForPresentation" not in (root / f"src/Backends/{name}Backend.cpp").read_text(), name
    return r'''
#include <vulkan/vulkan.h>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <utility>
''' + private_types + r'''
struct Backend { virtual ~Backend() = default;
''' + methods["Backend"] + r'''
};
struct SDL : Backend { ''' + methods["SDL"] + r''' };
struct Headless : Backend { ''' + methods["Headless"] + r''' };
struct Deferred : Backend { Backend *m_pChild;
explicit Deferred(Backend *child) : m_pChild(child) {}
''' + methods["Deferred"] + r''' };
struct Flags {
    bool bFlippable=false, bExportable=false, bMappable=false, bLinear=false, bOutputImage=false;
    VkImageUsageFlags imageUsage() const { return VK_IMAGE_USAGE_SAMPLED_BIT; }
};
static Backend *backend;
[[maybe_unused]] static Backend *GetBackend() { return backend; }
static int dmabuf_render_origin(void*) { return 0; }
struct Result { bool flip, exported, external, image_hint, memory_hint, output; };
static Result initialize(Flags flags, void *pDMA=nullptr) {
    int m_pBackendFb=0, pBackendFb=0, m_drmFormat=0, drmFormat=0, m_renderOrigin=0;
    bool m_bOutputImage=false, m_bExternal=false;
''' + prefix + r'''
    (void)m_pBackendFb; (void)m_drmFormat; (void)m_renderOrigin;
    (void)res; (void)usage; (void)properties;
    VkImageCreateInfo imageInfo{};
    wsi_image_create_info wsiImageCreateInfo{};
    VkExternalMemoryImageCreateInfo externalImageCreateInfo{};
''' + image_hints + r'''
    VkMemoryAllocateInfo allocInfo{};
    wsi_memory_allocate_info memory_wsi_info{};
''' + memory_hint + r'''
    return {flags.bFlippable, flags.bExportable, m_bExternal,
            wsiImageCreateInfo.sType == VK_STRUCTURE_TYPE_WSI_IMAGE_CREATE_INFO_MESA,
            memory_wsi_info.sType == VK_STRUCTURE_TYPE_WSI_MEMORY_ALLOCATE_INFO_MESA,
            m_bOutputImage};
}
namespace gamescope { struct CAcquireTimelinePoint {
    static constexpr std::pair<int,bool> k_InvalidEvent{-1,false};
}; }
struct wlr_dmabuf_attributes { int fd[1]; };
static bool is_dma;
static int queries, duplicates, waits, waited;
static bool wlr_buffer_get_dmabuf(int*, wlr_dmabuf_attributes *attributes) {
    ++queries; attributes->fd[0]=37; return is_dma;
}
[[maybe_unused]] static int dup(int fd) { ++duplicates; assert(fd==37); return 91; }
static void vulkan_wait(int sequence, bool blocking) { ++waits; waited=sequence; assert(blocking); }
struct Ready { int fence; bool known; };
static Ready ready(std::pair<int,bool> eventFd, std::optional<int> oTextureSeqNo, bool dma) {
    is_dma=dma; queries=duplicates=waits=waited=0;
    int storage=0, *buf=&storage, fence=-1;
    bool bKnownReady=false;
''' + readiness + r'''
    return {fence,bKnownReady};
}
int main() {
    Backend native; SDL sdl; Headless headless;
    Deferred deferred(&native);
    assert(native.UsesDmabufForPresentation());
    assert(!sdl.UsesDmabufForPresentation() && !headless.UsesDmabufForPresentation());
    assert(deferred.UsesDmabufForPresentation());
    deferred.m_pChild=&sdl; assert(!deferred.UsesDmabufForPresentation());
    for (Backend *selected : {&native, static_cast<Backend*>(&sdl), static_cast<Backend*>(&headless), static_cast<Backend*>(&deferred)}) {
        backend=selected;
        for (bool flip : {false,true}) for (bool exported : {false,true}) for (bool imported : {false,true}) {
            Flags flags; flags.bFlippable=flip; flags.bExportable=exported; flags.bOutputImage=true;
            auto result=initialize(flags, imported ? &flags : nullptr);
            const bool expected_flip=flip && selected==&native;
            assert(result.flip==expected_flip);
            assert(result.exported==(exported || expected_flip));
            assert(result.external==(imported || exported || expected_flip));
            assert(result.image_hint==expected_flip && result.memory_hint==expected_flip);
            assert(result.output);
        }
    }
    // Explicit acquire events dominate both texture timelines and DMA fences.
    for (bool known : {false,true}) for (bool dma : {false,true}) for (bool timeline : {false,true}) {
        auto result=ready({73,known}, timeline ? std::optional<int>(29) : std::nullopt, dma);
        assert(result.fence==73 && result.known==known);
        assert(queries==0 && duplicates==0 && waits==0);
    }
    for (bool dma : {false,true}) {
        auto result=ready({-1,false},29,dma);
        assert(result.fence==-1 && result.known && waits==1 && waited==29);
        assert(queries==0 && duplicates==0);
    }
    auto dma=ready({-1,false},std::nullopt,true);
    assert(dma.fence==91 && !dma.known && queries==1 && duplicates==1 && waits==0);
    auto software=ready({-1,false},std::nullopt,false);
    assert(software.fence==-1 && software.known && queries==1 && duplicates==0 && waits==0);
}
'''


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="gamescope-backend-exports-") as directory:
        path = Path(directory) / "fixture.cpp"
        path.write_text(fixture(root))
        binary = path.with_suffix("")
        subprocess.run(shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++20", "-Wall", "-Wextra", "-Werror", "-Wno-missing-field-initializers",
            str(path), "-o", str(binary)], check=True, timeout=60)
        subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == "__main__":
    main()
