#!/usr/bin/env python3
"""Execute temporary-image policy from production with CPU Vulkan recording.

Each fake capability profile gets a fresh process: the production capability
cache belongs to an immutable device. This does not prove GPU dispatch,
pixel range or real Vulkan feature availability. Small extracted
caller fragments cover fallback restoration and preemptive eligibility only.
"""
import os
from pathlib import Path
import shlex
import subprocess
from tempfile import TemporaryDirectory


def function(source, signature):
    if source.count(signature) != 1:
        raise ValueError('ambiguous/missing production function: ' + signature)
    begin = source.index(signature)
    # These renderer functions follow the surrounding top-level brace style.
    end = source.index('\n}', begin) + 2
    return source[begin:end]


def between(source, start, end):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('ambiguous/missing production declaration: ' + start)
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / 'src/rendervulkan.cpp').read_text()
    header = (root / 'src/rendervulkan.hpp').read_text()
    shared = (root / 'src/gamescope_shared.h').read_text()
    steam = (root / 'src/steamcompmgr.cpp').read_text()
    bodies = '\n'.join(function(source, signature) for signature in (
        'static bool tmp_image_format_supported(',
        'static uint32_t tmp_image_format(',
        'static bool update_tmp_images(',
        'bool vulkan_prepare_composite_intermediate(',
    ))
    extracts = {
        'owner': bodies,
        'flags': between(header, '\tstruct createFlags {',
                         '\n\tbool BInit( uint32_t width, uint32_t height, uint32_t depth, uint32_t drmFormat, createFlags flags'),
        'colorspaces': between(shared, 'enum GamescopeAppTextureColorspace\n',
                              '\nconst uint32_t GamescopeAppTextureColorspace_Count'),
        'blur': function(header, 'enum BlurMode {') + ';',
        'fallback': between(source, '\tconst bool bRequestedFSR =', '\n\t// Output-space cursor split, first half.'),
        'preemptive': between(steam, '\tbool bValidPreemptiveScale =', '\n\tbool bKnownReady ='),
        'pool': between(steam, '\t\tPooledImage_t *pTempImage = vulkan_prepare_composite_intermediate(', '\n\t\tif ( pTempImage )'),
    }
    with TemporaryDirectory(prefix='vulkan-tmp-images-') as directory:
        output = Path(directory)
        for name, body in extracts.items():
            (output / (name + '.inc')).write_text(body)
        fixture = output / 'fixture.cpp'
        fixture.write_text(FIXTURE)
        for variant, flags in (('normal', ['-O0']), ('fastmath', ['-O3', '-ffast-math'])):
            binary = output / variant
            command = shlex.split(os.environ.get('CXX', 'c++')) + [
                '-std=c++20', '-Wall', '-Wextra', '-Werror', '-UNDEBUG',
                '-Wno-missing-field-initializers', *flags, '-I', str(output),
                '-I', str(root / 'src'),
                '-I/usr/include/libdrm', str(fixture), '-o', str(binary),
            ]
            subprocess.run(command, check=True, timeout=60)
            for profile in PROFILES:
                subprocess.run([str(binary), profile], check=True, timeout=10)
            print(variant + ': temporary-image contracts passed', flush=True)


PROFILES = ('modern', 'legacy', 'extension', 'no-bgra-write', 'no-sampled', 'no-storage',
            'no-linear', 'no-write', 'no-view-sampled', 'no-view-linear')
FIXTURE = r'''
#include <vulkan/vulkan.h>
#include <drm_fourcc.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "Utils/Defer.h"
#include "colorspaces.inc"
#include "blur.inc"

static std::string profile;
static VkFormat DRMFormatToVulkan(uint32_t drm, bool srgb) {
    switch (drm) {
    case DRM_FORMAT_ARGB8888: return srgb ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM;
    case DRM_FORMAT_ABGR8888: return srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    case DRM_FORMAT_ABGR2101010: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case DRM_FORMAT_ABGR16161616F: return VK_FORMAT_R16G16B16A16_SFLOAT;
    default: return VK_FORMAT_UNDEFINED;
    }
}
struct FakeVk {
    unsigned queries=0, flags2Queries=0;
    void GetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties *properties) {
        properties->apiVersion=(profile=="legacy" || profile=="extension") ? VK_API_VERSION_1_2 : VK_API_VERSION_1_3;
    }
    void GetPhysicalDeviceFormatProperties2(VkPhysicalDevice, VkFormat format, VkFormatProperties2 *properties) {
        ++queries;
        VkFormatFeatureFlags2 features=VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT
            | VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT | VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
        const bool view=format==VK_FORMAT_B8G8R8A8_SRGB || format==VK_FORMAT_R8G8B8A8_SRGB;
        if (view) features &= ~(VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT);
        if (profile=="no-sampled" || (view && profile=="no-view-sampled")) features &= ~VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;
        if (profile=="no-storage") features &= ~VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT;
        if (profile=="no-linear" || (view && profile=="no-view-linear")) features &= ~VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        if (profile=="no-write") features &= ~VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
        if (profile=="no-bgra-write" && format==VK_FORMAT_B8G8R8A8_UNORM) features &= ~VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
        properties->formatProperties.optimalTilingFeatures=static_cast<VkFormatFeatureFlags>(features);
        properties->formatProperties.linearTilingFeatures=0; // Optimal images cannot use linear-tiling eligibility.
        if (properties->pNext) {
            assert(profile!="legacy");
            auto *extended=static_cast<VkFormatProperties3 *>(properties->pNext);
            assert(extended->sType==VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3);
            extended->optimalTilingFeatures=features;
            extended->linearTilingFeatures=0;
            ++flags2Queries;
        }
    }
    void GetPhysicalDeviceFormatProperties(VkPhysicalDevice device, VkFormat format, VkFormatProperties *properties) {
        VkFormatProperties2 result{};
        GetPhysicalDeviceFormatProperties2(device,format,&result);
        *properties=result.formatProperties;
    }
};
struct FakeDevice {
    FakeVk vk;
    std::vector<VkExtensionProperties> extensions;
    VkPhysicalDevice physDev() const { return VK_NULL_HANDLE; }
    std::vector<VkExtensionProperties> &supportedExtensions() { return extensions; }
} g_device;
namespace gamescope {
template<class T> struct Rc : std::shared_ptr<T> {
    Rc()=default;
    Rc(T *value) : std::shared_ptr<T>(value) {}
    using std::shared_ptr<T>::operator=;
    Rc &operator=(std::nullptr_t) { this->reset(); return *this; }
};
template<class T> using OwningRc=Rc<T>;
}
struct CVulkanTexture {
#include "flags.inc"
    static inline unsigned attempts=0, alive=0;
    static inline bool fail=false;
    uint32_t w=0,h=0,drm=DRM_FORMAT_INVALID;
    CVulkanTexture() { ++alive; }
    ~CVulkanTexture() { --alive; }
    uint32_t width() const { return w; }
    uint32_t height() const { return h; }
    uint32_t drmFormat() const { return drm; }
    VkFormat format() const { return DRMFormatToVulkan(drm,false); }
    bool BInit(uint32_t width, uint32_t height, uint32_t depth, uint32_t format, createFlags flags, void *dma=nullptr) {
        ++attempts; w=width; h=height; drm=format;
        assert(depth==1 && !dma && flags.bSampled && flags.bStorage);
        assert(!flags.bMappable && !flags.bLinear && !flags.bFlippable && !flags.bExportable);
        return !fail;
    }
};
struct { gamescope::Rc<CVulkanTexture> tmpOutput; uint32_t uOutputFormat=DRM_FORMAT_ARGB8888; } g_output;
struct Logger { template<class... A> void errorf(const char *, A...) {} } vk_log;
uint32_t currentOutputWidth=800,currentOutputHeight=600;
struct FrameInfo_t {
    struct Layer {
        GamescopeAppTextureColorspace colorspace=GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB;
        uint32_t w=320,h=200;
        gamescope::Rc<CVulkanTexture> tex;
        uint32_t integerWidth() const { return w; }
        uint32_t integerHeight() const { return h; }
    };
    struct Layers {
        Layer base;
        unsigned size=1;
        const Layer &get(unsigned index) const { assert(index==0 && size); return base; }
        unsigned count() const { return size; }
        bool empty() const { return size==0; }
    } layers;
    bool useFSRLayer0=false,useSGSRLayer0=false,useNISLayer0=false;
    BlurMode blurLayer0=BLUR_MODE_OFF;
};
#include "owner.inc"

static void fallback_path(FrameInfo_t *frameInfo, bool early, bool expectedFallback,
                          gamescope::Rc<CVulkanTexture> staged) {
#include "fallback.inc"
    assert(frameInfo->useFSRLayer0==!expectedFallback && frameInfo->useSGSRLayer0==!expectedFallback
        && frameInfo->useNISLayer0==!expectedFallback);
    assert(frameInfo->blurLayer0==(expectedFallback ? BLUR_MODE_OFF : BLUR_MODE_ALWAYS));
    frameInfo->layers.base.tex=std::move(staged);
    if (early) return;
    assert(frameInfo->layers.base.tex);
}
struct Focus { int eUpscaleFilter=0,eUpscaleScaler=0; };
struct Commit {
    bool pAcquirePoint=true;
    unsigned calls=0;
    bool ShouldPreemptivelyUpscale(int,int) { ++calls; return true; }
};
static bool cv_upscale_preemptive=true;
static std::string g_reshade_effect;
static bool preemptive(Commit *newCommit, Focus *pUpscaleFocus) {
#include "preemptive.inc"
    return bPreemptiveUpscale;
}
struct PooledImage_t {};
static unsigned poolCalls=0;
static uint32_t g_nOutputWidth=800,g_nOutputHeight=600;
static PooledImage_t *GetTempUpscaleImage(Focus *,uint32_t,uint32_t,uint32_t) {
    ++poolCalls; static PooledImage_t image; return &image;
}
static PooledImage_t *pool(FrameInfo_t upscaledFrameInfo) {
    Focus focus; Focus *pUpscaleFocus=&focus;
#include "pool.inc"
    return pTempImage;
}
static void caller_contracts(bool supported) {
    for (bool early : {false,true}) {
        FrameInfo_t frame;
        frame.useFSRLayer0=frame.useSGSRLayer0=frame.useNISLayer0=true;
        frame.blurLayer0=BLUR_MODE_ALWAYS;
        gamescope::Rc<CVulkanTexture> staged=new CVulkanTexture();
        g_output.tmpOutput=nullptr; CVulkanTexture::fail=true;
        fallback_path(&frame,early,true,staged);
        assert(frame.useFSRLayer0 && frame.useSGSRLayer0 && frame.useNISLayer0 && frame.blurLayer0==BLUR_MODE_ALWAYS);
        assert(frame.layers.base.tex==staged); // Scalar restoration must preserve staged layer mutations.
        if (supported) {
            CVulkanTexture::fail=false; g_output.tmpOutput=nullptr;
            fallback_path(&frame,early,false,staged);
            assert(frame.layers.base.tex==staged);
        }
    }
    Commit commit; Focus focus;
    g_reshade_effect="active";
    assert(!preemptive(&commit,&focus) && commit.calls==0);
    g_reshade_effect.clear();
    assert(preemptive(&commit,&focus) && commit.calls==1);
    assert(!preemptive(&commit,nullptr) && commit.calls==1);
    commit.pAcquirePoint=false;
    assert(!preemptive(&commit,&focus) && commit.calls==1);
    commit.pAcquirePoint=true; cv_upscale_preemptive=false;
    assert(!preemptive(&commit,&focus) && commit.calls==1);
    cv_upscale_preemptive=true;
    FrameInfo_t frame; frame.useFSRLayer0=true;
    CVulkanTexture::fail=true; g_output.tmpOutput=nullptr; poolCalls=0;
    assert(!pool(frame) && poolCalls==0);
    if (supported) {
        CVulkanTexture::fail=false;
        assert(pool(frame) && poolCalls==1);
    }
    CVulkanTexture::fail=false;
}

int main(int argc, char **argv) {
    assert(argc==2); profile=argv[1];
    if (profile=="extension") {
        VkExtensionProperties extension{};
        std::strcpy(extension.extensionName,VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME);
        g_device.extensions.push_back(extension);
    }
    const bool rawBad=profile=="no-sampled" || profile=="no-storage" || profile=="no-linear" || profile=="no-write";
    const bool viewBad=profile=="no-view-sampled" || profile=="no-view-linear";
    for (uint32_t drm : {DRM_FORMAT_ARGB8888,DRM_FORMAT_ABGR8888,DRM_FORMAT_ABGR2101010,DRM_FORMAT_ABGR16161616F}) {
        const bool sdr=drm==DRM_FORMAT_ARGB8888 || drm==DRM_FORMAT_ABGR8888;
        const bool expected=!rawBad && !(sdr && viewBad)
            && !((profile=="legacy" || profile=="no-bgra-write") && drm==DRM_FORMAT_ARGB8888);
        assert(tmp_image_format_supported(drm)==expected);
        const unsigned queries=g_device.vk.queries;
        assert(tmp_image_format_supported(drm)==expected && g_device.vk.queries==queries);
    }
    assert(!tmp_image_format_supported(DRM_FORMAT_INVALID));
    assert((g_device.vk.flags2Queries==0)==(profile=="legacy"));
    const uint32_t sdr=rawBad || viewBad || profile=="legacy" || profile=="no-bgra-write" ? DRM_FORMAT_ABGR8888 : DRM_FORMAT_ARGB8888;
    assert(tmp_image_format(GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB)==sdr);
    assert(tmp_image_format(GAMESCOPE_APP_TEXTURE_COLORSPACE_HDR10_PQ)==DRM_FORMAT_ABGR2101010);
    assert(tmp_image_format(GAMESCOPE_APP_TEXTURE_COLORSPACE_SCRGB)==DRM_FORMAT_ABGR16161616F);
    FrameInfo_t idle;
    assert(vulkan_prepare_composite_intermediate(&idle) && !g_output.tmpOutput && CVulkanTexture::attempts==0);
    idle.useFSRLayer0=true; idle.layers.size=0;
    assert(!vulkan_prepare_composite_intermediate(&idle) && !g_output.tmpOutput && CVulkanTexture::attempts==0);
    if (rawBad || viewBad) {
        assert(!update_tmp_images(320,200,DRM_FORMAT_ARGB8888));
        assert(!g_output.tmpOutput && CVulkanTexture::attempts==0);
        FrameInfo_t unsupported; unsupported.useFSRLayer0=true;
        assert(!vulkan_prepare_composite_intermediate(&unsupported) && !g_output.tmpOutput);
        assert(CVulkanTexture::attempts==0);
        if (viewBad) {
            assert(update_tmp_images(320,200,DRM_FORMAT_ABGR16161616F));
            const auto retained=g_output.tmpOutput;
            const unsigned attempts=CVulkanTexture::attempts;
            assert(!vulkan_prepare_composite_intermediate(&unsupported));
            assert(g_output.tmpOutput==retained && CVulkanTexture::attempts==attempts);
        }
    } else {
        assert(update_tmp_images(320,200,sdr));
        const auto old=g_output.tmpOutput; const unsigned attempts=CVulkanTexture::attempts;
        assert(update_tmp_images(320,200,sdr) && g_output.tmpOutput==old && CVulkanTexture::attempts==attempts);
        for (uint32_t drm : {DRM_FORMAT_ABGR2101010,DRM_FORMAT_ABGR16161616F,sdr}) {
            assert(update_tmp_images(320,200,drm));
            assert(g_output.tmpOutput->drmFormat()==drm);
        }
        assert(old->drmFormat()==sdr && old->width()==320); // Publication cannot invalidate a held old reference.
        const auto retained=g_output.tmpOutput;
        CVulkanTexture::fail=true;
        const unsigned beforeFailure=CVulkanTexture::alive;
        assert(!update_tmp_images(321,201,sdr) && g_output.tmpOutput==retained);
        assert(CVulkanTexture::alive==beforeFailure); // Failed local replacement is destroyed, held image survives.
        g_output.tmpOutput=nullptr;
        assert(!update_tmp_images(321,201,sdr) && !g_output.tmpOutput);
        CVulkanTexture::fail=false;
        FrameInfo_t frame;
        const unsigned noPassAttempts=CVulkanTexture::attempts;
        assert(vulkan_prepare_composite_intermediate(&frame) && !g_output.tmpOutput && CVulkanTexture::attempts==noPassAttempts);
        for (unsigned path=0;path<4;++path) {
            frame.useFSRLayer0=path==0;frame.useSGSRLayer0=path==1;frame.useNISLayer0=path==2;frame.blurLayer0=path==3 ? BLUR_MODE_ALWAYS : BLUR_MODE_OFF;
            g_output.tmpOutput=nullptr;
            assert(vulkan_prepare_composite_intermediate(&frame));
            assert(g_output.tmpOutput->width()==(path==3 ? currentOutputWidth : 320));
            assert(g_output.tmpOutput->height()==(path==3 ? currentOutputHeight : 200));
        }
        frame.useFSRLayer0=true; frame.blurLayer0=BLUR_MODE_ALWAYS;
        assert(vulkan_prepare_composite_intermediate(&frame));
        assert(g_output.tmpOutput->width()==320 && g_output.tmpOutput->height()==200);
    }
    caller_contracts(!rawBad && !viewBad);
    std::printf("%s PASS\n",argv[1]);
}
'''


if __name__ == '__main__':
    main()
