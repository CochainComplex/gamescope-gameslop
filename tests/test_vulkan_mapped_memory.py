#!/usr/bin/env python3
"""Exercise production image allocation/addressing with CPU-only Vulkan fixtures."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def between(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


def check_host_upload_contracts(source, header, directory):
    """Execute actual host-upload ownership guard and image-create declaration."""
    flags = between(header, "\tstruct createFlags {", "\n\tbool BInit(")
    init = between(source, "bool CVulkanTexture::BInit(", "bool CVulkanTexture::BInitFromSwapchain(")
    guard = between(init, "\tif ( flags.bHostUpload", "\n\tflags.bFlippable =")
    create = between(init, "\tVkImageCreateInfo imageInfo = {", "\n\tm_bConcurrentSharing =")
    tiling = next(line for line in init.splitlines() if "VkImageTiling tiling =" in line)
    staging = between(source, "\t\t\t\tCVulkanTexture::createFlags stagingFlags;",
                      "\n\t\t\t\tg_framegenMotion.netWeightsUpload = new")
    for name, body in (("flags", flags), ("guard", guard), ("create", create), ("staging", staging)):
        (directory / f"host_upload_{name}.inc").write_text(body)
    (directory / "host_upload_tiling.inc").write_text(tiling + "\n")
    fixture = r'''
#include <vulkan/vulkan.h>
#include <cassert>
#include <cstdint>
#include <initializer_list>
struct CVulkanTexture {
#include "host_upload_flags.inc"
};
static VkFormat DRMFormatToVulkan(uint32_t, bool) { return VK_FORMAT_R32_SFLOAT; }
static bool create(CVulkanTexture::createFlags flags, VkImageCreateInfo &observed,
                   bool imported=false, bool reused=false, bool backend=false) {
    const void *pDMA=imported ? &flags : nullptr;
    const void *pExistingImageToReuseMemory=reused ? &flags : nullptr;
    const void *pBackendFb=backend ? &flags : nullptr;
#include "host_upload_guard.inc"
    const uint32_t width=17, height=9, depth=1, drmFormat=0, nSharedQueueFamilyCount=0;
    const uint32_t nSharedQueueFamilies[1]={0};
    const VkImageUsageFlags usage=flags.imageUsage();
#include "host_upload_tiling.inc"
#include "host_upload_create.inc"
    observed=imageInfo;
    return true;
}
int main() {
    CVulkanTexture::createFlags defaults;
    assert(!defaults.bHostUpload);
    VkImageCreateInfo observed{};
    for (unsigned mask=0; mask<64; ++mask) {
        auto normal=defaults;
        normal.bMappable=mask&1; normal.bTransferSrc=mask&2; normal.bTransferDst=mask&4;
        normal.bLinear=mask&8; normal.bStorage=mask&16; normal.bSampled=mask&32;
        assert(create(normal,observed) && observed.initialLayout==VK_IMAGE_LAYOUT_UNDEFINED);
        assert(observed.tiling==((normal.bMappable || normal.bLinear) ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL));
    }
    for (bool exported : {false,true}) {
        auto capture=CVulkanTexture::createFlags::LinearCapture(exported);
        assert(!capture.bHostUpload && create(capture,observed) && observed.initialLayout==VK_IMAGE_LAYOUT_UNDEFINED);
    }
#include "host_upload_staging.inc"
    assert(stagingFlags.bHostUpload && stagingFlags.bMappable && stagingFlags.bTransferSrc);
    assert(create(stagingFlags,observed) && observed.initialLayout==VK_IMAGE_LAYOUT_PREINITIALIZED);
    assert(observed.tiling==VK_IMAGE_TILING_LINEAR && observed.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    for (int rejection=0; rejection<7; ++rejection) {
        auto invalid=stagingFlags;
        if (rejection==0) invalid.bMappable=false;
        if (rejection==1) invalid.bTransferSrc=false;
        if (rejection==2) invalid.bExportable=true;
        if (rejection==3) invalid.bFlippable=true;
        assert(!create(invalid,observed,rejection==4,rejection==5,rejection==6));
    }
}
'''
    path = directory / 'host_upload.cpp'
    path.write_text(fixture)
    binary = directory / 'host_upload'
    subprocess.run(shlex.split(os.environ.get('CXX', 'c++')) + [
        '-std=c++20', '-Wall', '-Wextra', '-Werror', '-Wno-missing-field-initializers',
        '-I', str(directory), str(path), '-o', str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=10)


def check_copy_image_contracts(source, header, directory):
    """Execute production region selection; record its region/owner handoff."""
    body = between(source, "void CVulkanCmdBuffer::copyImage(",
                   "\nvoid CVulkanCmdBuffer::copyImageRegions(")
    (directory / "copy_image.inc").write_text(body)
    roundup = between(header, "static inline uint32_t div_roundup(", "\n}") + "\n}\n"
    (directory / "copy_roundup.inc").write_text(roundup)
    fixture = r'''
#include <vulkan/vulkan.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>
namespace gamescope { template<class T> using Rc = std::shared_ptr<T>; }
#include "copy_roundup.inc"
constexpr uint32_t DRM_FORMAT_NV12 = 842094158;
struct CVulkanTexture {
    VkFormat pixel_format;
    uint32_t w, h;
    // Deliberately unrelated byte pitches: VkImageCopy uses texel extents.
    uint32_t luma_pitch, chroma_pitch;
    uint32_t width() const { return w; }
    uint32_t height() const { return h; }
    VkFormat format() const { return pixel_format; }
    bool isYcbcr() const { return pixel_format == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM; }
    uint32_t drmFormat() const { return isYcbcr() ? DRM_FORMAT_NV12 : 0; }
};
struct CVulkanCmdBuffer {
    unsigned calls = 0;
    std::vector<VkImageCopy> regions;
    std::vector<gamescope::Rc<CVulkanTexture>> owners;
    void copyImage(gamescope::Rc<CVulkanTexture> src, gamescope::Rc<CVulkanTexture> dst);
    void copyImageRegions(gamescope::Rc<CVulkanTexture> src, gamescope::Rc<CVulkanTexture> dst,
                          std::span<const VkImageCopy> copies) {
        ++calls;
        regions.assign(copies.begin(), copies.end());
        owners.emplace_back(std::move(src)); owners.emplace_back(std::move(dst));
    }
};
#include "copy_image.inc"
int main() {
    for (VkFormat format : {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_FORMAT_R32_SFLOAT, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM}) {
        for (const auto size : {std::array<uint32_t, 2>{800, 600}, {801, 601}, {1, 1}, {2, 3}}) {
            for (uint32_t padding : {0u, 96u}) {
                auto src = std::make_shared<CVulkanTexture>(CVulkanTexture{format, size[0], size[1], size[0]+padding, size[0]+2*padding});
                auto dst = std::make_shared<CVulkanTexture>(CVulkanTexture{format, size[0], size[1], size[0]+3*padding, size[0]+4*padding});
                std::weak_ptr<CVulkanTexture> src_owner = src, dst_owner = dst;
                CVulkanCmdBuffer cmd;
                cmd.copyImage(src, dst);
                assert(cmd.calls == 1 && cmd.owners.size() == 2);
                assert(cmd.owners[0].get() == src.get() && cmd.owners[1].get() == dst.get());
                const bool nv12 = format == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
                assert(cmd.regions.size() == (nv12 ? 2u : 1u));
                for (size_t i = 0; i < cmd.regions.size(); ++i) {
                    const auto &region = cmd.regions[i];
                    const VkImageAspectFlags aspect = nv12 ? (i ? VK_IMAGE_ASPECT_PLANE_1_BIT : VK_IMAGE_ASPECT_PLANE_0_BIT) : VK_IMAGE_ASPECT_COLOR_BIT;
                    for (const auto &subresource : {region.srcSubresource, region.dstSubresource}) {
                        assert(subresource.aspectMask == aspect);
                        assert(subresource.mipLevel == 0 && subresource.baseArrayLayer == 0 && subresource.layerCount == 1);
                    }
                    for (const auto &offset : {region.srcOffset, region.dstOffset})
                        assert(offset.x == 0 && offset.y == 0 && offset.z == 0);
                    assert(region.extent.width == (i ? (size[0]+1)/2 : size[0]));
                    assert(region.extent.height == (i ? (size[1]+1)/2 : size[1]));
                    assert(region.extent.depth == 1);
                }
                src.reset(); dst.reset();
                assert(!src_owner.expired() && !dst_owner.expired());
                cmd.owners.clear();
                assert(src_owner.expired() && dst_owner.expired());
            }
        }
    }
}
'''
    path = directory / "copy_image.cpp"
    path.write_text(fixture)
    binary = directory / "copy_image"
    subprocess.run(shlex.split(os.environ.get("CXX", "c++"))
                   + ["-std=c++20", "-Wall", "-Wextra", "-Werror",
                      "-Wno-missing-field-initializers", "-I", str(directory), str(path),
                      "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=10)


def check_nv12_contracts(source, header, directory):
    """Execute production view/export/pool branches with real Vulkan ABI types."""
    init = between(source, "bool CVulkanTexture::BInit(", "bool CVulkanTexture::BInitFromSwapchain(")
    views = between(init, "\tif ( flags.bStorage || flags.bSampled || flags.bColorAttachment )",
                    "\n\tif ( flags.bMappable )\n")
    flags = between(header, "\tstruct createFlags {", "\n\tbool BInit(")
    usage = next(line for line in init.splitlines() if "VkImageUsageFlags usage = flags.imageUsage();" in line)
    create_flags = next(line for line in init.splitlines() if "imageInfo.flags = flags.imageCreateFlags(" in line)
    capability = between(source, "static VkResult getExternalImageFormatProps(",
                         "\nstatic VkImageViewType VulkanImageTypeToViewType(")
    guard = between(init, "\tif ( flags.bMappable && !flags.bFlippable && flags.bExportable && pDMA == nullptr && tiling != VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT &&",
                    "\n\tres = g_device.vk.CreateImage(")
    external = between(init, "\tif ( pDMA != nullptr || flags.bExportable )",
                       "\n\tm_width = width;")
    pool = between(source, "static gamescope::Rc<CVulkanTexture> acquire_pooled_texture(",
                   "\ngamescope::Rc<CVulkanTexture> vulkan_acquire_screenshot_texture(")
    output_factory = between(source, "static bool framegen_create_output_texture(",
                             "\n// Base-mode generated frames:")
    for name, body in (("views", views), ("create_flags", flags), ("usage", usage),
                       ("flags", create_flags), ("capability", capability), ("guard", guard),
                       ("external", external), ("pool", pool),
                       ("output_factory", output_factory)):
        (directory / f"nv12_{name}.inc").write_text(body)
    fixture = r'''
#include <vulkan/vulkan.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <cinttypes>
#include <utility>
#include <vector>
constexpr uint32_t DRM_FORMAT_NV12 = 1, RGB = 2;
constexpr uint64_t DRM_FORMAT_MOD_INVALID = ~uint64_t{0};
using EStreamColorspace = int;
namespace gamescope {
template<class T> using Rc = T*;
template<class T> using OwningRc = T*;
}
static VkFormat DRMFormatToVulkan(uint32_t format, bool srgb) {
    return format == DRM_FORMAT_NV12 ? VK_FORMAT_G8_B8R8_2PLANE_420_UNORM : srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
}
static VkImageViewType VulkanImageTypeToViewType(VkImageType) { return VK_IMAGE_VIEW_TYPE_2D; }
static void vk_errorf(VkResult, const char*) {}
struct Log { template<class... Args> void errorf(const char*,Args...) {} } vk_log;
template<class Target,class Base> static const Target* pNextFind(const Base* base,VkStructureType type) {
    for (auto p=static_cast<const VkBaseInStructure*>(base->pNext);p;p=p->pNext)
        if (p->sType==type) return reinterpret_cast<const Target*>(p);
    return nullptr;
}
struct Observed { VkFormat format; VkImageAspectFlags aspect; VkImageUsageFlags usage; bool conversion; };
struct Device {
    unsigned fail = 0;
    VkResult query_result=VK_SUCCESS;
    VkExternalMemoryFeatureFlags external_features=VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT;
    VkExternalMemoryHandleTypeFlags compatible=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkImageTiling query_tiling{};
    VkImageUsageFlags query_usage{};
    VkImageCreateFlags query_flags{};
    bool query_modifier=false, query_formats=false;
    std::vector<uint32_t> query_families;
    std::vector<Observed> views;
    VkDevice device() { return VK_NULL_HANDLE; }
    VkPhysicalDevice physDev() { return VK_NULL_HANDLE; }
    VkSamplerYcbcrConversion ycbcrConversion() { return reinterpret_cast<VkSamplerYcbcrConversion>(uintptr_t{71}); }
    struct Calls {
        VkResult CreateImageView(VkDevice, const VkImageViewCreateInfo* info, void*, VkImageView* view);
        VkResult GetPhysicalDeviceImageFormatProperties2(VkPhysicalDevice,const VkPhysicalDeviceImageFormatInfo2*,VkImageFormatProperties2*);
    } vk;
} g_device;
VkResult Device::Calls::GetPhysicalDeviceImageFormatProperties2(VkPhysicalDevice,const VkPhysicalDeviceImageFormatInfo2* info,VkImageFormatProperties2* out) {
    g_device.query_tiling=info->tiling; g_device.query_usage=info->usage; g_device.query_flags=info->flags;
    auto external=pNextFind<VkPhysicalDeviceExternalImageFormatInfo>(info,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
    assert(external && external->handleType==VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    auto modifier=pNextFind<VkPhysicalDeviceImageDrmFormatModifierInfoEXT>(info,VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT);
    g_device.query_modifier=modifier;
    g_device.query_families.clear();
    if (modifier) {
        assert(modifier->sharingMode==VK_SHARING_MODE_CONCURRENT);
        g_device.query_families.assign(modifier->pQueueFamilyIndices,modifier->pQueueFamilyIndices+modifier->queueFamilyIndexCount);
    }
    g_device.query_formats=pNextFind<VkImageFormatListCreateInfo>(info,VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO);
    auto properties=static_cast<VkExternalImageFormatProperties*>(out->pNext);
    properties->externalMemoryProperties.externalMemoryFeatures=g_device.external_features;
    properties->externalMemoryProperties.compatibleHandleTypes=g_device.compatible;
    return g_device.query_result;
}
VkResult Device::Calls::CreateImageView(VkDevice, const VkImageViewCreateInfo* info, void*, VkImageView* view) {
    VkImageUsageFlags usage = 0;
    bool conversion = false;
    for (auto p = static_cast<const VkBaseInStructure*>(info->pNext); p; p = p->pNext) {
        if (p->sType == VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO)
            usage = reinterpret_cast<const VkImageViewUsageCreateInfo*>(p)->usage;
        if (p->sType == VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO) {
            conversion = true;
            assert(reinterpret_cast<const VkSamplerYcbcrConversionInfo*>(p)->conversion == g_device.ycbcrConversion());
            assert(info->components.a == VK_COMPONENT_SWIZZLE_IDENTITY);
        }
    }
    g_device.views.push_back({info->format, info->subresourceRange.aspectMask, usage, conversion});
    if (g_device.fail == g_device.views.size()) return VK_ERROR_OUT_OF_HOST_MEMORY;
    *view = reinterpret_cast<VkImageView>(uintptr_t{100} + g_device.views.size());
    return VK_SUCCESS;
}
class CVulkanTexture {
public:
#include "nv12_create_flags.inc"
    createFlags requested;
    uint32_t w=0,h=0,cw=0,ch=0,format=0;
    unsigned refs=0;
    int color=0;
    static inline bool failInit=false;
    bool BInit(uint32_t width,uint32_t height,uint32_t,uint32_t drm,createFlags f,void* = nullptr,uint32_t contentW = 0,uint32_t contentH = 0) {
        w=width; h=height; cw=contentW; ch=contentH; format=drm; requested=f; return !failInit;
    }
    uint32_t width() const { return w; } uint32_t height() const { return h; }
    uint32_t contentWidth() const { return cw; } uint32_t contentHeight() const { return ch; }
    uint32_t drmFormat() const { return format; } bool externalImage() const { return requested.bExportable; }
    unsigned GetRefCount() const { return refs; }
    int streamColorspace() const { return color; } void setStreamColorspace(int c) { color=c; }
    bool isYcbcr() const { return format == DRM_FORMAT_NV12; }
    VkImage m_vkImage = VK_NULL_HANDLE;
    VkImageView m_srgbView{},m_linearView{},m_lumaView{},m_chromaView{};
    bool views(createFlags flags) {
        const uint32_t drmFormat=format;
        VkResult res=VK_SUCCESS;
        bool bHasAlpha=false;
        VkImageCreateInfo imageInfo{};
#include "nv12_usage.inc"
        imageInfo.usage=usage;
#include "nv12_views.inc"
        return true;
    }
};
#include "nv12_capability.inc"
static bool export_guard(bool exportable,bool imported,VkImageTiling tiling,bool mapped=true,bool flippable=false) {
    auto flags=CVulkanTexture::createFlags::LinearCapture(exportable);
    flags.bMappable=mapped;flags.bFlippable=flippable;
    const void* pDMA=imported ? &flags : nullptr;
    const uint32_t drmFormat=DRM_FORMAT_NV12;
    VkImageCreateInfo imageInfo{};
    imageInfo.format=DRMFormatToVulkan(drmFormat,false);imageInfo.tiling=tiling;
    imageInfo.usage=flags.imageUsage();imageInfo.flags=flags.imageCreateFlags(imageInfo.format,imageInfo.format);
#include "nv12_guard.inc"
    return true;
}
struct Slot {
    CVulkanTexture* p=nullptr;
    ~Slot() { delete p; }
    Slot& operator=(CVulkanTexture* next) { delete p; p=next; return *this; }
    explicit operator bool() const { return p; }
    bool operator==(std::nullptr_t) const { return p==nullptr; }
    CVulkanTexture* get() const { return p; }
    CVulkanTexture* operator->() const { return p; }
};
#include "nv12_pool.inc"
static bool native_output=false, cursor_split=false, color_probe=false;
static bool framegen_owns_output_ring() { return native_output; }
static bool framegen_cursor_split_enabled() { return cursor_split; }
static bool framegen_color_probe_requested() { return color_probe; }
#include "nv12_output_factory.inc"
int main() {
    for (bool native : {false,true}) for (bool cursor : {false,true}) for (bool probe : {false,true}) {
        native_output=native;cursor_split=cursor;color_probe=probe;
        CVulkanTexture* output=nullptr;
        assert(framegen_create_output_texture(&output,640,480,RGB));
        const auto& flags=output->requested;
        assert(output->width()==640 && output->height()==480 && output->drmFormat()==RGB);
        // Only scanout-owned outputs may take the output/PRESENT path. Nested
        // intermediates remain sampleable with either cursor/probe setting.
        assert(flags.bOutputImage==native && flags.bFlippable==native);
        assert(flags.bSampled==(native ? cursor : true));
        assert(flags.bStorage && flags.bFramegenShared && flags.bTransferSrc==probe);
        assert(!flags.bTransferDst && !flags.bMappable && !flags.bLinear && !flags.bColorAttachment);
        const VkImageUsageFlags expected=VK_IMAGE_USAGE_STORAGE_BIT
            | (flags.bSampled ? VkImageUsageFlags(VK_IMAGE_USAGE_SAMPLED_BIT) : 0u)
            | (probe ? VkImageUsageFlags(VK_IMAGE_USAGE_TRANSFER_SRC_BIT) : 0u);
        assert(flags.imageUsage()==expected);
        delete output;
    }
    for (bool sampled : {false,true}) for (bool storage : {false,true}) {
        CVulkanTexture texture; texture.format=DRM_FORMAT_NV12;
        CVulkanTexture::createFlags f; f.bSampled=sampled; f.bStorage=storage;
        const unsigned count=2u*sampled+2u*storage;
        for (unsigned failure=0; failure<=count; ++failure) {
            g_device.views.clear(); g_device.fail=failure;
            assert(texture.views(f) == (failure==0));
            assert(g_device.views.size() == (failure ? failure : count));
            for (const auto& view : g_device.views) {
                if (view.aspect == VK_IMAGE_ASPECT_COLOR_BIT) {
                    assert(view.conversion && !(view.usage & VK_IMAGE_USAGE_STORAGE_BIT));
                } else {
                    assert(!view.conversion && view.usage == VK_IMAGE_USAGE_STORAGE_BIT);
                    assert(view.format == (view.aspect==VK_IMAGE_ASPECT_PLANE_0_BIT ? VK_FORMAT_R8_UNORM : VK_FORMAT_R8G8_UNORM));
                }
            }
        }
    }
    for (uint32_t drmFormat : {RGB,DRM_FORMAT_NV12}) for (bool storage : {false,true}) {
        CVulkanTexture::createFlags flags; flags.bStorage=storage;
        std::array<VkFormat,2> formats={DRMFormatToVulkan(drmFormat,false),DRMFormatToVulkan(drmFormat,true)};
        VkImageCreateInfo imageInfo{};
#include "nv12_flags.inc"
        assert(bool(imageInfo.flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT) == (drmFormat==DRM_FORMAT_NV12 && storage));
        assert(bool(imageInfo.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) == (drmFormat==RGB || storage));
        assert(flags.imageCreateFlags(formats[0],formats[0]) == (drmFormat==DRM_FORMAT_NV12 && storage ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT|VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0u));
    }
    const std::array<VkImageUsageFlags,5> bits={VK_IMAGE_USAGE_SAMPLED_BIT,VK_IMAGE_USAGE_STORAGE_BIT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,VK_IMAGE_USAGE_TRANSFER_SRC_BIT,VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    for (unsigned mask=0;mask<32;mask++) {
        CVulkanTexture::createFlags flags;
        flags.bSampled=mask&1;flags.bStorage=mask&2;flags.bColorAttachment=mask&4;
        flags.bTransferSrc=mask&8;flags.bTransferDst=mask&16;
        VkImageUsageFlags expected=0;
        for (unsigned bit=0;bit<5;bit++) if (mask&(1u<<bit)) expected|=bits[bit];
        assert(flags.imageUsage()==expected);
    }
    for (bool exported : {false,true}) {
        auto flags=CVulkanTexture::createFlags::LinearCapture(exported);
        assert(flags.bMappable && flags.bStorage && flags.bTransferDst && !flags.bSampled && !flags.bTransferSrc && !flags.bColorAttachment);
        assert(flags.bExportable==exported && flags.bLinear==exported && flags.imageType==VK_IMAGE_TYPE_2D);
    }
    for (int rejected=0;rejected<4;rejected++) {
        g_device.query_result=rejected==1 ? VK_ERROR_FORMAT_NOT_SUPPORTED : VK_SUCCESS;
        g_device.external_features=rejected==2 ? 0u : VkExternalMemoryFeatureFlags(VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT);
        g_device.compatible=rejected==3 ? 0u : VkExternalMemoryHandleTypeFlags(VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
        for (uint32_t format : {RGB,DRM_FORMAT_NV12}) {
            assert(vulkan_supports_pipewire_dmabuf(format)==(rejected==0));
            assert(!g_device.query_modifier && g_device.query_tiling==VK_IMAGE_TILING_LINEAR);
            assert(g_device.query_formats==(format==RGB));
            assert(g_device.query_usage==CVulkanTexture::createFlags::LinearCapture(true).imageUsage());
            assert(g_device.query_flags==CVulkanTexture::createFlags::LinearCapture(true).imageCreateFlags(DRMFormatToVulkan(format,false),DRMFormatToVulkan(format,true)));
        }
    }
    g_device.query_result=VK_SUCCESS;
    const uint32_t families[2]={1,3};
    VkImageCreateInfo query{};query.tiling=VK_IMAGE_TILING_LINEAR;query.sharingMode=VK_SHARING_MODE_CONCURRENT;
    query.queueFamilyIndexCount=2;query.pQueueFamilyIndices=families;
    VkExternalImageFormatProperties properties{};
    assert(getExternalImageFormatProps(&query,0,&properties)==VK_SUCCESS);
    assert(g_device.query_modifier && g_device.query_tiling==VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT);
    assert(g_device.query_families==std::vector<uint32_t>({1,3}));
    assert(getExternalImageFormatProps(&query,DRM_FORMAT_MOD_INVALID,&properties)==VK_SUCCESS);
    assert(!g_device.query_modifier && g_device.query_tiling==VK_IMAGE_TILING_LINEAR);
    g_device.query_result=VK_ERROR_FORMAT_NOT_SUPPORTED;
    assert(!export_guard(true,false,VK_IMAGE_TILING_LINEAR));
    assert(export_guard(false,false,VK_IMAGE_TILING_LINEAR));
    assert(export_guard(true,true,VK_IMAGE_TILING_LINEAR));
    assert(export_guard(true,false,VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT));
    // The capture guard must not reject legacy nonmapped or scanout exports.
    assert(export_guard(true,false,VK_IMAGE_TILING_LINEAR,false,false));
    assert(export_guard(true,false,VK_IMAGE_TILING_LINEAR,true,true));
    assert(export_guard(true,false,VK_IMAGE_TILING_LINEAR,false,true));
    g_device.query_result=VK_SUCCESS;
    g_device.external_features=VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT;
    g_device.compatible=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    assert(export_guard(true,false,VK_IMAGE_TILING_LINEAR));
    for (bool imported : {false,true}) for (bool exported : {false,true}) {
        CVulkanTexture::createFlags flags; flags.bExportable=exported;
        const void* pDMA=imported ? &flags : nullptr;
        VkImageCreateInfo imageInfo{};
        VkExternalMemoryImageCreateInfo externalImageCreateInfo{};
#include "nv12_external.inc"
        assert(bool(imageInfo.pNext) == (imported || exported));
        if (imageInfo.pNext) assert(externalImageCreateInfo.handleTypes == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    }
    std::array<Slot,1> pool;
    auto* first=acquire_pooled_texture(pool,5,3,false,DRM_FORMAT_NV12,1);
    assert(first && first->width()==6 && first->height()==4 && first->contentWidth()==5 && first->contentHeight()==3);
    assert(!first->requested.bSampled && first->requested.bStorage && !first->requested.bExportable && !first->requested.bLinear);
    assert(acquire_pooled_texture(pool,5,3,false,DRM_FORMAT_NV12,1)==first);
    auto* even=acquire_pooled_texture(pool,6,4,false,DRM_FORMAT_NV12,1);
    assert(even && even->contentWidth()==6 && even->contentHeight()==4);
    even=acquire_pooled_texture(pool,6,4,true,DRM_FORMAT_NV12,1);
    assert(even && even->externalImage());
    even=acquire_pooled_texture(pool,6,4,true,DRM_FORMAT_NV12,2);
    assert(even && even->streamColorspace()==2);
    assert(even->requested.bLinear);
    even->refs=1;
    assert(!acquire_pooled_texture(pool,6,4,true,DRM_FORMAT_NV12,2));
    even->refs=0;
    auto* rgb=acquire_pooled_texture(pool,7,5,false,RGB,1);
    assert(rgb && rgb->width()==7 && rgb->height()==5 && rgb->requested.bSampled);
    CVulkanTexture::failInit=true;
    assert(!acquire_pooled_texture(pool,8,8,false,RGB,1) && !pool[0]);
}
'''
    path = directory / "nv12_contracts.cpp"
    path.write_text(fixture)
    binary = directory / "nv12_contracts"
    subprocess.run(shlex.split(os.environ.get("CXX", "c++"))
                   + ["-std=c++20", "-Wall", "-Wextra", "-Werror",
                      "-Wno-missing-field-initializers", "-I", str(directory), str(path),
                      "-o", str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=10)


def main():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/rendervulkan.cpp").read_text()
    header = (root / "src/rendervulkan.hpp").read_text()
    compositor = (root / "src/steamcompmgr.cpp").read_text()
    init = between(source, "bool CVulkanTexture::BInit(",
                   "bool CVulkanTexture::BInitFromSwapchain(")
    properties = between(init, "\tif ( flags.bMappable == true )",
                         "\tif ( flags.bOutputImage == true )")
    allocation_info = between(init, "\tVkMemoryAllocateInfo allocInfo = {",
                              "\n\tm_size = allocInfo.allocationSize;")
    # Compile the ordinary-allocation branch itself, including its early return,
    # followed by the actual allocation call. The imported/reused paths are not
    # substituted with a second implementation of the allocation policy.
    selection = between(init, "\t\t\t// HOST_CACHED improves CPU readback,",
                        "\n\t\t}\n\t\t\n\t\tres = g_device.vk.AllocateMemory")
    allocate = next(line for line in init.splitlines()
                    if "res = g_device.vk.AllocateMemory(" in line)
    create = between(init, "\tres = g_device.vk.CreateImage(",
                     "\n\tVkMemoryRequirements memRequirements;")
    allocate_tail = between(init, allocate,
                            "\n\t}\n\telse\n\t{\n\t\tvk_log.infof")
    allocate_tail = allocate_tail[len(allocate):]
    reuse = between(init, '\t\tvk_log.infof("%d vs %d!",',
                    "\n\t}\n\t\n\tres = g_device.vk.BindImageMemory")
    destructor = between(source, "CVulkanTexture::~CVulkanTexture( void )",
                         "\nstatic bool is_image_format_modifier_supported(")
    swapchain = between(source, "bool CVulkanTexture::BInitFromSwapchain(",
                        "\ngamescope::Rc<CVulkanTexture> CVulkanTexture::AcquireDeviceLocalStagingImage()")
    find_type = between(source, "int32_t CVulkanDevice::findMemoryType(",
                        "\nstd::unique_ptr<CVulkanCmdBuffer> CVulkanDevice::commandBuffer(")
    texture_header = between(header, "class CVulkanTexture ", "\n};")
    members = [line for line in texture_header.splitlines()
               if line.startswith(("\tinline uint8_t *mappedData()",
                                   "\tinline uint8_t *mappedPixels()",
                                   "\tinline uint32_t rowPitch()",
                                   "\tinline VkDeviceSize totalSize()",
                                   "\tuint32_t m_unRowPitch =",
                                   "\tVkDeviceSize m_mappedOffset =",
                                   "\tVkDeviceSize m_size =",
                                   "\tuint32_t m_lumaOffset =",
                                   "\tuint32_t m_lumaPitch =",
                                   "\tuint32_t m_chromaOffset =",
                                   "\tuint32_t m_chromaPitch =",
                                   "\tbool m_bInitialized =",
                                   "\tbool m_bOwnsImage =",
                                   "\tbool m_bDeviceLocal =",
                                   "\tbool m_bOutputImage =",
                                   "\tVkImage m_vkImage =",
                                   "\tVkDeviceMemory m_vkImageMemory =",
                                   "\tVkImageView m_",
                                   "\tuint32_t m_drmFormat =",
                                   "\tuint32_t m_width =",
                                   "\tuint32_t m_height =",
                                   "\tuint32_t m_depth =",
                                   "\tuint32_t m_contentWidth =",
                                   "\tuint32_t m_contentHeight =",
                                   "\tVkFormat m_format =",
                                   "\tuint8_t *m_pMappedData ="))]
    layout = between(init, "\tif ( flags.bMappable == true )\n\t{\n\t\tassert( tiling",
                     "\n\tif ( flags.bExportable == true )")
    # Capture enables export independently of CPU mapping. Execute its actual
    # nonmodifier query and assignments instead of reproducing their policy.
    export_layout = between(init, "\n\t\t\tconst VkImageSubresource subresource = {\n",
                            "\n\t\t}\n\n\t\tm_dmabuf = dmabuf;")
    mapping = between(init, "\tif ( flags.bMappable )\n",
                      "\n\tm_bInitialized = true;")
    upload = between(source, "static void framegen_net_stage_weights(",
                     "\n}\n") + "\n}\n"
    # The completed-readback prefix owns addressing and row copies. Later
    # profile validation/persistence is outside this mapped-memory regression.
    readback = between(source, "static void framegen_net_profile_consume()\n{",
                       "\n\tfor ( uint32_t i = 0; i < k_uFramegenNetFloats; i++ )") + "\n}\n"
    screenshot_data = between(compositor, "\t\t\t\tconst uint8_t *mappedData = pScreenshotTexture->",
                              "\n\t\t\t\tconst uint32_t width =")
    screenshot_write = next(line for line in compositor.splitlines()
                            if "fwrite(" in line and "pScreenshotTexture->totalSize()" in line)
    with tempfile.TemporaryDirectory(prefix="gamescope-vk-memory-") as directory:
        directory = Path(directory)
        (directory / "memory_type.inc").write_text(find_type)
        (directory / "image_allocation.inc").write_text(
            properties + allocation_info + selection + "\n" + allocate + "\n")
        (directory / "image_create.inc").write_text(create)
        (directory / "image_allocation_tail.inc").write_text(allocate_tail)
        (directory / "image_reuse.inc").write_text(reuse)
        (directory / "image_destructor.inc").write_text(destructor)
        (directory / "image_swapchain.inc").write_text(swapchain)
        (directory / "mapped_members.inc").write_text("\n".join(members) + "\n")
        (directory / "mapped_layout.inc").write_text(layout)
        (directory / "export_layout.inc").write_text(export_layout)
        (directory / "mapped_mapping.inc").write_text(mapping)
        (directory / "mapped_upload.inc").write_text(upload)
        (directory / "mapped_readback.inc").write_text(readback)
        (directory / "screenshot_data.inc").write_text(screenshot_data + "\n")
        (directory / "screenshot_write.inc").write_text(screenshot_write + "\n")
        binary = directory / "test"
        subprocess.run(shlex.split(os.environ.get("CXX", "c++"))
                       + ["-std=c++20", "-Wall", "-Wextra", "-Werror",
                          "-Wno-missing-field-initializers", "-I", str(directory),
                          str(root / "tests/test_vulkan_mapped_memory.cpp"),
                          "-o", str(binary)], check=True, timeout=60)
        subprocess.run([str(binary)], check=True, timeout=10)
        check_nv12_contracts(source, header, directory)
        check_copy_image_contracts(source, header, directory)
        check_host_upload_contracts(source, header, directory)


if __name__ == "__main__":
    main()
