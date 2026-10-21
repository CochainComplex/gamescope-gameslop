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
    for name, body in (("views", views), ("create_flags", flags), ("usage", usage),
                       ("flags", create_flags), ("capability", capability), ("guard", guard),
                       ("external", external), ("pool", pool)):
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
namespace gamescope { template<class T> using Rc = T*; }
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
    bool BInit(uint32_t width,uint32_t height,uint32_t,uint32_t drm,createFlags f,void*,uint32_t contentW,uint32_t contentH) {
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
int main() {
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
                         "\nint CVulkanTexture::memoryFence()")
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
                     "\nstatic void framegen_record_net(")
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


if __name__ == "__main__":
    main()
