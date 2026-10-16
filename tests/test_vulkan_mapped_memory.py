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


def check_nv12_contracts(source, directory):
    """Execute production view/export/pool branches with real Vulkan ABI types."""
    init = between(source, "bool CVulkanTexture::BInit(", "bool CVulkanTexture::BInitFromSwapchain(")
    views = between(init, "\tif ( flags.bStorage || flags.bSampled || flags.bColorAttachment )",
                    "\n\tif ( flags.bMappable )\n")
    flags = between(init, "\tif ( drmFormat == DRM_FORMAT_NV12 && flags.bStorage )",
                    "\n\tif ( pDMA != nullptr )")
    external = between(init, "\tif ( pDMA != nullptr || flags.bExportable )",
                       "\n\tm_width = width;")
    pool = between(source, "static gamescope::Rc<CVulkanTexture> acquire_pooled_texture(",
                   "\ngamescope::Rc<CVulkanTexture> vulkan_acquire_screenshot_texture(")
    for name, body in (("views", views), ("flags", flags), ("external", external), ("pool", pool)):
        (directory / f"nv12_{name}.inc").write_text(body)
    fixture = r'''
#include <vulkan/vulkan.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>
constexpr uint32_t DRM_FORMAT_NV12 = 1, RGB = 2;
using EStreamColorspace = int;
namespace gamescope { template<class T> using Rc = T*; }
static VkFormat DRMFormatToVulkan(uint32_t format, bool) {
    return format == DRM_FORMAT_NV12 ? VK_FORMAT_G8_B8R8_2PLANE_420_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
}
static VkImageViewType VulkanImageTypeToViewType(VkImageType) { return VK_IMAGE_VIEW_TYPE_2D; }
static void vk_errorf(VkResult, const char*) {}
struct Observed { VkFormat format; VkImageAspectFlags aspect; VkImageUsageFlags usage; bool conversion; };
struct Device {
    unsigned fail = 0;
    std::vector<Observed> views;
    VkDevice device() { return VK_NULL_HANDLE; }
    VkSamplerYcbcrConversion ycbcrConversion() { return reinterpret_cast<VkSamplerYcbcrConversion>(uintptr_t{71}); }
    struct Calls {
        VkResult CreateImageView(VkDevice, const VkImageViewCreateInfo* info, void*, VkImageView* view);
    } vk;
} g_device;
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
    struct createFlags {
        bool bStorage=false, bSampled=false, bColorAttachment=false, bMappable=false;
        bool bTransferDst=false, bExportable=false, bLinear=false;
        VkImageType imageType=VK_IMAGE_TYPE_2D;
    } requested;
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
        imageInfo.usage = (flags.bStorage ? VkImageUsageFlags(VK_IMAGE_USAGE_STORAGE_BIT) : 0u) |
                          (flags.bSampled ? VkImageUsageFlags(VK_IMAGE_USAGE_SAMPLED_BIT) : 0u);
#include "nv12_views.inc"
        return true;
    }
};
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
        VkImageCreateInfo imageInfo{};
#include "nv12_flags.inc"
        assert(bool(imageInfo.flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT) == (drmFormat==DRM_FORMAT_NV12 && storage));
        assert(bool(imageInfo.flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) == (drmFormat==DRM_FORMAT_NV12 && storage));
    }
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
        check_nv12_contracts(source, directory)


if __name__ == "__main__":
    main()
