// Vulkan entry points and memory types are fixtures; selection/allocation are
// extracted verbatim from rendervulkan.cpp by test_vulkan_mapped_memory.py.
#include <array>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

using VkMemoryPropertyFlags = uint32_t;
using VkDeviceMemory = uint64_t;
using VkDeviceSize = uint64_t;
using VkImage = uint64_t;
using VkImageView = uint64_t;
using VkFormat = uint32_t;
using VkResult = int32_t;
constexpr VkImage VK_NULL_HANDLE = 0;
constexpr VkDeviceSize VK_WHOLE_SIZE = ~VkDeviceSize{0};
constexpr VkResult VK_SUCCESS = 0, VK_ERROR_FEATURE_NOT_PRESENT = -8;
constexpr uint32_t VK_IMAGE_ASPECT_COLOR_BIT = 1;
constexpr uint32_t VK_IMAGE_ASPECT_PLANE_0_BIT = 16, VK_IMAGE_ASPECT_PLANE_1_BIT = 32;
constexpr uint32_t VK_IMAGE_TILING_LINEAR = 1, VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT = 1000158000;
constexpr uint64_t DRM_FORMAT_MOD_INVALID = ~uint64_t{0};
constexpr uint32_t DRM_FORMAT_INVALID = 0, VK_FORMAT_UNDEFINED = 0;
constexpr uint32_t VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO = 15;
constexpr uint32_t VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO = 1000117002;
constexpr uint32_t VK_IMAGE_VIEW_TYPE_2D = 1, VK_COMPONENT_SWIZZLE_IDENTITY = 0;
constexpr uint32_t VK_IMAGE_USAGE_SAMPLED_BIT = 4;
constexpr uint32_t VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 2;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 4;
constexpr VkMemoryPropertyFlags VK_MEMORY_PROPERTY_HOST_CACHED_BIT = 8;
constexpr VkMemoryPropertyFlags coherent =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

static void require(bool condition, const char *message)
{
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

struct VkMemoryAllocateInfo
{
    uint32_t sType;
    const void *pNext;
    uint64_t allocationSize;
    uint32_t memoryTypeIndex;
};

struct VkImageSubresource { uint32_t aspectMask; };
struct VkSubresourceLayout { VkDeviceSize offset = 0, rowPitch = 0; };
struct VkImageCreateInfo {};
struct VkImageViewCreateInfo
{
    uint32_t sType;
    const void *pNext;
    VkImage image;
    uint32_t viewType;
    VkFormat format;
    struct { uint32_t r, g, b, a; } components;
    struct { uint32_t aspectMask, levelCount, layerCount; } subresourceRange;
};
struct VkImageViewUsageCreateInfo { uint32_t sType, usage; };
static uint32_t VulkanFormatToDRM(VkFormat format) { return format; }
static VkFormat ToLinearVulkanFormat(VkFormat format) { return format; }
static VkFormat ToSrgbVulkanFormat(VkFormat format) { return format; }
struct FixtureDmabuf
{
    int n_planes = 0;
    uint64_t modifier = 0;
    uint32_t offset[4] = {}, stride[4] = {};
};
static void wlr_dmabuf_attributes_finish(FixtureDmabuf *) {}

class CVulkanDevice
{
public:
    struct
    {
        uint32_t memoryTypeCount = 0;
        struct Type { VkMemoryPropertyFlags propertyFlags = 0; } memoryTypes[4];
    } m_memoryProperties;
    int32_t findMemoryType(VkMemoryPropertyFlags properties, uint32_t requiredTypeBits);
    void *device() { return nullptr; }
    bool completed = false;
    bool hasCompletedFramegen(uint64_t) { return completed; }
    struct
    {
        uint32_t calls = 0, selected = 0;
        uint32_t destroy_calls = 0;
        uint32_t map_calls = 0;
        uint32_t create_calls = 0, view_calls = 0, fail_view_call = 0;
        VkImage next_image = 1;
        VkResult create_result = VK_SUCCESS, allocate_result = VK_SUCCESS;
        std::vector<std::string> cleanup;
        void *mapped_base = nullptr;
        bool nv12 = false;
        VkSubresourceLayout color_layout{32, 20};
        VkSubresourceLayout luma_layout{16, 24}, chroma_layout{96, 28};
        std::vector<uint32_t> layout_queries;
        void GetImageSubresourceLayout(void *, VkImage image,
                                      const VkImageSubresource *subresource, VkSubresourceLayout *layout)
        {
            require(image == 1, "layout query used the wrong image");
            layout_queries.push_back(subresource->aspectMask);
            if (nv12)
            {
                require(subresource->aspectMask == VK_IMAGE_ASPECT_PLANE_0_BIT ||
                        subresource->aspectMask == VK_IMAGE_ASPECT_PLANE_1_BIT,
                        "NV12 layout query must select a format plane, not COLOR");
                *layout = subresource->aspectMask == VK_IMAGE_ASPECT_PLANE_0_BIT ? luma_layout : chroma_layout;
            }
            else
            {
                require(subresource->aspectMask == VK_IMAGE_ASPECT_COLOR_BIT,
                        "RGB layout query must select COLOR");
                *layout = color_layout;
            }
        }
        VkResult CreateImage(void *, const VkImageCreateInfo *, void *, VkImage *image)
        {
            ++create_calls;
            if (create_result == VK_SUCCESS)
                *image = next_image++;
            return create_result;
        }
        VkResult CreateImageView(void *, const VkImageViewCreateInfo *, void *, VkImageView *view)
        {
            ++view_calls;
            if (view_calls == fail_view_call)
                return -1;
            *view = 200 + view_calls;
            return VK_SUCCESS;
        }
        VkResult AllocateMemory(void *, const VkMemoryAllocateInfo *info, void *, VkDeviceMemory *memory)
        {
            ++calls;
            selected = info->memoryTypeIndex;
            require(selected < 4, "invalid memory type reached Vulkan allocation");
            if (allocate_result == VK_SUCCESS)
                *memory = 101;
            return allocate_result;
        }
        void DestroyImage(void *, VkImage image, void *)
        {
            require(image != VK_NULL_HANDLE, "destroyed a null image");
            ++destroy_calls;
            cleanup.push_back("image:" + std::to_string(image));
        }
        void DestroyImageView(void *, VkImageView view, void *)
        {
            require(view != VK_NULL_HANDLE, "destroyed a null view");
            cleanup.push_back("view:" + std::to_string(view));
        }
        void FreeMemory(void *, VkDeviceMemory memory, void *)
        {
            cleanup.push_back("memory:" + std::to_string(memory));
        }
        void UnmapMemory(void *, VkDeviceMemory memory)
        {
            cleanup.push_back("unmap:" + std::to_string(memory));
        }
        VkResult MapMemory(void *, VkDeviceMemory, VkDeviceSize offset, VkDeviceSize size,
                           uint32_t, void **data)
        {
            require(offset == 0 && size == VK_WHOLE_SIZE, "mapping must retain the allocation base");
            ++map_calls;
            *data = mapped_base;
            return VK_SUCCESS;
        }
    } vk;
} g_device;

struct
{
    template<class... T> void debugf(T...) {}
    template<class... T> void infof(T...) {}
} vk_log;
static uint32_t errors = 0;
void vk_errorf(VkResult, const char *) { ++errors; }

#include "memory_type.inc"

static bool allocate_image(bool mappable, uint32_t type_bits, VkImage &m_vkImage)
{
    struct { bool bMappable; } flags{mappable};
    struct { uint64_t size; uint32_t memoryTypeBits; } memRequirements{256, type_bits};
    VkMemoryPropertyFlags properties;
    VkResult res;
    VkDeviceMemory memoryHandle;
    bool m_bOwnsImage = true;
#include "image_allocation.inc"
    (void)m_bOwnsImage;
    return res == VK_SUCCESS;
}

static void check(std::initializer_list<VkMemoryPropertyFlags> types, bool mappable,
                  uint32_t type_bits, int32_t expected)
{
    g_device = {};
    errors = 0;
    for (auto flags : types)
        g_device.m_memoryProperties.memoryTypes[g_device.m_memoryProperties.memoryTypeCount++].propertyFlags = flags;
    VkImage image = 1;
    require(allocate_image(mappable, type_bits, image) == (expected >= 0), "unexpected allocation result");
    require(g_device.vk.calls == (expected >= 0 ? 1u : 0u), "allocation was not rejected before Vulkan");
    require(errors == (expected >= 0 ? 0u : 1u), "missing or unexpected allocation diagnostic");
    require(g_device.vk.destroy_calls == (expected >= 0 ? 0u : 1u), "rejected allocation leaked its image");
    require(image == (expected >= 0 ? 1u : VK_NULL_HANDLE), "rejected image handle was not cleared");
    if (expected >= 0)
        require(g_device.vk.selected == uint32_t(expected), "wrong memory type selected");
}

class CVulkanTexture
{
public:
#include "mapped_members.inc"
    ~CVulkanTexture();
    bool BInitFromSwapchain(VkImage image, uint32_t width, uint32_t height,
                            VkFormat format, VkResult *pResult);
    void *m_pBackendFb = nullptr;
    FixtureDmabuf m_dmabuf;
    bool ycbcr = false;
    bool isYcbcr() const { return ycbcr; }
    FixtureDmabuf dmabuf;
    bool create_image()
    {
        const VkImageCreateInfo imageInfo{};
        VkResult res;
#include "image_create.inc"
        return true;
    }
    bool allocate_memory()
    {
        struct { bool bMappable; } flags{true};
        struct { uint64_t size; uint32_t memoryTypeBits; } memRequirements{256, 1};
        VkMemoryPropertyFlags properties;
        VkResult res;
        VkDeviceMemory memoryHandle = VK_NULL_HANDLE;
        const void *pDMA = nullptr;
        struct { int fd = -1; } importMemoryInfo;
#include "image_allocation.inc"
#include "image_allocation_tail.inc"
        m_size = allocInfo.allocationSize;
        return true;
    }
    VkDeviceMemory reuse_memory(CVulkanTexture *pExistingImageToReuseMemory)
    {
        VkDeviceMemory memoryHandle = VK_NULL_HANDLE;
#include "image_reuse.inc"
        return memoryHandle;
    }
    void query_layout(bool mappable, bool exportable)
    {
        struct { bool bMappable, bExportable; } flags{mappable, exportable};
        const uint32_t tiling = VK_IMAGE_TILING_LINEAR;
        const VkImage m_vkImage = 1;
        g_device.vk.nv12 = ycbcr;
        g_device.vk.layout_queries.clear();
#include "mapped_layout.inc"
        if (flags.bExportable)
        {
#include "export_layout.inc"
        }
    }
    void set_layout(VkDeviceSize offset, uint32_t pitch)
    {
        g_device.vk.color_layout = {offset, pitch};
        query_layout(true, false);
    }
    bool map(CVulkanTexture *pExistingImageToReuseMemory = nullptr)
    {
        struct { bool bMappable; } flags{true};
        VkDeviceMemory memoryHandle = 1;
        VkResult res;
#include "mapped_mapping.inc"
        return true;
    }
};

#include "image_destructor.inc"
#include "image_swapchain.inc"

constexpr uint32_t k_uFramegenNetTexW = 3, k_uFramegenNetTexH = 2, k_uFramegenNetFloats = 5;
struct
{
    CVulkanTexture *netWeightsUpload = nullptr;
    CVulkanTexture *netProfileReadback = nullptr;
} g_framegenMotion;
struct FramegenHistory_t
{
    uint64_t ulNetProfileSeqNo = 0, ulNetProfileConsumedSeqNo = 0;
} g_framegenHistory;
std::vector<float> g_framegenNetReadbackWeights;

#include "mapped_upload.inc"
#include "mapped_readback.inc"

static const void *screenshot_write_data;
static size_t screenshot_write_bytes;
static std::vector<uint8_t> screenshot_output;

static size_t capture_fwrite(const void *data, size_t size, size_t count, void *)
{
    screenshot_write_data = data;
    screenshot_write_bytes = size * count;
    const auto *bytes = static_cast<const uint8_t *>(data);
    screenshot_output.assign(bytes, bytes + screenshot_write_bytes);
    return count;
}

static const uint8_t *screenshot_data(CVulkanTexture *pScreenshotTexture)
{
#include "screenshot_data.inc"
    return mappedData;
}

static void write_nv12_screenshot(CVulkanTexture *pScreenshotTexture)
{
    void *file = nullptr;
#include "screenshot_data.inc"
#define fwrite capture_fwrite
#include "screenshot_write.inc"
#undef fwrite
}

static void check_screenshot_allocation()
{
    // The logical allocation ends at byte 64. Spare fixture storage provides
    // a canary so a shifted whole-allocation read is observable without UB.
    std::array<uint8_t, 128> storage;
    storage.fill(0xe5);
    for (size_t i = 0; i < 64; ++i)
        storage[i] = uint8_t(i);
    g_device.vk.mapped_base = storage.data();
    CVulkanTexture screenshot;
    screenshot.set_layout(16, 24);
    screenshot.m_size = 64;
    require(screenshot.map(), "screenshot mapping fixture failed");
    require(screenshot_data(&screenshot) == storage.data() + 16,
            "single-plane screenshot lost its texel offset");
    CVulkanTexture nv12;
    nv12.ycbcr = true;
    g_device.vk.luma_layout = {16, 8};
    g_device.vk.chroma_layout = {48, 12};
    nv12.query_layout(true, true);
    nv12.m_size = 64;
    require(nv12.map(), "NV12 screenshot mapping fixture failed");
    require(screenshot_data(&nv12) == storage.data(), "NV12 screenshot used a shifted pointer");
    write_nv12_screenshot(&nv12);
    require(screenshot_write_data == storage.data(), "whole-allocation fwrite lost its allocation base");
    require(screenshot_write_bytes == 64, "whole-allocation fwrite changed its byte count");
    require(screenshot_output == std::vector<uint8_t>(storage.begin(), storage.begin() + 64),
            "NV12 screenshot copied shifted pixels or the trailing canary");
    require(std::all_of(storage.begin() + 64, storage.end(), [](uint8_t byte) { return byte == 0xe5; }),
            "screenshot fixture modified the trailing canary");
}

static void check_layout_queries()
{
    for (bool nv12 : {false, true})
    {
        for (bool mappable : {false, true})
        {
            CVulkanTexture texture;
            texture.ycbcr = nv12;
            g_device.vk.color_layout = {32, 20};
            g_device.vk.luma_layout = {16, 24};
            g_device.vk.chroma_layout = {96, 28};
            texture.query_layout(mappable, true);
            std::vector<uint32_t> expected;
            if (mappable)
            {
                if (nv12)
                    expected = {VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT};
                else
                    expected = {VK_IMAGE_ASPECT_COLOR_BIT};
                require(texture.m_mappedOffset == (nv12 ? 16u : 32u), "mapped offset lost queried layout");
                require(texture.rowPitch() == (nv12 ? 24u : 20u), "mapped pitch lost queried layout");
                if (nv12)
                {
                    require(texture.m_lumaOffset == 16 && texture.m_lumaPitch == 24,
                            "NV12 luma layout assignments changed");
                    require(texture.m_chromaOffset == 96 && texture.m_chromaPitch == 28,
                            "NV12 chroma layout assignments changed");
                }
            }
            else
                require(texture.m_mappedOffset == 0 && texture.rowPitch() == 0,
                        "nonmappable export changed mapped layout");
            expected.push_back(nv12 ? VK_IMAGE_ASPECT_PLANE_0_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
            require(g_device.vk.layout_queries == expected, "unexpected layout query sequence");
            require(texture.dmabuf.n_planes == 1 && texture.dmabuf.modifier == DRM_FORMAT_MOD_INVALID,
                    "nonmodifier export metadata changed");
            require(texture.dmabuf.offset[0] == 0 && texture.dmabuf.stride[0] == (nv12 ? 24u : 20u),
                    "nonmodifier export layout assignments changed");
        }
    }
}

static void check_mapped_pixels()
{
    CVulkanTexture unmapped;
    unmapped.set_layout(VkDeviceSize{1} << 33, 20);
    require(unmapped.m_mappedOffset == (VkDeviceSize{1} << 33), "subresource offset was truncated");
    require(unmapped.mappedPixels() == nullptr, "unmapped image produced a pixel pointer");

    std::array<uint8_t, 128> storage;
    storage.fill(0xcd);
    g_device.vk.mapped_base = storage.data();
    g_device.vk.map_calls = 0;
    CVulkanTexture upload;
    upload.set_layout(32, 20);
    require(upload.map(), "mapping fixture failed");
    require(upload.mappedData() == storage.data(), "allocation-base contract changed");
    require(upload.mappedPixels() == storage.data() + 32, "single-plane offset was omitted");
    const std::vector<float> weights{1, 2, 3, 4, 5};
    g_framegenMotion.netWeightsUpload = &upload;
    framegen_net_stage_weights(weights);
    auto expected = std::array<uint8_t, 128>{};
    expected.fill(0xcd);
    std::memcpy(expected.data() + 32, weights.data(), 12);
    std::memcpy(expected.data() + 52, weights.data() + 3, 8);
    require(storage == expected, "upload wrote the prefix, row padding, or wrong texels");

    // A reused allocation keeps its base mapping, but each image has its own
    // subresource offset. Reusing the original texel pointer would add twice.
    CVulkanTexture readback;
    readback.set_layout(64, 24);
    require(readback.map(&upload), "reused mapping fixture failed");
    require(g_device.vk.map_calls == 1, "reused allocation was mapped twice");
    require(readback.mappedData() == upload.mappedData(), "reuse lost the allocation base");
    require(readback.mappedPixels() == storage.data() + 64, "reuse applied the wrong offset");
    const std::vector<float> result{9, 8, 7, 6, 5};
    std::memcpy(storage.data() + 64, result.data(), 12);
    std::memcpy(storage.data() + 88, result.data() + 3, 8);
    g_framegenMotion.netProfileReadback = &readback;
    g_framegenHistory = {1, 0};
    g_framegenNetReadbackWeights.clear();
    g_device.completed = false;
    framegen_net_profile_consume();
    require(g_framegenNetReadbackWeights.empty(), "readback ignored GPU completion");
    g_device.completed = true;
    framegen_net_profile_consume();
    require(g_framegenNetReadbackWeights == result, "readback copied prefix or row padding");
    require(g_framegenHistory.ulNetProfileConsumedSeqNo == 1, "readback did not consume completed work");

    CVulkanTexture zero_offset;
    zero_offset.set_layout(0, 20);
    require(zero_offset.map(&upload), "zero-offset reuse failed");
    require(zero_offset.mappedPixels() == storage.data(), "zero-offset layout changed");
}

static void prepare_lifetime()
{
    g_device = {};
    errors = 0;
    g_device.m_memoryProperties.memoryTypeCount = 1;
    g_device.m_memoryProperties.memoryTypes[0].propertyFlags = coherent;
}

static void check_image_ownership()
{
    prepare_lifetime();
    {
        CVulkanTexture texture;
        require(!texture.m_bOwnsImage && texture.m_vkImage == VK_NULL_HANDLE,
                "default texture must not own an image");
    }
    require(g_device.vk.cleanup.empty(), "default texture destroyed resources");

    prepare_lifetime();
    g_device.vk.create_result = -1;
    {
        CVulkanTexture texture;
        require(!texture.create_image(), "failed image creation succeeded");
        require(!texture.m_bOwnsImage, "failed image creation took ownership");
    }
    require(g_device.vk.cleanup.empty(), "failed image creation destroyed resources");

    prepare_lifetime();
    {
        CVulkanTexture texture;
        require(texture.create_image() && texture.m_bOwnsImage,
                "successful image creation did not take ownership");
        // Model an early BInit return before memory allocation (e.g. an import
        // rejection). The real create prefix and real destructor execute.
    }
    require(g_device.vk.cleanup == std::vector<std::string>{"image:1"},
            "preallocation exit did not destroy its image exactly once");

    prepare_lifetime();
    g_device.vk.allocate_result = -1;
    {
        CVulkanTexture texture;
        require(texture.create_image(), "allocation-failure image creation failed");
        require(!texture.allocate_memory(), "failed memory allocation succeeded");
        require(texture.m_vkImageMemory == VK_NULL_HANDLE, "failed allocation retained memory");
    }
    require(g_device.vk.cleanup == std::vector<std::string>{"image:1"},
            "allocation failure leaked image or freed unowned memory");

    prepare_lifetime();
    g_device.m_memoryProperties.memoryTypeCount = 0;
    {
        CVulkanTexture texture;
        require(texture.create_image(), "no-type image creation failed");
        require(!texture.allocate_memory(), "missing memory type was accepted");
        require(texture.m_vkImage == VK_NULL_HANDLE && !texture.m_bOwnsImage,
                "explicit no-type cleanup did not clear image ownership");
        require(g_device.vk.cleanup == std::vector<std::string>{"image:1"},
                "no-type rejection did not destroy image immediately");
    }
    require(g_device.vk.cleanup == std::vector<std::string>{"image:1"},
            "no-type rejection destroyed image twice");

    prepare_lifetime();
    std::array<uint8_t, 256> storage{};
    g_device.vk.mapped_base = storage.data();
    {
        CVulkanTexture texture;
        require(texture.create_image() && texture.allocate_memory(), "owned image fixture failed");
        require(texture.map(), "owned image mapping failed");
        // Distinct handles represent all four independently-created views.
        texture.m_srgbView = 201;
        texture.m_linearView = 202;
        texture.m_lumaView = 203;
        texture.m_chromaView = 204;
    }
    require(g_device.vk.cleanup == std::vector<std::string>{
                "unmap:101", "view:201", "view:202", "view:203", "view:204", "image:1", "memory:101"},
            "owned texture must destroy all views before image before memory");

    prepare_lifetime();
    {
        CVulkanTexture owner;
        require(owner.create_image() && owner.allocate_memory(), "reuse owner fixture failed");
        {
            CVulkanTexture alias;
            require(alias.create_image(), "reuse alias creation failed");
            require(alias.reuse_memory(&owner) == owner.m_vkImageMemory,
                    "reuse lost the source allocation handle");
            require(alias.m_vkImageMemory == VK_NULL_HANDLE && alias.m_bOwnsImage,
                    "reuse confused image ownership with allocation ownership");
        }
        require(g_device.vk.cleanup == std::vector<std::string>{"image:2"},
                "reused allocation alias leaked its image or freed shared memory");
        require(owner.m_vkImageMemory == 101, "alias cleanup changed source allocation");
    }
    require(g_device.vk.cleanup == std::vector<std::string>{"image:2", "image:1", "memory:101"},
            "reused allocation owner cleanup changed");

    for (uint32_t fail_call : {0u, 1u, 2u})
    {
        prepare_lifetime();
        g_device.vk.fail_view_call = fail_call;
        VkResult result = -99;
        {
            CVulkanTexture borrowed;
            require(borrowed.BInitFromSwapchain(77, 8, 8, 1, &result) == (fail_call == 0),
                    "borrowed swapchain result ignored view failure");
            require((result == VK_SUCCESS) == (fail_call == 0), "swapchain result output changed");
            require(!borrowed.m_bOwnsImage && borrowed.m_vkImageMemory == VK_NULL_HANDLE,
                    "swapchain wrapper took image or allocation ownership");
        }
        std::vector<std::string> expected;
        if (fail_call != 1)
            expected.push_back("view:201");
        if (fail_call == 0)
            expected.push_back("view:202");
        require(g_device.vk.cleanup == expected,
                "swapchain cleanup leaked views or destroyed borrowed image/allocation");
    }
}

int main()
{
    // Prefer cached even when an uncached coherent type precedes it.
    check({coherent, coherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT}, true, 3, 1);
    check({coherent}, true, 1, 0);
    check({VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, coherent}, true, 3, 1);
    check({VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT}, true, 1, -1);
    check({VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT}, true, 1, -1);
    check({coherent, coherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT}, true, 1, 0);
    check({coherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT}, true, 0, -1);
    // Device-local images retain their required-property selection.
    check({coherent, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}, false, 3, 1);
    check({coherent}, false, 1, -1);
    std::cout << "Vulkan image memory selection: PASS\n";
    check_layout_queries();
    std::cout << "Vulkan RGB/NV12 mapped and export layout queries: PASS\n";
    check_mapped_pixels();
    std::cout << "Vulkan mapped image addressing: PASS\n";
    check_screenshot_allocation();
    std::cout << "Vulkan screenshot allocation addressing: PASS\n";
    check_image_ownership();
    std::cout << "Vulkan image ownership and destructor cleanup: PASS\n";
}
