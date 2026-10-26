#include <vulkan/vulkan.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "texture_state.inc"

static void require(bool condition, const char *message)
{
    if (!condition) { std::fprintf(stderr, "ASSERTION: %s\n", message); std::abort(); }
}

struct CVulkanTexture {
    VkImage image;
    uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED;
    explicit CVulkanTexture(uintptr_t id) : image(reinterpret_cast<VkImage>(id)) {}
    VkImage vkImage() const { return image; }
    bool concurrentSharing() const { return false; }
};
struct Backend { VkImageLayout GetPresentLayout() const { return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; } };
static Backend *GetBackend() { static Backend backend; return &backend; }
struct Recorder {
    std::vector<VkImageMemoryBarrier2> sync2;
    std::vector<VkImageMemoryBarrier> legacy;
    void CmdPipelineBarrier2(VkCommandBuffer, const VkDependencyInfo *info) {
        if (info->imageMemoryBarrierCount) sync2.insert(sync2.end(), info->pImageMemoryBarriers, info->pImageMemoryBarriers+info->imageMemoryBarrierCount);
    }
    void CmdPipelineBarrier(VkCommandBuffer, VkPipelineStageFlags source, VkPipelineStageFlags destination,
                            VkDependencyFlags, uint32_t, const VkMemoryBarrier *, uint32_t,
                            const VkBufferMemoryBarrier *, uint32_t count, const VkImageMemoryBarrier *images) {
        require(source==VK_PIPELINE_STAGE_ALL_COMMANDS_BIT && destination==source, "legacy publication execution scope");
        if (count) legacy.insert(legacy.end(), images, images+count);
    }
    void clear() { sync2.clear(); legacy.clear(); }
};
struct CVulkanDevice { Recorder vk; bool supportsModifiers() const { return false; } };
struct CVulkanCmdBuffer {
#include "image_types.inc"
    VkCommandBuffer m_cmdBuffer = VK_NULL_HANDLE;
    CVulkanDevice *m_device;
    uint32_t m_queueFamily = 0; // General queue: compute, fragment and transfer pairs are legal.
    std::vector<TrackedTextureState> m_textureState;
    std::vector<VkImageMemoryBarrier> m_imageBarriers;
    std::vector<ImageUse> m_imageUses;
    std::vector<VkImageMemoryBarrier2> m_imageBarriers2;
    explicit CVulkanCmdBuffer(CVulkanDevice &device) : m_device(&device) { m_textureState.reserve(16); }
    std::pair<TextureState *, bool> trackTexture(CVulkanTexture *);
    TextureState *findTextureState(CVulkanTexture *);
    void discardImage(CVulkanTexture *);
    void markDirty(CVulkanTexture *);
    void addImageUse(CVulkanTexture *, VkPipelineStageFlags2, VkAccessFlags2, bool);
    void emitSync2Barriers();
    void insertBarrier(bool flush=false);
    void emitExternalAcquireBarriers() {
        // External ownership/waits are outside this internal-visibility contract.
        for (const auto &entry : m_textureState)
            require(!entry.state.needsImport, "external acquire stub requires prior legacy import publication");
    }
};
#include "owner.inc"

constexpr auto C = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
constexpr auto T = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
constexpr auto F = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
constexpr auto R = VK_ACCESS_2_SHADER_READ_BIT;
constexpr auto W = VK_ACCESS_2_SHADER_WRITE_BIT;
constexpr auto TR = VK_ACCESS_2_TRANSFER_READ_BIT;
constexpr auto TW = VK_ACCESS_2_TRANSFER_WRITE_BIT;
constexpr auto SR = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
constexpr auto SAMPLED = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

static void use(CVulkanCmdBuffer &cmd, CVulkanTexture &image, VkPipelineStageFlags2 stage, VkAccessFlags2 access, bool write=false)
{
    cmd.trackTexture(&image);
    cmd.m_imageUses.clear(); // Per-operation batch input, not a copied hazard tracker.
    cmd.addImageUse(&image, stage, access, write);
    cmd.emitSync2Barriers();
    if (write) cmd.markDirty(&image); // Dispatch callers mark writes after barriers.
}
static bool access_covers(VkAccessFlags2 mask, VkAccessFlags2 access)
{
    if (mask & VK_ACCESS_2_SHADER_READ_BIT) mask |= SR | SAMPLED;
    if (mask & VK_ACCESS_2_MEMORY_READ_BIT) mask |= R | SR | SAMPLED | TR;
    return (mask & access)==access;
}
static bool visible(const Recorder &record, VkImage image, VkPipelineStageFlags2 stage, VkAccessFlags2 access)
{
    for (const auto &barrier : record.sync2)
        if (barrier.image==image && ((barrier.dstStageMask & stage)==stage || (barrier.dstStageMask & VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT))
            && access_covers(barrier.dstAccessMask, access)) return true;
    for (const auto &barrier : record.legacy)
        if (barrier.image==image && access_covers(barrier.dstAccessMask, access)) return true;
    return false;
}
static void full_color(const VkImageSubresourceRange &range)
{
    require(range.aspectMask==VK_IMAGE_ASPECT_COLOR_BIT && range.baseMipLevel==0 && range.levelCount==1
        && range.baseArrayLayer==0 && range.layerCount==1, "full color subresource");
}
static void reset_scope(const TextureState &state)
{
    require(state.visibleReadStage==0 && state.visibleReadAccess==0, "legacy publication resets local read visibility coverage");
}

static void layout_read_other()
{
    CVulkanDevice device; CVulkanCmdBuffer cmd(device); CVulkanTexture image(1);
    cmd.discardImage(&image); use(cmd,image,C,R);
    require(device.vk.sync2.size()==1 && device.vk.sync2[0].oldLayout==VK_IMAGE_LAYOUT_UNDEFINED, "initial layout transition");
    full_color(device.vk.sync2[0].subresourceRange);
    use(cmd,image,T,TR);
    require(visible(device.vk,image.image,T,TR), "layout-write visibility reaches transfer reader");
}
static void compute_raw_transfer()
{
    CVulkanDevice device; CVulkanCmdBuffer cmd(device); CVulkanTexture image(1);
    use(cmd,image,C,W,true); device.vk.clear(); use(cmd,image,C,R); use(cmd,image,T,TR);
    require(visible(device.vk,image.image,T,TR), "compute write reaches later transfer reader");
}
static void transfer_raw_compute()
{
    CVulkanDevice device; CVulkanCmdBuffer cmd(device); CVulkanTexture image(1);
    use(cmd,image,T,TW,true); device.vk.clear(); use(cmd,image,T,TR); use(cmd,image,C,R);
    require(visible(device.vk,image.image,C,R), "transfer write reaches later compute reader");
}
static void covered_rr()
{
    CVulkanDevice device; CVulkanCmdBuffer cmd(device); CVulkanTexture image(1);
    use(cmd,image,C,W,true); use(cmd,image,C,R); const auto count=device.vk.sync2.size();
    use(cmd,image,C,R); require(device.vk.sync2.size()==count, "covered read-read needs no barrier");
}
static void published_clean_rr()
{
    CVulkanDevice producer; CVulkanCmdBuffer first(producer); CVulkanTexture image(1);
    use(first,image,C,W,true); first.insertBarrier(true);
    require(visible(producer.vk,image.image,T,TR), "prior command buffer publishes broad visibility");
    CVulkanDevice consumer; CVulkanCmdBuffer next(consumer);
    use(next,image,C,R); use(next,image,T,TR);
    require(consumer.vk.sync2.empty(), "clean externally published read-read does not add internal barriers");
}
static void false_cross_product()
{
    CVulkanDevice device; CVulkanCmdBuffer cmd(device); CVulkanTexture image(1);
    use(cmd,image,C,W,true); device.vk.clear(); use(cmd,image,C,SR); use(cmd,image,F,SAMPLED);
    require(visible(device.vk,image.image,F,SAMPLED), "fragment reader scope was actually published");
    const bool already=visible(device.vk,image.image,C,SAMPLED); const auto count=device.vk.sync2.size();
    use(cmd,image,C,SAMPLED);
    require(visible(device.vk,image.image,C,SAMPLED), "metadata unions cannot invent unpublished stage-access cross products");
    if (already) require(device.vk.sync2.size()==count, "actual union publication covers subsequent cross-product reader");
}
static void all_reader_war()
{
    CVulkanDevice device; CVulkanCmdBuffer cmd(device); CVulkanTexture image(1);
    use(cmd,image,C,R); use(cmd,image,T,TR); device.vk.clear(); use(cmd,image,C,W,true);
    require(device.vk.sync2.size()==1, "read-to-write dependency");
    const auto &barrier=device.vk.sync2[0];
    require((barrier.srcStageMask & (C|T))==(C|T) && barrier.srcAccessMask==0, "WAR waits on every reader with execution scope");
    require(barrier.srcQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED && barrier.dstQueueFamilyIndex==VK_QUEUE_FAMILY_IGNORED, "internal sync2 has no ownership transfer");
}
static void dirty_flush()
{
    CVulkanDevice device; CVulkanCmdBuffer cmd(device); CVulkanTexture image(1);
    use(cmd,image,C,W,true); use(cmd,image,C,R);
    require(cmd.findTextureState(&image)->dirty, "dirty remains latched through narrower RAW");
    device.vk.legacy.clear(); cmd.insertBarrier(true);
    require(!device.vk.legacy.empty() && visible(device.vk,image.image,T,TR), "dirty end flush publishes later transfer reads");
    require(!cmd.findTextureState(&image)->dirty, "flush clears dirty latch");
}
static void layout_end_flush()
{
    CVulkanDevice device; CVulkanCmdBuffer cmd(device); CVulkanTexture image(1);
    cmd.discardImage(&image); use(cmd,image,C,R); device.vk.legacy.clear(); cmd.insertBarrier(true);
    require(!device.vk.legacy.empty() && access_covers(device.vk.legacy.back().dstAccessMask,TR), "layout-only end flush publishes visibility");
    CVulkanDevice later; CVulkanCmdBuffer next(later); use(next,image,T,TR);
    require(later.vk.sync2.empty(), "published layout allows clean first read in new command buffer");
}
static void discard_flush()
{
    CVulkanDevice device; CVulkanCmdBuffer cmd(device); CVulkanTexture image(1);
    cmd.discardImage(&image); cmd.insertBarrier(true);
    require(device.vk.legacy.size()==1 && device.vk.legacy[0].oldLayout==VK_IMAGE_LAYOUT_UNDEFINED, "direct discard flush transitions layout");
    require(access_covers(device.vk.legacy[0].dstAccessMask,R|TR), "direct discard flush publishes implicit layout write");
    full_color(device.vk.legacy[0].subresourceRange);
}
static void reset_legacy_import()
{
    CVulkanDevice device; CVulkanCmdBuffer cmd(device); CVulkanTexture image(1);
    use(cmd,image,C,W,true); use(cmd,image,C,R); use(cmd,image,T,TR); cmd.insertBarrier(false);
    reset_scope(*cmd.findTextureState(&image));
    cmd.findTextureState(&image)->needsImport=true; cmd.insertBarrier(false);
    require(!cmd.findTextureState(&image)->needsImport, "legacy import input reset");
    reset_scope(*cmd.findTextureState(&image));
    TextureState fresh; reset_scope(fresh);
    require(fresh.lastAccess==0 && !fresh.bLastWasWrite && !fresh.dirty && !fresh.discarded, "new tracker constructor reset");
}
static void independent_images()
{
    CVulkanDevice device; CVulkanCmdBuffer cmd(device); CVulkanTexture a(1),b(2),c(3);
    cmd.discardImage(&a); use(cmd,a,C,R); device.vk.clear(); use(cmd,b,T,TR);
    require(device.vk.sync2.empty() && !cmd.findTextureState(&b)->dirty, "per-image layout/write state stays independent");
    cmd.discardImage(&c); use(cmd,c,C,R);
    require(device.vk.sync2.size()==1 && device.vk.sync2[0].image==c.image, "discard only transitions selected image");
}
struct Case { const char *name; void (*run)(); };
static const Case cases[] = {
    {"layout-read-other",layout_read_other},{"compute-raw-transfer",compute_raw_transfer},
    {"transfer-raw-compute",transfer_raw_compute},{"covered-rr",covered_rr},
    {"published-clean-rr",published_clean_rr},{"false-cross-product",false_cross_product},
    {"all-reader-war",all_reader_war},{"dirty-flush",dirty_flush},
    {"layout-end-flush",layout_end_flush},{"discard-flush",discard_flush},
    {"reset-legacy-import",reset_legacy_import},{"independent-images",independent_images},
};
int main()
{
    for (const auto &test : cases) {
        test.run();
        std::printf("%s PASS\n", test.name);
    }
    return 0;
}
