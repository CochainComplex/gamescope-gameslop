#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

static void require(bool condition, const char *message)
{
    if (!condition) { std::fprintf(stderr, "ASSERTION: %s\n", message); std::abort(); }
}

static std::vector<uint32_t> events;
struct CVulkanCmdBuffer {
    uint32_t id;
    explicit CVulkanCmdBuffer(uint32_t value) : id(value) {}
    void reset() { events.push_back(100+id); }
};
struct Timeline { VkSemaphore pVkSemaphore = reinterpret_cast<VkSemaphore>(2); };
struct Query { VkResult result = VK_SUCCESS; uint64_t begin = 10, end = 30; };
struct Recorder {
    uint64_t completed = 0;
    VkSemaphore expectedTimeline = reinterpret_cast<VkSemaphore>(2);
    std::array<Query, 8> queries;
    VkResult GetSemaphoreCounterValue(VkDevice, VkSemaphore timeline, uint64_t *value) {
        require(timeline == expectedTimeline, "owning completion timeline");
        events.push_back(1);
        *value = completed;
        return VK_SUCCESS;
    }
    VkResult GetQueryPoolResults(VkDevice, VkQueryPool pool, uint32_t first, uint32_t count,
                                 size_t size, void *data, VkDeviceSize stride, VkQueryResultFlags flags) {
        require(pool == reinterpret_cast<VkQueryPool>(3), "framegen query pool");
        require(count == 2 && first % 2 == 0 && first / 2 < queries.size(), "timestamp pair");
        require(size == 16 && stride == 8 && flags == VK_QUERY_RESULT_64_BIT, "nonblocking 64-bit query");
        events.push_back(1000+first/2);
        const Query &query = queries[first/2];
        auto *ticks = static_cast<uint64_t *>(data);
        ticks[0] = query.begin;
        ticks[1] = query.end;
        return query.result;
    }
};

#include "cache.inc"
#define vk_check(expression) require((expression) == VK_SUCCESS, "fatal Vulkan operation")
struct CVulkanDevice {
    static constexpr uint32_t kFramegenLadderSlots = 8, kFramegenGeneratedCountSlots = 4;
#include "association.inc"
    Recorder vk;
    bool m_bHasFramegenQueue = true, split = false;
    Timeline timeline;
    Timeline *m_framegenTimeline = &timeline;
    VkSemaphore m_scratchTimelineSemaphore = reinterpret_cast<VkSemaphore>(1);
    VkQueryPool m_framegenQueryPool = reinterpret_cast<VkQueryPool>(3);
    std::atomic<uint64_t> m_framegenCompletedSeqNo{0}, m_submissionCompletedSeqNo{0};
    std::vector<std::pair<uint64_t, std::unique_ptr<CVulkanCmdBuffer>>> m_pendingFramegenCmdBufs;
    std::vector<std::unique_ptr<CVulkanCmdBuffer>> m_unusedFramegenCmdBufs, m_unusedCmdBufs;
    std::vector<std::pair<uint64_t, FramegenQueryAssoc_t>> m_framegenQuerySlotBySeqNo;
    uint32_t m_uFramegenTimestampValidBits = 64;
    double m_flFramegenTimestampPeriodNs = 1.5;
    uint64_t m_aFramegenRungCostNs[kFramegenLadderSlots][kFramegenGeneratedCountSlots] = {};
    uint32_t m_aFramegenRungSamples[kFramegenLadderSlots][kFramegenGeneratedCountSlots] = {};
    uint64_t m_ulFramegenLastRawGpuTimeNs = 0, m_ulFramegenTrainingCostNs = 0, m_ulFramegenTrainingSampleSeqNo = 0;
    VkDevice device() const { return VK_NULL_HANDLE; }
    bool framegenFamilySplit() const { events.push_back(2); return split; }
    void framegenGarbageCollect();
    void framegenResetTrainingCost();
    void framegenResetRungCosts();
#include "helper.inc"
    void pending(uint64_t sequence, uint32_t id) {
        m_pendingFramegenCmdBufs.emplace_back(sequence, std::make_unique<CVulkanCmdBuffer>(id));
    }
    void sample(uint64_t sequence, uint32_t slot, uint32_t rung=1, uint32_t count=2, bool training=false) {
        m_framegenQuerySlotBySeqNo.emplace_back(sequence, FramegenQueryAssoc_t{slot, rung, count, training});
    }
};
#include "owner.inc"

static void retirement(bool dedicated, bool split)
{
    CVulkanDevice d;
    d.m_bHasFramegenQueue = dedicated;
    d.split = split;
    d.vk.expectedTimeline = dedicated ? d.timeline.pVkSemaphore : d.m_scratchTimelineSemaphore;
    events.clear();
    d.framegenGarbageCollect();
    require(events.empty(), "empty collector never observes Vulkan");
    d.pending(4, 1); d.pending(7, 2); d.pending(9, 3);
    d.sample(4, 0); d.sample(7, 1); d.sample(9, 2);
    d.vk.completed = 7;
    d.framegenGarbageCollect();
    const std::vector<uint32_t> expected = dedicated
        ? std::vector<uint32_t>{1,101,2,102,2,1000,1001}
        : std::vector<uint32_t>{1,1000,1001};
    require(events == expected, "poll then completed-prefix recycle then query order");
    require(d.m_pendingFramegenCmdBufs.size() == (dedicated ? 1u : 3u), "shared buffers remain owned elsewhere");
    require(d.m_unusedFramegenCmdBufs.size() == (dedicated && split ? 2u : 0u), "split-family pool");
    require(d.m_unusedCmdBufs.size() == (dedicated && !split ? 2u : 0u), "same-family pool");
    require(d.m_framegenQuerySlotBySeqNo.size() == 1 && d.m_framegenQuerySlotBySeqNo[0].first == 9,
            "exact completed boundary, no pending query read");
    require(d.m_framegenCompletedSeqNo == (dedicated ? 7u : 0u), "framegen cache provenance");
    require(d.m_submissionCompletedSeqNo == (dedicated ? 0u : 7u), "composite cache provenance");
    require(d.m_aFramegenRungCostNs[1][2] == 30 && d.m_aFramegenRungSamples[1][2] == 2, "consecutive shape samples");
    events.clear(); d.framegenGarbageCollect();
    require(events == std::vector<uint32_t>{1}, "same completion observation cannot consume a sample twice");
}

static void rejected_queries()
{
    CVulkanDevice d;
    d.vk.completed = 10;
    d.vk.queries[0].result = VK_NOT_READY;
    d.vk.queries[1].result = VK_ERROR_DEVICE_LOST;
    d.vk.queries[2].end = 10;
    for (uint32_t slot = 0; slot < 3; ++slot) d.sample(slot+1, slot, 1, 2, true);
    d.sample(4, 3, 8, 2, true); d.sample(5, 4, 1, 4, true);
    events.clear(); d.framegenGarbageCollect();
    require(events == std::vector<uint32_t>({1,1000,1001,1002,1003,1004}), "rejected queries still read in order");
    require(d.m_framegenQuerySlotBySeqNo.empty(), "failed, empty and invalid results are consumed, never retried");
    require(d.m_ulFramegenLastRawGpuTimeNs == 0 && d.m_ulFramegenTrainingCostNs == 0
            && d.m_aFramegenRungSamples[1][2] == 0, "rejected samples publish nothing");
    events.clear(); d.framegenGarbageCollect();
    require(events.empty(), "consumed errors cause no repeated polling");
}

static void accounting()
{
    CVulkanDevice d;
    d.vk.completed = 3;
    d.vk.queries[0] = {VK_SUCCESS, (1ull<<36)-5, 4};
    d.vk.queries[1] = {VK_SUCCESS, 10, 100};
    d.vk.queries[2] = {VK_SUCCESS, 10, 13};
    d.m_uFramegenTimestampValidBits = 36;
    d.m_aFramegenRungCostNs[1][2] = 20;
    d.m_aFramegenRungSamples[1][2] = UINT32_MAX;
    d.sample(1, 0, 1, 2, true); // 9 ticks -> 13ns, EMA floors to 19.
    d.sample(2, 1, 0, 0, true); // Training-only 135ns must not seed serving costs.
    d.sample(3, 2, 2, 1, true); // 4ns, latest sequence but not the maximum cost.
    d.framegenGarbageCollect();
    require(d.m_ulFramegenTrainingCostNs == 135 && d.m_ulFramegenTrainingSampleSeqNo == 3, "training maximum and latest admitted sequence");
    require(d.m_aFramegenRungCostNs[1][2] == 19 && d.m_aFramegenRungSamples[1][2] == UINT32_MAX, "exact EMA and saturating samples");
    require(d.m_aFramegenRungCostNs[0][0] == 0 && d.m_aFramegenRungSamples[0][0] == 0, "training-only never serves");
    require(d.m_aFramegenRungCostNs[2][1] == 4 && d.m_ulFramegenLastRawGpuTimeNs == 4, "exact shape and conversion truncation");
    d.m_uFramegenTimestampValidBits = 64;
    d.vk.queries[0] = {VK_SUCCESS, UINT64_MAX-2, 2};
    d.sample(4, 0, 7, 3); d.vk.completed = 4; d.framegenGarbageCollect();
    require(d.m_aFramegenRungCostNs[7][3] == 7 && d.m_ulFramegenTrainingCostNs == 135, "64-bit wrap and nontraining sample");
}

static void resets()
{
    CVulkanDevice d;
    d.sample(1, 0, 1, 2, true);
    d.m_ulFramegenTrainingCostNs = 99; d.m_ulFramegenTrainingSampleSeqNo = 9;
    d.framegenResetTrainingCost();
    require(d.m_framegenQuerySlotBySeqNo.size() == 1 && !d.m_framegenQuerySlotBySeqNo[0].second.bOnlineTraining,
            "training reset retains serving association but removes training eligibility");
    d.vk.completed = 1; d.framegenGarbageCollect();
    require(d.m_aFramegenRungCostNs[1][2] == 30 && d.m_ulFramegenTrainingCostNs == 0
            && d.m_ulFramegenTrainingSampleSeqNo == 0, "old training cannot reseed after reset");
    d.pending(2, 1); d.sample(2, 1, 1, 2, true);
    d.framegenResetRungCosts();
    require(d.m_framegenQuerySlotBySeqNo.empty() && d.m_pendingFramegenCmdBufs.size() == 1, "scene reset drops observations, retains GPU ownership");
    events.clear(); d.vk.completed = 2; d.framegenGarbageCollect();
    require(events == std::vector<uint32_t>({1,101,2}), "scene-reset work retires without stale query publication");
    require(d.m_ulFramegenLastRawGpuTimeNs == 0 && d.m_ulFramegenTrainingCostNs == 0, "reset raw costs");
    for (uint32_t r = 0; r < d.kFramegenLadderSlots; ++r)
        for (uint32_t c = 0; c < d.kFramegenGeneratedCountSlots; ++c)
            require(d.m_aFramegenRungCostNs[r][c] == 0 && d.m_aFramegenRungSamples[r][c] == 0, "all scene costs reset");
}

int main()
{
    retirement(false, false); // Shared queue: command buffers retire elsewhere.
    retirement(true, false); // Dedicated queue in the composite family.
    retirement(true, true); // Dedicated queue in a separate family.
    rejected_queries(); accounting(); resets();
    std::puts("completed-timing retirement, rejection, accounting and reset contracts passed");
    return 0;
}
