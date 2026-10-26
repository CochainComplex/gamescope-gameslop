// Policy plus production-extracted owner transactions. No Vulkan/GPU execution.
#include "framegen/learning.hpp"
#include "framegen/types.hpp"
#include <cassert>
#include <cstdint>
#include <limits>

static void policy_contracts()
{
	using namespace gamescope::framegen;
	NetTrainingCadence cadence;
	assert(!cadence.consider(0, 3) && !cadence.consider(1, 0));
	for (uint64_t pair = 1; pair <= 10; ++pair)
	{
		assert(cadence.due(pair, 3) == (pair % 3 == 1));
		assert(cadence.consider(pair, 3) == (pair % 3 == 1));
		const auto counter = cadence.counter;
		for (int poll = 0; poll < 5; ++poll)
			assert(!cadence.due(pair, 3) && !cadence.consider(pair, 3));
		assert(cadence.counter == counter);
	}
	NetTrainingCadence every;
	assert(every.consider(1, 1) && every.consider(2, 1));
	assert(!background_learning_due(0, 1, 0, 0));
	assert(!background_learning_due(0, 1, 3, 3));
	assert(!background_learning_due(2, 1, 3, 2));
	assert(!background_learning_due(10, 250'000'009, 3, 2));
	assert(background_learning_due(10, 250'000'010, 3, 2));
	assert(!background_learning_fits(0, 10'000'000, 0));
	assert(!background_learning_fits(10, 10, 1));
	assert(!background_learning_fits(10, 9, 1));
	assert(!background_learning_fits(0, 3'000'000, 2'000'000));
	assert(background_learning_fits(0, 3'000'001, 2'000'000));
	assert(!background_learning_fits(0, 4'000'000, 2'000'000, 1'000'000));
	assert(background_learning_fits(0, 4'000'001, 2'000'000, 1'000'000));
	assert(!background_learning_fits(0, UINT64_MAX, UINT64_MAX-1, UINT64_MAX));
}

#ifdef FRAMEGEN_PRODUCTION_FIXTURE
#include <vulkan/vulkan.h>
#include "framegen/adaptation.hpp"
#include "framegen/atomic_file.hpp"
#include "framegen/metrics.hpp"
#include "framegen/net_profile.hpp"
#include "framegen/numeric.hpp"
#include "framegen/scheduling.hpp"
#include <array>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace gamescope { template<class T> using Rc = T*; }
using NetLearningSkip = gamescope::framegen::NetLearningSkip;
struct FramegenMotionNetOptPush_t {};
struct FramegenMotionNetTrainPush_t {};
constexpr int SHADER_TYPE_FRAMEGEN_MOTION_NET_OPT = 1, SHADER_TYPE_FRAMEGEN_MOTION_NET_TRAIN = 2;
#include "learning_constants.inc"
struct CVulkanTexture {
	VkImage image = VK_NULL_HANDLE;
	std::vector<uint8_t> bytes;
	uint32_t pitch = 0;
	uint32_t width() const { return 800; }
	uint32_t height() const { return 600; }
	uint8_t *mappedPixels() { return bytes.empty() ? nullptr : bytes.data()+37; }
	uint32_t rowPitch() const { return pitch; }
	VkImage vkImage() const { return image; }
};
struct CVulkanCmdBuffer {
	unsigned dispatches = 0, copies = 0, barriers = 0;
	int pipeline = 0;
	std::vector<std::pair<int, CVulkanTexture*>> slot_zero;
	void markFramegen() {}
	void bindPipeline(int next) { pipeline = next; }
	void bindTarget(CVulkanTexture*) {}
	void bindTarget2(CVulkanTexture*) {}
	template<class T, class... A> void pushConstants(A...) {}
	void dispatch(uint32_t, uint32_t) { ++dispatches; }
	void copyImage(CVulkanTexture*, CVulkanTexture*) { ++copies; }
	void hostReadBarrier() { ++barriers; }
	VkCommandBuffer rawBuffer() const { return reinterpret_cast<VkCommandBuffer>(uintptr_t{44}); }
};
struct FramegenMotionResources_t {
	bool bNetWeightsUploaded = false, bNetStatePending = false, bNetActive = false, bUncheckedFieldValid = false;
	bool bNetUploadInitialized = false;
	uint32_t uNetTrainStep = 0, width = 0, height = 0;
	uint64_t uMotionFieldFrameId = 0, uMotionFieldIntervalNs = 0;
	uint64_t uMotionHistoryFrameId = 0, uMotionHistoryIntervalNs = 0, uLumaReservoirFrameId[2]{};
	gamescope::framegen::NetTrainingCadence netTrainingCadence;
	struct LearningCost_t {
		uint64_t warmedSeqNo = 0, pendingRecordSeqNo = 0, pendingRecordNs = 0, recordCostNs = 0, lastAttemptNs = 0;
	} learning;
	CVulkanTexture *netState = nullptr, *netGradSlices = nullptr, *netWeightsPrior = nullptr, *statsAccum = nullptr;
	CVulkanTexture *netWeightsGpu = nullptr, *netWeightsUpload = nullptr, *netProfileReadback = nullptr, *statsReadback = nullptr;
	CVulkanTexture *mvFieldRevChk = nullptr, *mvField = nullptr, *mvFieldHistory = nullptr;
	CVulkanTexture *lumaPrev = nullptr, *lumaCur = nullptr, *lumaReservoir[2]{};
};
static FramegenMotionResources_t g_framegenMotion;
struct FramegenHistory_t {
	uint64_t lastGeneratedSeqNo = 0, lastLearningSeqNo = 0, lastFramegenWorkSeqNo = 0;
	uint64_t lastCompositeSeqNo = 0, lastBaseIngestSeqNo = 0, currentFrameId = 0;
	uint64_t ulNetProfileSeqNo = 0, ulNetProfileConsumedSeqNo = 0, ulAdaptStatsSeqNo = 0, ulAdaptConsumedSeqNo = 0;
	CVulkanTexture *previousReal = nullptr, *currentReal = nullptr, *genReadA = nullptr, *genReadB = nullptr, *genReadReference = nullptr;
	uint64_t genReadSeqNo = 0;
	bool valid = false, bBaseLayer = false, bHudFreePrevious = false, bHudFreeCurrent = false;
	uint32_t nDegradeSteps = 0, drmFormat = 0;
	struct Pending { uint64_t ulWakeDeadlineNs = 0; };
	std::vector<Pending> pending;
	struct { uint64_t intervalNs = 0; } cadence;
	struct { int epoch = 0; } bidir;
	int causalAnchor = 0, causalDeadlineMisses = 0, bidirCutEpisode = 0, adaptation = 0;
	gamescope::framegen::RecoveryState_t recovery;
	uint32_t nStableFrames = 0, nLastGeneratedSlot = 0, nLastGenerationGapVblanks = 0, nDegradeHold = 0;
	uint64_t previousPresentTimeNs = 0, currentPresentTimeNs = 0, ulCurrentCadenceTimeNs = 0, ulCurrentRealVblankNs = 0;
	uint64_t ulDeadlineGridEpoch = 0, ulDeadlineGridIntervalNs = 0, ulDeadlineDisplayChainGeneration = 0, ulLastPlannedTargetNs = 0;
	uint64_t ulBidirConsumedFrameId = 0, ulVrrAwaitingRealFrameId = 0, ulVrrAwaitingCompositeSeqNo = 0, ulVrrAwaitingCadenceNs = 0;
	uint64_t ulVrrPanelIntervalNs = 0, ulVrrMidTargetNs = 0, ulVrrMidWakeNs = 0, ulNetRecordSeqNo = 0, ulNetRecordConsumedSeqNo = 0;
	bool bCadenceUsesSourceTime = false, bDeadlineProvenanceInitialized = false, bDeadlineSourceTimestampsReliable = false, bCausalDeadlineMissed = false;
	CVulkanTexture *bidirLastOutput = nullptr, *pLastBaseTexture = nullptr;
	uint64_t ulLastBaseCommitID = 0;
};
static FramegenHistory_t g_framegenHistory;
struct { uint64_t trainedSteps = 5, publishedSteps = 0, rejectedNetReadbacks = 0, droppedNetReadbacks = 0; } g_framegenMetricsPendingEvents;
static gamescope::framegen::NetLearningPair g_framegenNetLearningPair;
static uint64_t g_ulFramegenNetProgress = 10, g_ulFramegenNetLiveProgress = 10;
static bool g_bFramegenNetStagingStale = false, g_bFramegenDebug = false, g_bFramegenNetKeyActive = true, g_bFramegenNetProfileLoaded = false;
static uint32_t g_uFramegenNetActiveKey = 0xf0, g_uFramegenRecordCount = 0;
static std::vector<float> g_framegenNetLiveWeights, g_framegenNetReadbackWeights, g_framegenNetActivePrior;
static std::atomic<uint64_t> g_ulFramegenNetSavedProgress{0}, g_ulFramegenNetSavedRevision{0};
static std::atomic<bool> g_bFramegenNetWriteInFlight{false};
static std::thread g_framegenNetWriteThread;
static std::string profile_path;
struct FramegenNetStore_t {
	std::vector<gamescope::framegen::NetProfileSet> sets;
	std::vector<float> blob, neutral;
	std::vector<uint32_t> profileKeys;
	uint64_t ulRevision = 0;
} store;
static FramegenNetStore_t &framegen_net_store() { return store; }
struct Log { template<class... A> void infof(const char*, A...) {} template<class... A> void errorf(const char*, A...) {} } vk_log;
static std::vector<std::string> events;
struct Device {
	struct Barrier {
		VkCommandBuffer command;
		VkPipelineStageFlags source, destination;
		VkDependencyFlags dependencies;
		VkImageMemoryBarrier image;
	};
	struct Calls {
		std::vector<Barrier> barriers;
		void CmdPipelineBarrier(VkCommandBuffer command, VkPipelineStageFlags source,
			VkPipelineStageFlags destination, VkDependencyFlags dependencies,
			uint32_t memoryCount, const VkMemoryBarrier *memory, uint32_t bufferCount,
			const VkBufferMemoryBarrier *buffers, uint32_t imageCount, const VkImageMemoryBarrier *images) {
			assert(memoryCount == 0 && memory == nullptr && bufferCount == 0 && buffers == nullptr);
			assert(imageCount == 1 && images);
			barriers.push_back({command, source, destination, dependencies, images[0]});
		}
	} vk;
	uint64_t completed = 100, baseCompleted = 100, nextSeq = 101, trainingCost = 1'000'000, trainingSample = 0;
	unsigned commands = 0, submitted = 0;
	bool split = false, commandOK = true;
	int query = 1;
	bool hasFramegenQueue() const { return true; }
	bool framegenFamilySplit() const { return split; }
	bool hasCompletedFramegen(uint64_t seq) const { return seq <= completed; }
	bool hasCompleted(uint64_t seq) const { return seq <= baseCompleted; }
	void waitFramegen(uint64_t seq) {
		assert(g_framegenHistory.genReadA && g_framegenHistory.genReadB);
		assert(g_framegenHistory.genReadA != g_framegenHistory.genReadB);
		assert(g_framegenMotion.netState && g_framegenMotion.netProfileReadback);
		events.emplace_back("wait-learning"); completed = seq;
	}
	void wait(uint64_t seq) { events.emplace_back("wait-base"); baseCompleted = seq; }
	void framegenGarbageCollect() {}
	void framegenResetTrainingCost() { trainingCost = 0; }
	void framegenResetRungCosts() { events.emplace_back("reset-costs"); framegenResetTrainingCost(); }
	uint64_t framegenTrainingCostNs() const { return trainingCost; }
	uint64_t framegenTrainingSampleSeqNo() const { return trainingSample; }
	std::unique_ptr<CVulkanCmdBuffer> commandBuffer(bool) { ++commands; return commandOK ? std::make_unique<CVulkanCmdBuffer>() : nullptr; }
	int framegenTimestampBegin(CVulkanCmdBuffer*) const { return query; }
	void framegenTimestampEnd(CVulkanCmdBuffer*, int) {}
	int pipeline(int shader) const { return shader; }
	uint64_t submitFramegen(std::unique_ptr<CVulkanCmdBuffer> cmd, uint64_t wait, int, uint32_t, uint32_t, bool online) {
		assert(online && cmd->copies == 1 && cmd->barriers == 1 && cmd->dispatches >= 3);
		assert(wait == std::max(g_framegenHistory.lastCompositeSeqNo, g_framegenHistory.lastBaseIngestSeqNo));
		++submitted; return nextSeq++;
	}
} g_device;
#include "learning_upload.inc"
static uint32_t every = 1;
static bool prepare_ok = true, resources_ready = true, online_enabled = true, bidir_active = false;
static uint64_t clock_start = 1'000'000'000, clock_end = 1'000'500'000, deadline = 1'020'000'000;
static unsigned clock_reads = 0, prepares = 0;
static uint64_t get_time_in_nanos() { return clock_reads++ == 0 ? clock_start : clock_end; }
static uint64_t framegen_learning_deadline() { return deadline; }
static bool vulkan_framegen_is_enabled() { return true; }
static bool vulkan_framegen_bidir_active() { return bidir_active; }
static bool framegen_net_online_enabled() { return online_enabled; }
static uint32_t framegen_net_online_every() { return every; }
static float framegen_net_online_lr() { return 0.001f; }
static bool framegen_net_requested(GamescopeFramegenPipeline) { return true; }
static bool framegen_adapt_enabled(GamescopeFramegenPipeline) { return true; }
static const char *framegen_record_dir() { return nullptr; }
static const char *framegen_color_record_dir() { return nullptr; }
static uint32_t framegen_record_max() { return 1000; }
static uint32_t framegen_hud_level() { return 0; }
static const char *framegen_net_profile_path() { return profile_path.empty() ? nullptr : profile_path.c_str(); }
struct FramegenDispatch_t {};
static const FramegenDispatch_t &framegen_dispatch_for_format(uint32_t) { static FramegenDispatch_t d; return d; }
struct { GamescopeFramegenMode mode = GamescopeFramegenMode::Motion; GamescopeFramegenPipeline pipeline = GamescopeFramegenPipeline::Checked; } effective;
static auto framegen_effective_config(uint32_t) { return effective; }
static GamescopeFramegenPipeline g_eFramegenPipeline = GamescopeFramegenPipeline::Spatial;
static bool framegen_learning_resources_ready(const FramegenDispatch_t&) { return resources_ready; }
static void framegen_adapt_consume(GamescopeFramegenPipeline) {}
static void framegen_motion_bind_sampler(CVulkanCmdBuffer *cmd, uint32_t slot, CVulkanTexture *texture, bool) {
	if (slot == 0) cmd->slot_zero.emplace_back(cmd->pipeline, texture);
}
static bool framegen_shading_enabled(GamescopeFramegenPipeline) { return false; }
static bool framegen_motion_history_valid(GamescopeFramegenPipeline) { return false; }
static int framegen_luma_reservoir_read_index() { return -1; }
static float framegen_motion_history_time_scale() { return 1; }
static void framegen_net_note_skip(NetLearningSkip skip) { g_framegenNetLearningPair.skip(skip); }
static bool framegen_prepare_motion(CVulkanCmdBuffer*, uint32_t, uint32_t, const FramegenDispatch_t&, GamescopeFramegenPipeline) {
	++prepares; g_framegenMotion.uMotionFieldFrameId = 0; g_framegenMotion.uMotionFieldIntervalNs = 0;
	g_framegenMotion.bNetActive = false; g_framegenMotion.bUncheckedFieldValid = true; return prepare_ok;
}
static void framegen_record_net(CVulkanCmdBuffer *cmd, uint32_t, uint32_t, bool) {
	framegen_record_net_upload_barrier(cmd);
	g_framegenMotion.bNetActive = true; g_framegenMotion.bNetWeightsUploaded = false;
	g_bFramegenNetStagingStale = true; g_framegenMotion.bNetStatePending = true;
}
static void framegen_record_adapt_probe(CVulkanCmdBuffer*, uint32_t, uint32_t) {}
enum class FramegenResetReason_t { Grid, Cut, Hitch, Ring };
static void framegen_metrics_note_reset(FramegenResetReason_t) {}
static void framegen_metrics_note_discard(unsigned) {}
static void framegen_invalidate_deadline_shadow_content() {}
static void framegen_color_probe_consume() {}
static void framegen_clear_pending() { g_framegenHistory.pending.clear(); }
static unsigned framegen_erase_interpolations() { framegen_clear_pending(); return 0; }
static struct { bool bPresentBiasHitchEpisode = false; uint64_t token = 999; } g_framegenPresentState;
static struct { std::vector<int> framegenOutputImages, framegenPresentImages, framegenCursorHistoryImages; } g_output;
static struct { CVulkanTexture *anchor = nullptr, *reference = nullptr; uint64_t anchorId = 0, referenceId = 0, anchorTimeNs = 0, referenceTimeNs = 0, lastRealTimeNs = 0; } g_framegenColorProbe;

static void framegen_net_profile_consume();
#include "learning_busy.inc"
#include "learning_drop.inc"
#include "learning_store.inc"
#include "learning_write.inc"
#include "learning_snapshot.inc"
#include "learning_join.inc"
#include "learning_flush.inc"
#include "learning_consume.inc"
#include "learning_key.inc"
#include "learning_train.inc"
#include "learning_pair.inc"
#include "learning_transaction.inc"
#include "learning_idle.inc"
#include "learning_pins.inc"
#include "learning_scene.inc"
#include "learning_reset.inc"

static CVulkanTexture texture, previous_texture, current_texture, prior_texture, gradient_texture, upload_texture, weights, stats;
static void setup()
{
	assert(!g_framegenNetWriteThread.joinable());
	g_framegenMotion = {}; g_framegenHistory = {}; g_device = {}; effective = {};
	g_framegenMotion.bNetWeightsUploaded = true; g_framegenMotion.uNetTrainStep = 7;
	g_framegenMotion.bNetUploadInitialized = true;
	g_framegenMotion.width = 100; g_framegenMotion.height = 75;
	g_framegenMotion.uMotionFieldFrameId = 20; g_framegenMotion.uMotionFieldIntervalNs = 30;
	g_framegenMotion.learning.warmedSeqNo = 1; g_framegenMotion.learning.recordCostNs = 500'000;
	g_framegenHistory.lastGeneratedSeqNo = g_framegenHistory.lastFramegenWorkSeqNo = 2;
	g_framegenHistory.lastCompositeSeqNo = 3; g_framegenHistory.currentFrameId = 20;
	g_framegenHistory.valid = g_framegenHistory.bHudFreePrevious = g_framegenHistory.bHudFreeCurrent = true;
	g_framegenHistory.nDegradeSteps = 1;
	g_framegenMotion.netState = &texture;
	g_framegenMotion.netGradSlices = &gradient_texture; g_framegenMotion.netWeightsPrior = &prior_texture;
	g_framegenMotion.netWeightsUpload = &upload_texture;
	upload_texture.image = reinterpret_cast<VkImage>(uintptr_t{71});
	g_framegenMotion.statsAccum = g_framegenMotion.netWeightsGpu = g_framegenMotion.mvFieldRevChk = &texture;
	g_framegenMotion.lumaPrev = g_framegenMotion.lumaCur = g_framegenMotion.mvField = &texture;
	g_framegenMotion.netProfileReadback = &weights; g_framegenMotion.statsReadback = &stats;
	g_framegenHistory.previousReal = &previous_texture;
	g_framegenHistory.currentReal = &current_texture;
	weights.pitch = k_uFramegenNetTexW*sizeof(float)+64;
	weights.bytes.assign(37 + weights.pitch*k_uFramegenNetTexH, 0);
	stats.pitch = gamescope::framegen::k_uAdaptationStatsCount*sizeof(uint32_t);
	stats.bytes.assign(37 + stats.pitch, 0);
	uint32_t total = 1; memcpy(stats.mappedPixels(), &total, sizeof(total));
	g_framegenMetricsPendingEvents = {}; g_framegenMetricsPendingEvents.trainedSteps = 5;
	g_ulFramegenNetProgress = g_ulFramegenNetLiveProgress = 10;
	g_framegenNetLearningPair = {}; g_framegenNetLearningPair.begin(20);
	g_bFramegenNetStagingStale = false; g_bFramegenNetKeyActive = true; g_uFramegenNetActiveKey = 0xf0;
	store = {}; store.neutral.assign(k_uFramegenNetFloats, 0.0f);
	g_framegenNetLiveWeights.assign(k_uFramegenNetFloats, 0.25f); g_framegenNetReadbackWeights.clear();
	g_ulFramegenNetSavedProgress = 0; g_ulFramegenNetSavedRevision = 0; g_bFramegenNetWriteInFlight = false;
	every = 1; prepare_ok = resources_ready = online_enabled = true; bidir_active = false;
	clock_reads = prepares = 0; clock_start = 1'000'000'000; clock_end = 1'000'500'000; deadline = 1'020'000'000;
	profile_path.clear(); events.clear();
}
static auto recording_state()
{
	const auto &m = g_framegenMotion; const auto &p = g_framegenNetLearningPair;
	return std::tuple(m.bNetWeightsUploaded, g_bFramegenNetStagingStale, m.bNetStatePending,
		m.bNetUploadInitialized, m.bNetActive, m.bUncheckedFieldValid, m.uNetTrainStep, m.netTrainingCadence.consideredPair,
		m.netTrainingCadence.counter, m.uMotionFieldFrameId, m.uMotionFieldIntervalNs,
		g_ulFramegenNetProgress, g_framegenMetricsPendingEvents.trainedSteps, p.realId, p.stepped, p.reason);
}
static void upload_contracts()
{
	setup(); g_framegenMotion.bNetUploadInitialized = false;
	CVulkanCmdBuffer command;
	const auto before = recording_state();
	{
		FramegenLearningRecording cancelled;
		framegen_record_net_upload_barrier(&command);
		assert(g_framegenMotion.bNetUploadInitialized);
	}
	assert(recording_state() == before);
	{
		FramegenLearningRecording accepted;
		framegen_record_net_upload_barrier(&command);
		accepted.committed = true;
	}
	framegen_record_net_upload_barrier(&command);
	assert(g_framegenMotion.bNetUploadInitialized && g_device.vk.barriers.size() == 3);
	assert(command.dispatches == 0 && command.copies == 0 && command.barriers == 0 && events.empty());
	for (size_t index = 0; index < g_device.vk.barriers.size(); ++index)
	{
		const auto &observed = g_device.vk.barriers[index];
		const auto &image = observed.image;
		assert(observed.command == command.rawBuffer());
		assert(observed.source == VK_PIPELINE_STAGE_HOST_BIT && observed.destination == VK_PIPELINE_STAGE_TRANSFER_BIT);
		assert(observed.dependencies == 0 && image.sType == VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER && image.pNext == nullptr);
		assert(image.srcAccessMask == VK_ACCESS_HOST_WRITE_BIT && image.dstAccessMask == VK_ACCESS_TRANSFER_READ_BIT);
		assert(image.oldLayout == (index < 2 ? VK_IMAGE_LAYOUT_PREINITIALIZED : VK_IMAGE_LAYOUT_GENERAL));
		assert(image.newLayout == VK_IMAGE_LAYOUT_GENERAL && image.image == upload_texture.vkImage());
		assert(image.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED && image.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);
		const auto &range = image.subresourceRange;
		assert(range.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT && range.baseMipLevel == 0 && range.levelCount == 1);
		assert(range.baseArrayLayer == 0 && range.layerCount == 1);
	}
	// Key/scene changes reuse the same staging image; a full resource reset
	// creates a different image and must start from PREINITIALIZED again.
	framegen_net_activate_key(0x100f0);
	assert(g_framegenMotion.bNetUploadInitialized);
	vulkan_framegen_invalidate_history("net_input_key_change");
	assert(g_framegenMotion.bNetUploadInitialized);
	vulkan_framegen_reset("upload-resource-reset");
	assert(!g_framegenMotion.bNetUploadInitialized);

	setup(); g_framegenMotion.bNetStatePending = true;
	CVulkanCmdBuffer initial;
	assert(framegen_record_net_train(&initial, GamescopeFramegenPipeline::Learned, true));
	std::vector<CVulkanTexture*> optimizer_inputs;
	for (const auto &[pipeline, texture] : initial.slot_zero)
		if (pipeline == SHADER_TYPE_FRAMEGEN_MOTION_NET_OPT) optimizer_inputs.push_back(texture);
	assert((optimizer_inputs == std::vector<CVulkanTexture*>{&prior_texture, &gradient_texture}));
	CVulkanCmdBuffer subsequent;
	assert(framegen_record_net_train(&subsequent, GamescopeFramegenPipeline::Learned, true));
	optimizer_inputs.clear();
	for (const auto &[pipeline, texture] : subsequent.slot_zero)
		if (pipeline == SHADER_TYPE_FRAMEGEN_MOTION_NET_OPT) optimizer_inputs.push_back(texture);
	assert((optimizer_inputs == std::vector<CVulkanTexture*>{&gradient_texture}));
	assert(g_framegenMotion.netTrainingCadence.counter == 0); // GPU recorder has no scheduling policy.
}
static void recording_contracts()
{
	for (int failure = 0; failure < 5; ++failure)
	{
		setup();
		if (failure == 0) deadline = clock_start+1'000'000; // Initial denial.
		if (failure == 1) prepare_ok = false;
		if (failure == 2) g_device.query = -1;
		if (failure == 3) g_device.commandOK = false;
		if (failure == 4) clock_end = deadline; // Late cancellation after real training mutations.
		const auto before = recording_state();
		vulkan_framegen_train_idle();
		assert(g_device.submitted == 0 && recording_state() == before);
		assert(g_framegenHistory.lastGeneratedSeqNo == 2 && g_framegenHistory.lastLearningSeqNo == 0);
		if (failure == 0) assert(g_device.commands == 0 && prepares == 0);
		if (failure == 4) assert(prepares == 1 && g_framegenMotion.learning.recordCostNs >= 20'000'000);
	}
	setup(); every = 3;
	CVulkanCmdBuffer foreground;
	assert(framegen_record_net_train_pair(&foreground, GamescopeFramegenPipeline::Learned, true));
	assert(!framegen_record_net_train_pair(&foreground, GamescopeFramegenPipeline::Learned, true));
	const auto progress = g_ulFramegenNetProgress;
	for (int poll = 0; poll < 4; ++poll) vulkan_framegen_train_idle();
	assert(g_ulFramegenNetProgress == progress && g_framegenMotion.netTrainingCadence.counter == 1);
	g_framegenHistory.currentFrameId++;
	for (int poll = 0; poll < 4; ++poll) vulkan_framegen_train_idle();
	assert(g_framegenMotion.netTrainingCadence.counter == 2 && g_device.commands == 0);
	assert(g_framegenMotion.netTrainingCadence.consideredPair == g_framegenHistory.currentFrameId);
	assert(!framegen_record_net_train_pair(&foreground, GamescopeFramegenPipeline::Learned, true));

	setup(); clock_end = deadline; vulkan_framegen_train_idle();
	assert(g_framegenMotion.netTrainingCadence.consideredPair == 0);
	clock_start += 300'000'000; clock_end = clock_start+500'000; deadline = clock_start+100'000'000; clock_reads = 0;
	vulkan_framegen_train_idle();
	assert(g_device.submitted == 1 && g_ulFramegenNetProgress == 11);
	assert(g_framegenMetricsPendingEvents.trainedSteps == 6 && g_framegenNetLearningPair.stepped);
	assert(g_framegenHistory.lastGeneratedSeqNo == 2 && g_framegenPresentState.token == 999);
	assert(g_framegenHistory.pending.empty() && g_framegenHistory.lastLearningSeqNo == 101);
	assert(g_framegenHistory.genReadA == &previous_texture && g_framegenHistory.genReadB == &current_texture);
	assert(g_framegenHistory.genReadSeqNo == 101 && g_framegenHistory.ulNetProfileSeqNo == 101);
	assert(!framegen_record_net_train_pair(&foreground, GamescopeFramegenPipeline::Learned, true));
	assert(g_ulFramegenNetProgress == 11);
}
static void admission_contracts()
{
	for (int gate = 0; gate < 8; ++gate)
	{
		setup();
		if (gate == 0) g_device.trainingCost = 0;
		if (gate == 1) g_framegenMotion.learning.warmedSeqNo = 0;
		if (gate == 2) g_framegenMotion.learning.recordCostNs = 0;
		if (gate == 3) resources_ready = false;
		if (gate == 4) { g_device.completed = 3; g_framegenHistory.lastLearningSeqNo = 4; }
		if (gate == 5) g_device.baseCompleted = 2;
		if (gate == 6) { g_device.split = true; g_framegenHistory.bBaseLayer = true; }
		if (gate == 7) effective.pipeline = GamescopeFramegenPipeline::Learned;
		vulkan_framegen_train_idle(); assert(g_device.commands == 0 && g_device.submitted == 0);
	}
	setup(); g_device.completed = 2; g_framegenHistory.lastFramegenWorkSeqNo = 3;
	assert(framegen_work_completed()); // Base copies are lifetime work, not generation headroom.
	g_framegenHistory.lastLearningSeqNo = 3; assert(!framegen_work_completed());
	g_device.completed = 3; assert(framegen_work_completed());
}
static void denied_learning_recovery_contract()
{
	using namespace gamescope::framegen;
	setup(); g_device.trainingCost = 0;
	bool tried = false;
	for (uint32_t decision = 0;
		decision < k_uDeadlineRecoveryHeadroomDecisions + k_uDeadlineRecoveryBaseBackoffDecisions; ++decision)
	{
		// Each real pair offers optional learning; unknown GPU cost denies all
		// of them. Ordinary admitted generation independently supplies headroom.
		g_framegenHistory.currentFrameId++;
		clock_start += 300'000'000; clock_reads = 0; deadline = clock_start+20'000'000;
		const auto before = g_framegenHistory.recovery;
		vulkan_framegen_train_idle();
		const auto &after = g_framegenHistory.recovery;
		assert(after.streak == before.streak && after.backoffDecisions == before.backoffDecisions);
		assert(after.probationRemaining == before.probationRemaining && after.decisionsSinceClimb == before.decisionsSinceClimb);
		assert(after.blockedThresholdReported == before.blockedThresholdReported);
		assert(g_device.commands == 0 && g_device.submitted == 0);
		const auto evaluation = evaluate_ladder_recovery(after,
			g_framegenHistory.nDegradeSteps, g_framegenHistory.nDegradeHold,
			2'000'000, k_uDeadlineMinSamples, 3'000'000, k_uDeadlineMinSamples, 10'000'000);
		g_framegenHistory.recovery = evaluation.state;
		if (evaluation.tryRecover) { tried = true; break; }
	}
	assert(tried && g_device.submitted == 0 && g_framegenMotion.netTrainingCadence.counter == 0);
}
static void write_weights(float value)
{
	for (uint32_t i = 0; i < k_uFramegenNetFloats; ++i)
		memcpy(weights.mappedPixels()+(i/k_uFramegenNetTexW)*weights.pitch+(i%k_uFramegenNetTexW)*sizeof(float), &value, sizeof(value));
}
static void warmth_contracts()
{
	for (int mismatch = 0; mismatch < 6; ++mismatch)
	{
		setup(); write_weights(0.5f); g_framegenMotion.learning = {};
		g_framegenHistory.ulNetProfileSeqNo = g_framegenHistory.ulAdaptStatsSeqNo = 101;
		g_device.completed = g_device.trainingSample = 101; g_ulFramegenNetProgress = 11;
		g_framegenMotion.learning.pendingRecordSeqNo = 101; g_framegenMotion.learning.pendingRecordNs = 800'000;
		if (mismatch == 0) g_framegenHistory.ulAdaptStatsSeqNo = 100;
		if (mismatch == 1) g_device.trainingSample = 100;
		if (mismatch == 2) g_device.trainingCost = 0;
		if (mismatch == 3) { uint32_t cut = 1; memcpy(stats.mappedPixels()+88*sizeof(uint32_t), &cut, sizeof(cut)); }
		if (mismatch == 4) memset(stats.mappedPixels(), 0, stats.pitch);
		if (mismatch == 5) g_framegenMotion.learning.pendingRecordSeqNo = 100;
		framegen_net_profile_consume();
		assert(g_ulFramegenNetLiveProgress == 11); // Healthy publication is separate from cost evidence.
		assert(g_framegenMotion.learning.warmedSeqNo == (mismatch == 5 ? 101u : 0u));
		assert(g_framegenMotion.learning.recordCostNs == 0);
	}
}
static void readback_contracts(const char *path)
{
	setup(); write_weights(0.5f);
	store.sets.push_back({0x100f0, std::vector<float>(k_uFramegenNetFloats, 0.75f)});
	const std::vector<float> expected(k_uFramegenNetFloats, 0.5f);
	g_framegenMotion.learning = {};
	g_framegenHistory.ulNetProfileSeqNo = g_framegenHistory.ulAdaptStatsSeqNo = 101;
	g_device.trainingSample = 101; g_ulFramegenNetProgress = 11;
	framegen_net_profile_consume(); assert(g_ulFramegenNetLiveProgress == 10);
	g_device.completed = 101; framegen_net_profile_consume();
	assert(g_framegenNetLiveWeights == expected && g_ulFramegenNetLiveProgress == 11);
	assert(g_framegenMotion.learning.warmedSeqNo == 101 && g_framegenMotion.learning.recordCostNs == 0);
	g_framegenHistory.ulNetProfileSeqNo = g_framegenHistory.ulAdaptStatsSeqNo = 102;
	g_device.trainingSample = g_device.completed = 102;
	g_framegenMotion.learning.pendingRecordSeqNo = 102; g_framegenMotion.learning.pendingRecordNs = 800'000;
	g_ulFramegenNetProgress = 12; framegen_net_profile_consume();
	assert(g_framegenMotion.learning.recordCostNs == 800'000);
	framegen_net_profile_consume(); assert(g_framegenMetricsPendingEvents.publishedSteps == 2);
	profile_path = path; framegen_net_profile_flush();
	assert(g_ulFramegenNetSavedProgress == 12);
	std::ifstream input(path, std::ios::binary);
	const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
	const auto saved = gamescope::framegen::decode_net_profile(bytes);
	assert(saved && saved->size() == 2 && (*saved)[1].key == 0xf0 && (*saved)[1].weights == expected);
	assert((*saved)[0].key == 0x100f0 && (*saved)[0].weights == std::vector<float>(k_uFramegenNetFloats, 0.75f));
	for (float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
	{
		g_framegenHistory.ulNetProfileSeqNo++; g_device.completed++;
		write_weights(bad); g_ulFramegenNetProgress++;
		framegen_net_profile_consume(); framegen_net_profile_flush();
		assert(g_framegenNetLiveWeights.front() == 0.5f && g_ulFramegenNetLiveProgress == 12 && g_ulFramegenNetSavedProgress == 12);
		assert(!g_framegenMotion.bNetWeightsUploaded && g_framegenMotion.bNetStatePending);
	}
	assert(g_framegenMetricsPendingEvents.rejectedNetReadbacks == 2);
	online_enabled = false; g_ulFramegenNetLiveProgress++;
	framegen_net_profile_flush(); assert(g_ulFramegenNetSavedProgress == 12);
	std::ifstream unchanged(path, std::ios::binary);
	assert(std::vector<uint8_t>((std::istreambuf_iterator<char>(unchanged)), {}) == bytes);
}
static void transition_contracts()
{
	setup();
	store.sets.push_back({0x100f0, std::vector<float>(k_uFramegenNetFloats, 0.75f)});
	g_framegenHistory.ulNetProfileSeqNo = 101; write_weights(0.9f);
	framegen_net_activate_key(0x100f0);
	assert(g_uFramegenNetActiveKey == 0x100f0 && g_framegenNetLiveWeights.empty());
	assert(g_framegenNetActivePrior.front() == 0.75f && store.sets[1].key == 0xf0 && store.sets[1].weights.front() == 0.25f);
	assert(g_framegenHistory.ulNetProfileSeqNo == 0 && g_framegenMetricsPendingEvents.droppedNetReadbacks == 1);
	assert(g_framegenMotion.learning.warmedSeqNo == 0 && g_device.trainingCost == 0);
	g_device.completed = 101; framegen_net_profile_consume(); assert(g_framegenNetLiveWeights.empty());
	setup();
	g_framegenHistory.lastLearningSeqNo = g_framegenHistory.lastFramegenWorkSeqNo = g_framegenHistory.genReadSeqNo = 101;
	g_framegenHistory.genReadA = &previous_texture; g_framegenHistory.genReadB = &current_texture;
	g_framegenHistory.ulNetProfileSeqNo = 101;
	vulkan_framegen_invalidate_history("net_input_key_change");
	assert(g_framegenHistory.genReadA == &previous_texture && g_framegenHistory.genReadB == &current_texture && g_framegenHistory.genReadSeqNo == 101);
	assert(!g_framegenHistory.valid && g_framegenMotion.learning.warmedSeqNo == 0 && g_framegenHistory.ulNetProfileSeqNo == 0);
	assert(g_framegenMetricsPendingEvents.droppedNetReadbacks == 1);
	vulkan_framegen_reset("fixture-pending-learning");
	assert(!events.empty() && events.front() == "reset-costs");
	assert(std::ranges::find(events, "wait-learning") != events.end());
	assert(g_framegenHistory.lastLearningSeqNo == 0 && g_framegenHistory.genReadA == nullptr);
}
#endif

int main(int argc, char **argv)
{
	policy_contracts();
#ifdef FRAMEGEN_PRODUCTION_FIXTURE
	assert(argc == 2);
	upload_contracts(); recording_contracts(); admission_contracts(); denied_learning_recovery_contract();
	warmth_contracts(); readback_contracts(argv[1]); transition_contracts();
#else
	(void)argc; (void)argv;
#endif
}
