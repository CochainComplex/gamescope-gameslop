// CPU fixture. Policy/selector/caller/correction bodies are extracted
// unchanged from the selected production renderer, never reimplemented here.
#include "framegen/deadline.hpp"
#include "framegen/metrics.hpp"
#include "framegen/policy.hpp"
#include "framegen/temporal.hpp"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using namespace gamescope::framegen;
using FramegenEffective_t = EffectiveConfig;
using FramegenSlotRequest_t = SlotRequest;
static constexpr uint64_t T = 16'666'667u, R = 1'000'000'000u;
static constexpr uint64_t wakeLead = 2'000'000u;
static constexpr uint64_t k_ulFramegenMaxRealFrameGapNs = 1'000'000'000u;
static constexpr float k_flFramegenMaxForwardStrength = 1.5f;
static GamescopeFramegenMode g_eFramegenMode = GamescopeFramegenMode::Motion;
static GamescopeFramegenPipeline g_eFramegenPipeline = GamescopeFramegenPipeline::Checked;
static uint32_t g_nFramegenMultiplier = 2u;
static float g_flFramegenStrength = 0.5f;
static bool g_bFramegenDebug = false;

static unsigned failures = 0, checks = 0;
static const char *caseName = "setup";
static void check(bool value, const char *expression, int line)
{
    ++checks;
    if (!value) {
        ++failures;
        std::fprintf(stderr, "FAIL %s line%d: %s\n", caseName, line, expression);
    }
}
#define CHECK(x) check(static_cast<bool>(x), #x, __LINE__)

namespace gamescope {
struct VBlankScheduleTime {
    uint64_t ulTargetVBlank = 0, ulScheduledWakeupPoint = 0;
};
}
static struct Timer {
    gamescope::VBlankScheduleTime first, fresh;
    unsigned queries = 0;
    gamescope::VBlankScheduleTime CalcNextWakeupTime(bool preemptive) {
        CHECK(preemptive);
        return ++queries == 1u ? first : fresh;
    }
} timer;
static Timer &GetVBlankTimer() { return timer; }
static struct { gamescope::VBlankScheduleTime schedule; } g_SteamCompMgrVBlankTime;
static struct Backend {
    bool native = false;
    bool OwnsKMSPresentTiming() const { return native; }
} backend;
static Backend *GetBackend() { return &backend; }

struct FramegenHistory_t {
    struct PendingGenerated_t {
        bool bReal = false;
        uint64_t ulAnchorRealFrameId = 42u;
        bool bProvisional = true;
        std::optional<uint64_t> oTargetVBlankNs;
        uint64_t ulWakeDeadlineNs = 0;
    };
    bool valid = true, bCadenceUsesSourceTime = true, bCausalDeadlineMissed = false;
    void *previousReal = nullptr, *currentReal = nullptr;
    uint64_t currentPresentTimeNs = R, currentFrameId = 42u;
    uint64_t ulLastBaseCommitID = 42u;
    uint64_t ulDeadlineGridIntervalNs = T, ulDeadlineGridEpoch = 1u;
    uint64_t ulLastPlannedTargetNs = 0, lastCompositeSeqNo = 1u;
    uint32_t nDegradeSteps = 0, nDegradeHold = 0, drmFormat = 0;
    RealAnchorState_t causalAnchor;
    CadencePredictorState cadence;
    DeadlineMissState_t causalDeadlineMisses;
    RecoveryState_t recovery;
    std::vector<PendingGenerated_t> pending;
} g_framegenHistory;
static struct {
    DisplayChainTimingState_t displayTiming;
    bool bPresentBiasHitchEpisode = false;
} g_framegenPresentState;
static struct {
    uint64_t uMotionFieldFrameId = 0;
    GamescopeFramegenPipeline eMotionFieldPipeline = GamescopeFramegenPipeline::Checked;
    bool bMotionFieldBidir = false;
} g_framegenMotion;
struct FramegenDispatch_t { bool motionSupported = true; };
static FramegenDispatch_t dispatch;
static const FramegenDispatch_t &framegen_dispatch_for_format(uint32_t) { return dispatch; }
static bool motionAvailable = true;
static void *framegen_motion_field() { return motionAvailable ? &dispatch : nullptr; }

struct CVulkanDevice {
    static constexpr uint32_t kFramegenLadderSlots = 8u;
    std::array<std::array<LadderRungCost_t, 3>, kFramegenLadderSlots> costs = {};
    std::vector<std::pair<uint32_t,uint32_t>> reads;
    uint64_t framegenRungCostNs(uint32_t rung, uint32_t key) {
        reads.emplace_back(rung,key);
        return costs.at(rung).at(key).costNs;
    }
    uint32_t framegenRungSampleCount(uint32_t rung, uint32_t key) {
        return costs.at(rung).at(key).samples;
    }
} g_device;
static struct Logger { template<class... Args> void infof(const char *, Args...) {} } vk_log;
static bool FramegenDebugShouldLog(uint64_t &) { return false; }
static bool enabled = true, causal = true, vrr = false, bidir = false, recoverEnabled = true;
static bool completed = true, submitSuccess = true;
static uint64_t clockNs = 0, commitLeadOverride = 0;
static unsigned completionQueries = 0, submitCalls = 0, admissionSkips = 0;
static unsigned realEligibilityQueries = 0;
static uint64_t nextRealEligibleVBlankNs = 0;
static NetLearningSkip lastSkip = NetLearningSkip::NoSkip;
static FramegenSlotRequest_t submittedRequest = {};
static FramegenEffective_t submittedEffective = {};
static bool vulkan_framegen_is_enabled() { return enabled; }
static bool framegen_causal_deadline_enabled() { return causal; }
static bool vulkan_framegen_vrr_hybrid_active() { return vrr; }
static bool vulkan_framegen_bidir_active() { return bidir; }
static bool framegen_recovery_enabled() { return recoverEnabled; }
static bool framegen_work_completed() { ++completionQueries; return completed; }
static uint64_t get_time_in_nanos() { return clockNs; }
static uint64_t framegen_commit_lead_override_ns() { return commitLeadOverride; }
static uint64_t framegen_predicted_interval_ns() {
    return predicted_cadence_interval_ns(g_framegenHistory.cadence);
}
[[maybe_unused]] static uint64_t steamcompmgr_framegen_next_real_eligible_vblank(
    uint64_t commitID, uint64_t periodNs)
{
    ++realEligibilityQueries;
    CHECK(commitID == g_framegenHistory.ulLastBaseCommitID);
    CHECK(periodNs == g_framegenHistory.ulDeadlineGridIntervalNs);
    return nextRealEligibleVBlankNs;
}
static void framegen_net_note_skip(NetLearningSkip skip) { lastSkip = skip; }
static void framegen_metrics_note_admission_skip() { ++admissionSkips; }
static void vulkan_framegen_invalidate_history(const char *) { g_framegenHistory.valid = false; }
static void framegen_log_ladder_recovery_blocked(uint32_t,uint32_t) {}
static void framegen_log_ladder_recovery(uint32_t,uint32_t,uint32_t) {}
static bool framegen_submit_planned(const FramegenSlotRequest_t *requests,
    uint32_t count, uint32_t, const FramegenEffective_t &effective,
    uint64_t, uint32_t, bool clear, void *, uint64_t target, bool keyed)
{
    ++submitCalls;
    CHECK(count == 1u);
    CHECK(!clear);
    CHECK(keyed);
    CHECK(target == requests[0].targetFlipNs);
    submittedRequest = requests[0];
    submittedEffective = effective;
    return submitSuccess;
}

#include "config.inc"
#include "commit_plan.inc"
#include "recovery_active.inc"
#include "recovery_capacity.inc"
#include "recovery_degradation.inc"
#include "selector.inc"
#include "caller.inc"
#include "learning_deadline.inc"

static size_t runCorrection(const AnchorCorrection_t &correction)
{
    const size_t before = g_framegenHistory.pending.size();
    const uint64_t ulNowNs = get_time_in_nanos();
    const uint64_t ulArrivalGuardNs = cadence_arrival_guard_ns(g_framegenHistory.ulDeadlineGridIntervalNs);
#include "correction.inc"
    return before - g_framegenHistory.pending.size();
}

static void setCost(uint32_t rung, uint64_t cost, uint32_t samples = k_uDeadlineMinSamples)
{
    g_device.costs.at(rung).at(1) = {cost,samples};
    g_device.costs.at(rung).at(2) = {cost,samples};
}

static void setup(int64_t bias = 0)
{
    g_framegenHistory = {};
    g_framegenPresentState = {};
    g_framegenMotion = {};
    dispatch = {};
    g_device = {};
    timer = {};
    backend = {};
    enabled = causal = completed = submitSuccess = recoverEnabled = motionAvailable = true;
    vrr = bidir = false;
    clockNs = R+T-wakeLead-1'000'000u;
    commitLeadOverride = 0;
    completionQueries = submitCalls = admissionSkips = 0;
    realEligibilityQueries = 0;
    nextRealEligibleVBlankNs = 0;
    lastSkip = NetLearningSkip::NoSkip;
    submittedRequest = {};
    submittedEffective = {};
    timer.first = timer.fresh = {R+T,R+T-wakeLead};
    g_SteamCompMgrVBlankTime.schedule = {R,R-wakeLead};
    g_framegenPresentState.displayTiming.presentBias.emaNs = bias;
    g_framegenHistory.previousReal = &timer;
    g_framegenHistory.currentReal = &backend;
    g_framegenHistory.cadence = {.intervalNs = 2u*T, .samples = k_uCadencePredictorMinSamples};
    g_framegenHistory.causalAnchor = {
        .realFrameId = 42u, .sourceReadyNs = R-2u*T,
        .provisionalTargetNs = apply_present_bias_ns(R,bias),
        .provisionalBiasNs = bias, .epoch = 1u};
    for (uint32_t rung=0; rung<CVulkanDevice::kFramegenLadderSlots; ++rung)
        setCost(rung,500'000u);
}

static auto recoveryKey(const RecoveryState_t &state)
{
    return std::tuple(state.streak,state.backoffDecisions,state.probationRemaining,
        state.decisionsSinceClimb,state.blockedThresholdReported);
}

static void positiveAndBoundaryCases()
{
    caseName = "positive bias phase-only skip preserves ladder";
    for (int64_t bias : {int64_t(1'000),int64_t(1'000'000)}) {
        setup(bias);
        setCost(0u,bias==1'000 ? 850'001u : 1'500'000u);
        g_framegenHistory.nDegradeHold = 2u;
        g_framegenHistory.recovery = {12u,360u,20u,40u,true};
        const auto before = recoveryKey(g_framegenHistory.recovery);
        CHECK(!framegen_causal_submit(1u,false));
        CHECK(submitCalls == 0u);
        CHECK(g_framegenHistory.nDegradeSteps == 0u);
        CHECK(g_framegenHistory.nDegradeHold == 2u);
        CHECK(recoveryKey(g_framegenHistory.recovery) == before);
        CHECK(timer.queries == 2u); // one planning snapshot + one fresh raw guard
    }
    caseName = "85 percent raw bound exact equality";
    for (uint64_t cost : {850'000u,850'001u}) {
        setup(1'000'000);
        setCost(0u,cost);
        const bool accepted = framegen_causal_submit(1u,false);
        CHECK(accepted == (cost==850'000u));
        CHECK(submitCalls == (cost==850'000u ? 1u : 0u));
        if (accepted) {
            CHECK(submittedRequest.wakeDeadlineNs == R+T-wakeLead);
            CHECK(submittedRequest.targetFlipNs == R+T+1'000'000u);
            CHECK(submittedRequest.targetVBlankNs == std::optional<uint64_t>(R+T));
        }
    }
    caseName = "negative and zero bias keep stricter deadline";
    for (int64_t bias : {int64_t(-200'000),int64_t(0)}) {
        setup(bias);
        setCost(0u,700'000u);
        CHECK(framegen_causal_submit(1u,false) == (bias==0));
        if (bias==0) CHECK(submittedRequest.wakeDeadlineNs == R+T-wakeLead);
        CHECK(g_framegenHistory.nDegradeSteps == 0u);
    }
}

static void rawObservationAndStartCases()
{
    caseName = "fresh helper wake is observed once, not stale planning wake";
    setup(1'000'000);
    clockNs = timer.first.ulScheduledWakeupPoint-2'000'000u;
    timer.fresh.ulScheduledWakeupPoint -= 1'000'000u;
    setCost(0u,1'500'000u);
    CHECK(!framegen_causal_submit(1u,false));
    CHECK(submitCalls == 0u);
    CHECK(timer.queries == 2u);

    caseName = "one-period wake lead rejects cold cost at provisional start";
    setup(3'000'000);
    // Source-supported overload boundary: vblankmanager permits offset==T.
    // This is not presented as a typical N1 runtime state.
    clockNs = R-1'000'000u;
    g_framegenHistory.currentPresentTimeNs = clockNs;
    g_framegenHistory.bCadenceUsesSourceTime = false;
    g_framegenHistory.causalAnchor.sourceReadyNs = 0u;
    timer.first = timer.fresh = {R+T,R};
    g_SteamCompMgrVBlankTime.schedule = {R,R-T};
    setCost(0u,0u,0u);
    CHECK(!framegen_causal_submit(1u,false));
    CHECK(submitCalls == 0u);
    CHECK(timer.queries == 2u);
    CHECK(g_device.reads.empty());

    caseName = "cost starts at later provisional source bound";
    setup(1'000'000);
    clockNs = R-1'000'000u;
    g_framegenHistory.currentPresentTimeNs = clockNs;
    // Use the supported dedicated-queue fallback so a newly due source sample
    // does not reject the planner before this start-bound contract is reached.
    g_framegenHistory.bCadenceUsesSourceTime = false;
    g_framegenHistory.causalAnchor.sourceReadyNs = 0u;
    setCost(0u,deadline_budget_ns(T-wakeLead)+1u);
    CHECK(framegen_causal_submit(1u,false));
    CHECK(g_framegenHistory.nDegradeSteps > 0u);
    CHECK(submittedEffective.pipeline != g_eFramegenPipeline);
}

static void ladderCases()
{
    caseName = "prior miss cheaper-rung walk uses raw cost budget";
    setup(1'000'000);
    setCost(0u,2'000'000u);
    setCost(1u,1'500'000u);
    setCost(2u,500'000u);
    g_framegenHistory.bCausalDeadlineMissed = true;
    CHECK(framegen_causal_submit(1u,false));
    CHECK(g_framegenHistory.nDegradeSteps == 2u);
    CHECK(g_framegenHistory.nDegradeHold == k_uDeadlineHoldFrames);

    caseName = "recovery must not climb to richer raw-unfitting rung";
    setup(1'000'000);
    g_framegenHistory.nDegradeSteps = 1u;
    setCost(0u,1'500'000u);
    setCost(1u,200'000u);
    g_framegenHistory.recovery = {
        .streak = k_uDeadlineRecoveryHeadroomDecisions-1u,
        .decisionsSinceClimb = k_uDeadlineRecoveryBaseBackoffDecisions};
    CHECK(framegen_causal_submit(1u,false));
    CHECK(g_framegenHistory.nDegradeSteps == 1u);
    CHECK(g_framegenHistory.recovery.probationRemaining == 0u);

    caseName = "genuine capacity hold advances without submission";
    setup(1'000'000);
    setCost(0u,20'000'000u);
    g_framegenHistory.nDegradeHold = 2u;
    CHECK(!framegen_causal_submit(1u,false));
    CHECK(g_framegenHistory.nDegradeSteps == 0u);
    CHECK(g_framegenHistory.nDegradeHold == 1u);

    caseName = "cold current rung gets original nonblocking probe";
    for (auto sample : {LadderRungCost_t{0u,0u},LadderRungCost_t{2'000'000u,k_uDeadlineMinSamples-1u}}) {
        setup(1'000'000);
        setCost(0u,sample.costNs,sample.samples);
        CHECK(framegen_causal_submit(1u,false));
        CHECK(g_framegenHistory.nDegradeSteps == 0u);
        CHECK(completionQueries == 1u);
        CHECK(submittedRequest.wakeDeadlineNs == R+T-wakeLead);
    }
    caseName = "actual sampler selects cached work-class key";
    setup(1'000'000);
    g_framegenMotion.uMotionFieldFrameId = g_framegenHistory.currentFrameId;
    g_device.costs[0][1] = {1'500'000u,k_uDeadlineMinSamples};
    g_device.costs[0][2] = {500'000u,k_uDeadlineMinSamples};
    CHECK(framegen_causal_submit(1u,false));
    CHECK(!g_device.reads.empty() && g_device.reads.front().second == 2u);
}

static void nativeCases()
{
    caseName = "native warmup retains biased wake without raw helper";
    setup(1'000'000);
    backend.native = true;
    setCost(0u,1'500'000u);
    CHECK(framegen_causal_submit(1u,false));
    CHECK(timer.queries == 1u);
    CHECK(realEligibilityQueries == 0u);
    CHECK(submittedRequest.wakeDeadlineNs == R+T-wakeLead+1'000'000u);
    CHECK(!submittedRequest.targetVBlankNs);

    caseName = "native commit reservation only skips without degradation";
    setup(1'000'000);
    backend.native = true;
    g_framegenPresentState.displayTiming.presentLead.samples = k_uPresentLeadWarmupSamples;
    g_framegenPresentState.displayTiming.presentLead.emaNs = 2'000'000;
    commitLeadOverride = 2'000'000u;
    setCost(0u,1'500'000u);
    CHECK(!framegen_causal_submit(1u,false));
    CHECK(submitCalls == 0u);
    CHECK(g_framegenHistory.nDegradeSteps == 0u);
    CHECK(g_framegenHistory.ulLastPlannedTargetNs != 0u);
    CHECK(timer.queries == 1u);
    CHECK(realEligibilityQueries == 0u);
}

static void realEligibilityCases()
{
    caseName = "fresh owner eligibility reaches the provisional planner";
    const struct { uint64_t target; bool backup; } cases[] = {
        {0u,false}, {R+T+1'000u,false}, {R+2u*T,true},
    };
    for (const auto &row : cases) {
        setup();
        g_framegenHistory.ulLastBaseCommitID = 9001u;
        g_framegenHistory.causalAnchor.sourceReadyNs = clockNs + 100'000u - 2u*T;
        nextRealEligibleVBlankNs = row.target;
        CHECK(framegen_causal_submit(1u,false) == row.backup);
        CHECK(submitCalls == (row.backup ? 1u : 0u));
        CHECK(realEligibilityQueries == 1u);
        if (row.backup) CHECK(submittedRequest.targetVBlankNs == std::optional<uint64_t>(R+T));
    }
}

static void guardAndLearningCases()
{
    caseName = "busy batch returns before scheduler and cost reads";
    setup(1'000'000);
    completed = false;
    CHECK(!framegen_causal_submit(1u,false));
    CHECK(completionQueries == 1u && timer.queries == 0u && g_device.reads.empty());
    CHECK(realEligibilityQueries == 0u);
    CHECK(submitCalls == 0u);

    caseName = "pending batch preserves early lazy return";
    setup(1'000'000);
    g_framegenHistory.pending.push_back({});
    CHECK(!framegen_causal_submit(1u,false));
    CHECK(completionQueries == 0u && timer.queries == 0u && g_device.reads.empty());
    CHECK(realEligibilityQueries == 0u);
    CHECK(submitCalls == 0u);

    caseName = "failed submit does not consume successful-submit hold";
    setup(1'000'000);
    submitSuccess = false;
    g_framegenHistory.nDegradeHold = 2u;
    CHECK(!framegen_causal_submit(1u,false));
    CHECK(submitCalls == 1u && g_framegenHistory.nDegradeHold == 2u);
    CHECK(g_framegenHistory.ulLastPlannedTargetNs == 0u);

    caseName = "learning consumer honors the stored effective deadline";
    setup(1'000'000);
    CHECK(framegen_causal_submit(1u,false));
    // Model unchanged publication of the request's deadline into pending state;
    // execute the actual learning_deadline consumer after the timer advances.
    FramegenHistory_t::PendingGenerated_t pending;
    pending.ulWakeDeadlineNs = submittedRequest.wakeDeadlineNs;
    g_framegenHistory.pending.push_back(pending);
    timer.fresh = {R+2u*T,R+2u*T-wakeLead};
    CHECK(framegen_learning_deadline() == R+T-wakeLead);

    caseName = "inactive, VRR and bidir paths do not query owner eligibility";
    for (unsigned variant=0; variant<3; ++variant) {
        setup();
        if (variant==0) enabled = false;
        if (variant==1) vrr = true;
        if (variant==2) bidir = true;
        CHECK(!framegen_causal_submit(1u,false));
        CHECK(realEligibilityQueries == 0u && timer.queries == 0u);
    }
}

static AnchorCorrection_t prepareCorrection(int64_t bias, uint64_t cost,
    uint32_t samples = k_uDeadlineMinSamples)
{
    setup(bias);
    // Correction has no initial planning snapshot: its first query is fresh.
    timer.queries = 1u;
    setCost(0u,cost,samples);
    g_framegenHistory.pending.push_back({false,42u,true,R+T});
    return {.anchor = {.realFrameId = 42u}, .matched = true, .discardProvisional = true};
}

static void correctionCases()
{
    caseName = "positive correction cannot discard on biased-only budget";
    auto correction = prepareCorrection(1'000'000,1'500'000u);
    CHECK(runCorrection(correction) == 0u);
    CHECK(g_framegenHistory.pending.size() == 1u);
    CHECK(completionQueries == 0u); // short circuit: no speculative completion query

    caseName = "correction exact boundary and lazy completion";
    for (uint64_t cost : {850'000u,850'001u}) {
        correction = prepareCorrection(1'000'000,cost);
        CHECK(runCorrection(correction) == (cost==850'000u ? 1u : 0u));
        CHECK(completionQueries == (cost==850'000u ? 1u : 0u));
    }
    caseName = "negative correction preserves stricter bias bound";
    correction = prepareCorrection(-200'000,700'000u);
    CHECK(runCorrection(correction) == 0u);
    CHECK(completionQueries == 0u);

    caseName = "unknown replacement never discards pixels";
    for (auto sample : {LadderRungCost_t{0u,k_uDeadlineMinSamples},LadderRungCost_t{1u,k_uDeadlineMinSamples-1u}}) {
        correction = prepareCorrection(1'000'000,sample.costNs,sample.samples);
        CHECK(runCorrection(correction) == 0u);
        CHECK(completionQueries == 0u);
    }
    caseName = "busy completion retains pixels and ladder state";
    correction = prepareCorrection(1'000'000,500'000u);
    completed = false;
    g_framegenHistory.nDegradeHold = 2u;
    const auto state = recoveryKey(g_framegenHistory.recovery);
    CHECK(runCorrection(correction) == 0u);
    CHECK(completionQueries == 1u);
    CHECK(g_framegenHistory.nDegradeHold == 2u);
    CHECK(recoveryKey(g_framegenHistory.recovery) == state);

    caseName = "correction identity and native entries preserve lazy guards";
    for (unsigned variant=0; variant<4; ++variant) {
        correction = prepareCorrection(1'000'000,500'000u);
        auto &entry = g_framegenHistory.pending.front();
        if (variant==0u) entry.bReal = true;
        if (variant==1u) entry.ulAnchorRealFrameId = 41u;
        if (variant==2u) entry.bProvisional = false;
        if (variant==3u) entry.oTargetVBlankNs.reset();
        CHECK(runCorrection(correction) == (variant==3u ? 1u : 0u));
        CHECK(completionQueries == 0u);
        CHECK(timer.queries == 1u); // no raw query for identity skips/native discard
    }
}

static void recordedNegativeCases()
{
    caseName = "22 original negative bias correction observations";
#include "recorded.inc"
    CHECK(std::size(lost) == 22u);
    for (const auto &d : lost) {
        const int64_t bias = static_cast<int64_t>(d.biasedWakeNs)-static_cast<int64_t>(d.rawWakeNs);
        auto correction = prepareCorrection(bias,d.costNs);
        g_framegenHistory.ulDeadlineGridIntervalNs = 8'333'333u;
        clockNs = R+d.arrivalNs;
        timer.first = timer.fresh = {R+d.rawTargetNs,R+d.rawWakeNs};
        g_SteamCompMgrVBlankTime.schedule.ulTargetVBlank = R+d.rawTargetNs-8'333'333u;
        g_framegenHistory.pending.front().oTargetVBlankNs = R+d.rawTargetNs;
        CHECK(runCorrection(correction) == 0u);
        CHECK(g_framegenHistory.pending.size() == 1u);
        CHECK(completionQueries == 0u);
    }
}

int main()
{
    positiveAndBoundaryCases();
    rawObservationAndStartCases();
    ladderCases();
    nativeCases();
    realEligibilityCases();
    guardAndLearningCases();
    correctionCases();
    recordedNegativeCases();
    std::printf("CPU production causal fragments: %u checks, %u failures\n",checks,failures);
    return failures ? 1 : 0;
}
