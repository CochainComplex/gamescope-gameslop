// CPU owner fixture: scheduling bodies come unchanged from steamcompmgr.cpp.
// External focus/window/connector state is controlled here, not Wayland/GPU state.
#include "Utils/VBlankSchedule.h"
#include "refresh_rate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

static unsigned checks = 0, failures = 0;
static const char *caseName = "setup";
static void check(bool value, const char *expression, int line)
{
    ++checks;
    if (!value) {
        ++failures;
        std::fprintf(stderr,"FAIL %s line%d: %s\n",caseName,line,expression);
    }
}
#define CHECK(value) check(static_cast<bool>(value), #value, __LINE__)

namespace gamescope { template<class T> using Rc = std::shared_ptr<T>; }
struct commit_t {
    uint64_t commitID = 0, win_seq = 0;
    bool fifo = true, done = true;
};
struct steamcompmgr_win_t {
    uint64_t seq = 7u, last_commit_first_latch_time = 0;
    bool exempt = false, vrApp = false;
    std::vector<gamescope::Rc<commit_t>> commit_queue;
};
enum HeldCommitTypes_t { HELD_COMMIT_BASE, HELD_COMMIT_FADE, HELD_COMMIT_COUNT };
struct global_focus_t {
    steamcompmgr_win_t *focusWindow = nullptr, *overrideWindow = nullptr, *fadeWindow = nullptr;
    bool bPendingFade = false;
    uint64_t uFadeOutStartTime = 0;
    std::array<gamescope::Rc<commit_t>, HELD_COMMIT_COUNT> HeldCommits;
};
static steamcompmgr_win_t window, overrideWindow;
static global_focus_t focus;
static bool focusAvailable = true;
static global_focus_t *GetCurrentFocus() { return focusAvailable ? &focus : nullptr; }
static bool window_is_limiter_exempt(steamcompmgr_win_t *w) { return !w || w->exempt; }
static bool window_is_vr_app(steamcompmgr_win_t *w) { return w && w->vrApp; }
static struct Connector {
    bool vrr = false;
    bool IsVRRActive() const { return vrr; }
} connector;
static struct Backend {
    bool native = false, connected = true;
    bool OwnsKMSPresentTiming() const { return native; }
    Connector *GetCurrentConnector() const { return connected ? &connector : nullptr; }
} backend;
static bool backendAvailable = true;
static Backend *GetBackend() { return backendAvailable ? &backend : nullptr; }
static int g_nNestedRefresh = 60'000, g_nOutputRefresh = 60'000;
static int g_nSteamCompMgrTargetFPS = 30;
static uint64_t g_SteamCompMgrAppRefreshCycle = gamescope::mHzToRefreshCycle(60'000);
static uint64_t g_SteamCompMgrLimitedAppRefreshCycle = gamescope::mHzToRefreshCycle(30'000);
#include "state.inc"
#include "limit.inc"
#include "gate.inc"
#include "note.inc"
#include "owner.inc"

static constexpr uint64_t rawStart = 1'000'000'000u;
static uint64_t period() { return gamescope::mHzToRefreshCycle(g_nNestedRefresh ? g_nNestedRefresh : g_nOutputRefresh); }
static gamescope::Rc<commit_t> commit(uint64_t id, bool fifo = true, uint64_t seq = 7u)
{
    return std::make_shared<commit_t>(commit_t{id,seq,fifo,true});
}
static void setup()
{
    window = {};
    overrideWindow = {};
    overrideWindow.seq = 8u;
    focus = {};
    focus.focusWindow = &window;
    focusAvailable = backendAvailable = true;
    backend = {};
    connector = {};
    g_nNestedRefresh = g_nOutputRefresh = 60'000;
    g_nSteamCompMgrTargetFPS = 30;
    g_SteamCompMgrAppRefreshCycle = period();
    g_SteamCompMgrLimitedAppRefreshCycle = gamescope::mHzToRefreshCycle(30'000);
    s_ulVBlankIdx = s_ulPrevVBlankIdx = s_ulLastPassVBlank = 0;
    s_ulLastPassRefreshCycle = 0;
    s_oLowestFPSLimitScheduleVRR.reset();
    auto held = commit(100u);
    focus.HeldCommits[HELD_COMMIT_BASE] = held;
    window.commit_queue = {held};
}

// Tests below call only the extracted owner/gate and actual VBlank helpers.
static void sharedPolicyCases()
{
    caseName = "shared fixed limiter divisor retains integer policy";
    CHECK(gamescope::VBlank::LimiterDivisor(60,30,true) == 2);
    CHECK(gamescope::VBlank::LimiterDivisor(144,60,true) == 2);
    CHECK(gamescope::VBlank::LimiterDivisor(60,59,true) == 1);
    CHECK(gamescope::VBlank::LimiterDivisor(60,60,true) == 0);
    CHECK(gamescope::VBlank::LimiterDivisor(60,0,true) == 0);
    CHECK(gamescope::VBlank::LimiterDivisor(0,30,true) == 0);
    CHECK(gamescope::VBlank::LimiterDivisor(60,30,false) == 0);
    CHECK(gamescope::ConvertmHzToHz(143'990) == 144);
    CHECK(gamescope::ConvertmHzToHz(60'001) == 60);

    caseName = "next limiter target rejects unknown and overflow";
    constexpr uint64_t T = 10u;
    constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
    CHECK(gamescope::VBlank::NextLimiterTarget(0u,100u,T,2u) == 120u);
    CHECK(gamescope::VBlank::NextLimiterTarget(1u,100u,T,2u) == 110u);
    CHECK(gamescope::VBlank::NextLimiterTarget(2u,100u,T,2u) == 120u);
    CHECK(gamescope::VBlank::NextLimiterTarget(1u,0u,T,2u) == 0u);
    CHECK(gamescope::VBlank::NextLimiterTarget(1u,100u,0u,2u) == 0u);
    CHECK(gamescope::VBlank::NextLimiterTarget(1u,100u,T,0u) == 0u);
    CHECK(gamescope::VBlank::NextLimiterTarget(maximum,100u,T,2u) == 0u);
    CHECK(gamescope::VBlank::NextLimiterTarget(0u,maximum-T,T,2u) == 0u);
    CHECK(gamescope::VBlank::NextLimiterTarget(0u,maximum-2u*T,T,2u) == maximum);
}

static void fixedGateCases()
{
    caseName = "actual fixed gate compares processed divisor buckets";
    setup();
    const struct { uint64_t previous, current; bool opens; } cases[] = {
        {0u,1u,false}, {1u,2u,true}, {2u,2u,false},
        {2u,3u,false}, {2u,4u,true}, {1u,5u,true},
    };
    for (const auto &row : cases) {
        s_ulPrevVBlankIdx = row.previous;
        CHECK(steamcompmgr_should_vblank_window(true,row.current,&window,0u) == row.opens);
    }
    s_ulPrevVBlankIdx = 2u;
    CHECK(steamcompmgr_should_vblank_window(false,3u,&window,0u));
    g_nSteamCompMgrTargetFPS = 0;
    CHECK(steamcompmgr_should_vblank_window(true,3u,&window,0u));
    g_nSteamCompMgrTargetFPS = 60;
    CHECK(steamcompmgr_should_vblank_window(true,3u,&window,0u));
    g_nSteamCompMgrTargetFPS = 30;
    g_nNestedRefresh = 0;
    g_nOutputRefresh = 144'000;
    s_ulPrevVBlankIdx = 4u;
    CHECK(!steamcompmgr_should_vblank_window(true,7u,&window,0u));
    CHECK(steamcompmgr_should_vblank_window(true,8u,&window,0u));
}

static void processed(uint64_t index = 2u, uint64_t previous = 1u)
{
    s_ulVBlankIdx = index;
    s_ulPrevVBlankIdx = previous;
    s_ulLastPassVBlank = rawStart;
    s_ulLastPassRefreshCycle = period();
}
static uint64_t eligible()
{
    return steamcompmgr_framegen_next_real_eligible_vblank(100u,period());
}
static auto state()
{
    return std::tuple(s_ulVBlankIdx,s_ulPrevVBlankIdx,s_ulLastPassVBlank,
        s_ulLastPassRefreshCycle,s_oLowestFPSLimitScheduleVRR,
        window.last_commit_first_latch_time,overrideWindow.last_commit_first_latch_time,
        window.commit_queue.size(),overrideWindow.commit_queue.size());
}

static void ownerPhaseCases()
{
    caseName = "owner predicts next fixed bucket using processed phase";
    setup();
    CHECK(eligible() == 0u); // no processed pass yet
    processed(2u,1u);
    CHECK(eligible() == rawStart+2u*period());
    processed(3u,2u);
    CHECK(eligible() == rawStart+period());
    processed(4u,1u); // skipped passes preserve display-index phase
    CHECK(eligible() == rawStart+2u*period());
    processed(2u,2u); // repeated pass does not invent a newly open bucket
    CHECK(eligible() == rawStart+2u*period());
    const auto before = state();
    CHECK(eligible() == eligible());
    CHECK(state() == before);

    caseName = "live cap changes recompute eligibility without cached divisor";
    setup();
    processed();
    g_nSteamCompMgrTargetFPS = 20;
    CHECK(eligible() == rawStart+period());
    CHECK(!steamcompmgr_should_vblank_window(true,2u,&window,0u));
    g_nSteamCompMgrTargetFPS = 30;
    CHECK(eligible() == rawStart+2u*period());
    CHECK(steamcompmgr_should_vblank_window(true,2u,&window,0u));
    g_nSteamCompMgrTargetFPS = 40; // existing integer divisor is one
    CHECK(eligible() == rawStart+period());
    CHECK(steamcompmgr_should_vblank_window(true,2u,&window,0u));
    g_nSteamCompMgrTargetFPS = 0;
    CHECK(eligible() == 0u);
    g_nSteamCompMgrTargetFPS = 60;
    CHECK(eligible() == 0u);
    g_nSteamCompMgrTargetFPS = 120;
    CHECK(eligible() == 0u);
}

static void passCases()
{
    caseName = "actual pass owner preserves same skipped and repeated phase";
    setup();
    processed(10u,9u);
    const uint64_t T = period();
    steamcompmgr_note_vblank_pass(rawStart,T);
    CHECK(s_ulVBlankIdx == 10u && s_ulPrevVBlankIdx == 10u);
    CHECK(s_ulLastPassVBlank == rawStart);
    CHECK(eligible() == rawStart+2u*T);
    steamcompmgr_note_vblank_pass(rawStart+T,T);
    CHECK(s_ulVBlankIdx == 11u && s_ulPrevVBlankIdx == 10u);
    CHECK(s_ulLastPassVBlank == rawStart+T);
    CHECK(eligible() == rawStart+2u*T);
    CHECK(!steamcompmgr_should_vblank_window(true,s_ulVBlankIdx,&window,0u));
    steamcompmgr_note_vblank_pass(rawStart+5u*T,T);
    CHECK(s_ulVBlankIdx == 15u && s_ulPrevVBlankIdx == 11u);
    CHECK(s_ulLastPassVBlank == rawStart+5u*T);
    CHECK(eligible() == rawStart+6u*T);
    CHECK(steamcompmgr_should_vblank_window(true,s_ulVBlankIdx,&window,0u));
    steamcompmgr_note_vblank_pass(rawStart+5u*T,T);
    CHECK(s_ulVBlankIdx == 15u && s_ulPrevVBlankIdx == 15u);
    CHECK(!steamcompmgr_should_vblank_window(true,s_ulVBlankIdx,&window,0u));
    CHECK(eligible() == rawStart+6u*T);

    caseName = "processed period changes invalidate old configuration provenance";
    const uint64_t changedT = gamescope::mHzToRefreshCycle(90'000);
    steamcompmgr_note_vblank_pass(rawStart+6u*T,changedT);
    CHECK(s_ulLastPassRefreshCycle == changedT);
    CHECK(eligible() == 0u);
    g_nNestedRefresh = 90'000;
    g_SteamCompMgrAppRefreshCycle = changedT;
    CHECK(steamcompmgr_framegen_next_real_eligible_vblank(100u,changedT) > s_ulLastPassVBlank);
}

static void ownerGuardCases()
{
    caseName = "owner rejects unknown ownership identity fade and nonfixed paths";
    for (unsigned variant=0; variant<12; ++variant) {
        setup();
        processed();
        if (variant==0u) focusAvailable = false;
        if (variant==1u) focus.HeldCommits[HELD_COMMIT_BASE].reset();
        if (variant==2u) focus.HeldCommits[HELD_COMMIT_BASE]->commitID = 99u;
        if (variant==3u) focus.HeldCommits[HELD_COMMIT_BASE]->fifo = false;
        if (variant==4u) window.seq = 9u;
        if (variant==5u) focus.focusWindow = nullptr;
        if (variant==6u) focus.fadeWindow = &window;
        if (variant==7u) focus.bPendingFade = true;
        if (variant==8u) backendAvailable = false;
        if (variant==9u) backend.native = true;
        if (variant==10u) connector.vrr = true;
        if (variant==11u) window.exempt = true;
        const auto before = state();
        CHECK(eligible() == 0u);
        CHECK(state() == before);
    }
    setup();
    processed();
    window.vrApp = true;
    CHECK(eligible() == 0u);
    CHECK(!steamcompmgr_window_should_limit_fps(&window));
    CHECK(steamcompmgr_framegen_next_real_eligible_vblank(0u,period()) == 0u);

    caseName = "owner requires argument app configured and processed period agreement";
    setup();
    processed();
    CHECK(steamcompmgr_framegen_next_real_eligible_vblank(100u,0u) == 0u);
    CHECK(steamcompmgr_framegen_next_real_eligible_vblank(100u,period()+1u) == 0u);
    ++g_SteamCompMgrAppRefreshCycle;
    CHECK(eligible() == 0u);
    g_SteamCompMgrAppRefreshCycle = period();
    ++s_ulLastPassRefreshCycle;
    CHECK(eligible() == 0u);
    s_ulLastPassRefreshCycle = period();
    g_nNestedRefresh = 90'000;
    g_SteamCompMgrAppRefreshCycle = period();
    CHECK(eligible() == 0u); // live config changed before a new processed pass
    setup();
    processed();
    g_nNestedRefresh = 0;
    CHECK(eligible() == rawStart+2u*period());
    g_nOutputRefresh = 90'000;
    CHECK(eligible() == 0u);
    setup();
    processed();
    const uint64_t T = period();
    g_nNestedRefresh = g_nOutputRefresh = 0;
    CHECK(steamcompmgr_framegen_next_real_eligible_vblank(100u,T) == 0u);
}

static void overrideCases()
{
    caseName = "held sequence resolves exact focus or override window";
    setup();
    processed();
    auto held = focus.HeldCommits[HELD_COMMIT_BASE];
    focus.overrideWindow = &overrideWindow;
    held->win_seq = overrideWindow.seq;
    overrideWindow.commit_queue = {held};
    window.commit_queue.clear();
    window.exempt = true;
    CHECK(eligible() == rawStart+2u*period());
    overrideWindow.exempt = true;
    CHECK(eligible() == 0u);
    overrideWindow.exempt = false;
    overrideWindow.seq = 99u;
    CHECK(eligible() == 0u);
    focus.overrideWindow = nullptr;
    CHECK(eligible() == 0u);

    setup();
    processed();
    focus.overrideWindow = &overrideWindow;
    overrideWindow.exempt = true;
    CHECK(eligible() == rawStart+2u*period()); // unrelated override is not the owner
}

static void queueCases()
{
    caseName = "owner queue accepts held tail or one strictly newer FIFO only";
    const struct { unsigned shape; bool accepted; } cases[] = {
        {0u,true}, {1u,true}, {2u,false}, {3u,false}, {4u,false},
        {5u,false}, {6u,false}, {7u,false}, {8u,true}, {9u,true},
        {10u,false}, {11u,false}, {12u,false},
    };
    for (const auto &row : cases) {
        setup();
        processed();
        auto held = focus.HeldCommits[HELD_COMMIT_BASE];
        auto newer = commit(101u);
        switch (row.shape) {
            case 0: window.commit_queue = {held}; break;
            case 1: window.commit_queue = {held,newer}; break;
            case 2: window.commit_queue.clear(); break;
            case 3: window.commit_queue = {held,commit(101u,false)}; break;
            case 4: window.commit_queue = {held,newer,commit(102u)}; break;
            case 5: window.commit_queue = {held,commit(101u,false),commit(102u)}; break;
            case 6: window.commit_queue = {held,commit(99u)}; break;
            case 7: window.commit_queue = {nullptr,newer}; break;
            case 8: window.commit_queue = {commit(99u,false),held}; break;
            case 9: window.commit_queue = {commit(98u),held,newer}; break;
            case 10: window.commit_queue = {held,nullptr}; break;
            case 11: window.commit_queue = {newer}; break;
            case 12: window.commit_queue = {commit(99u)}; break;
        }
        const auto before = state();
        CHECK(eligible() == (row.accepted ? rawStart+2u*period() : 0u));
        CHECK(state() == before);
    }
}

static void ownerOverflowCases()
{
    caseName = "owner exposes no unrepresentable target or index";
    setup();
    processed();
    s_ulLastPassVBlank = 0;
    CHECK(eligible() == 0u);
    processed();
    s_ulLastPassVBlank = std::numeric_limits<uint64_t>::max()-period();
    CHECK(eligible() == 0u);
    processed(std::numeric_limits<uint64_t>::max(),0u);
    CHECK(eligible() == 0u);
}

static void vrrCases()
{
    caseName = "VRR gate retains schedule mutation while owner stays read only";
    setup();
    processed();
    connector.vrr = true;
    window.last_commit_first_latch_time = rawStart;
    const uint64_t scheduled = rawStart+g_SteamCompMgrLimitedAppRefreshCycle;
    const auto before = state();
    CHECK(eligible() == 0u);
    CHECK(state() == before);
    CHECK(!steamcompmgr_should_vblank_window(true,2u,&window,rawStart));
    CHECK(s_oLowestFPSLimitScheduleVRR == std::optional<uint64_t>(scheduled));
    overrideWindow.last_commit_first_latch_time = rawStart-1'000'000u;
    CHECK(!steamcompmgr_should_vblank_window(true,2u,&overrideWindow,rawStart));
    CHECK(s_oLowestFPSLimitScheduleVRR == std::optional<uint64_t>(scheduled-1'000'000u));
    const auto mutated = state();
    CHECK(eligible() == 0u);
    CHECK(state() == mutated);
    CHECK(steamcompmgr_should_vblank_window(true,2u,&window,scheduled-k_ulVRRScheduleFudge));
    CHECK(state() == mutated);
    CHECK(steamcompmgr_should_vblank_window(false,2u,&window,rawStart));
    CHECK(steamcompmgr_should_vblank_window(true,2u,nullptr,rawStart));
    g_nSteamCompMgrTargetFPS = 59;
    CHECK(steamcompmgr_should_vblank_window(true,2u,&window,rawStart));
    CHECK(state() == mutated);
}

int main()
{
    sharedPolicyCases();
    fixedGateCases();
    ownerPhaseCases();
    passCases();
    ownerGuardCases();
    overrideCases();
    queueCases();
    ownerOverflowCases();
    vrrCases();
    std::printf("CPU production real-admission owner: %u checks, %u failures\n",checks,failures);
    return failures ? 1 : 0;
}
