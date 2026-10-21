// Wayland resources are fixtures; focus publication and hotkey handling are production code.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <string_view>
#include <linux/input-event-codes.h>

static void require(bool condition, const char *message)
{
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

// Use the production waiter, including its constructor and Shutdown before any
// input Init. Only its logger is a fixture.
struct LogScope
{
    template<class... T> void errorf(T...) { require(false, "waiter setup failed"); }
    template<class... T> void errorf_errno(T...) { require(false, "waiter operation failed"); }
} g_WaitableLog;
#include "wayland_fullscreen_waiter.inc"

static std::atomic<bool> worker_finished{false};
struct WorkerFinished
{
    ~WorkerFinished() { worker_finished = true; worker_finished.notify_all(); }
};

// Force the worker to finish its production startup prefix before the destructor
// can continue beyond notify_all. Wake-before-stop must reach the backend read.
struct StartupGate
{
    std::atomic<bool> value{false};
    void operator=(bool initialized) { value = initialized; }
    void wait(bool previous) { value.wait(previous); }
    void notify_all() { value.notify_all(); worker_finished.wait(false); }
};

// Pause the production hotkey after it has pinned the request flag, before its
// write. The real global is bool; this conversion supplies the scheduling seam.
struct FullscreenValue
{
    bool value = false;
    bool block = false;
    std::atomic<bool> entered{false}, release{false};
    unsigned reads = 0;

    operator bool()
    {
        ++reads;
        if (block)
        {
            entered = true;
            entered.notify_all();
            release.wait(false);
        }
        return value;
    }
} g_bFullscreen;

enum class GamescopeUpscaleFilter { PIXEL, LINEAR, FSR, NIS };
static GamescopeUpscaleFilter g_wantedUpscaleFilter;
static int g_upscaleFilterSharpness;
static bool fail_connector_init = false;
static unsigned destroyed_connectors = 0;

namespace gamescope
{
    struct CScreenshotManager
    {
        static CScreenshotManager &Get() { static CScreenshotManager manager; return manager; }
        void TakeScreenshot(bool) {}
    };

    struct IBackendConnector
    {
        virtual ~IBackendConnector() { ++destroyed_connectors; }
    };

    class CWaylandBackend;
    class CWaylandConnector : public IBackendConnector
    {
    public:
        CWaylandConnector(CWaylandBackend *backend, uint64_t) : m_pBackend(backend) {}
        ~CWaylandConnector() override;
        bool Init() { return !fail_connector_init; }
        std::weak_ptr<std::atomic<bool>> GetFullscreenState() const;
    private:
        CWaylandBackend *m_pBackend;
#include "wayland_fullscreen_connector_state.inc"
    };

    class CWaylandInputThread
    {
    public:
        CWaylandInputThread();
        ~CWaylandInputThread();
        void ThreadFunc();
        void SetFullscreenState(std::weak_ptr<std::atomic<bool>> pState);
        void RunHotkey(uint32_t uKey);
    private:
        CWaylandBackend *m_pBackend = nullptr;
        CWaiter<4> m_Waiter;
        std::thread m_Thread;
        StartupGate m_bInitted;
#include "wayland_fullscreen_input_state.inc"
    };

    class CWaylandBackend
    {
    public:
        std::shared_ptr<IBackendConnector> CreateVirtualConnector(uint64_t ulVirtualConnectorKey);
#include "wayland_fullscreen_focus.inc"
        CWaylandInputThread m_InputThread;
#include "wayland_fullscreen_connector_destroyed.inc"
    };

// Fail at the first backend-read statement without evaluating a null backend.
#define wl_display_get_fd(...) (require(false, "worker read backend before input Init"), -1)
#include "wayland_fullscreen_methods.inc"
#undef wl_display_get_fd
}

static std::weak_ptr<std::atomic<bool>> state(const std::shared_ptr<gamescope::IBackendConnector> &connector)
{
    return static_cast<gamescope::CWaylandConnector *>(connector.get())->GetFullscreenState();
}

int main(int argc, char **argv)
{
    require(argc == 2, "expected case");
    const std::string_view test_case = argv[1];
    if (test_case == "startup")
    {
        { gamescope::CWaylandInputThread input; }
        require(worker_finished, "uninitialized input worker was not joined");
        return 0;
    }
    gamescope::CWaylandBackend backend;
    auto owner = backend.CreateVirtualConnector(0);
    require(owner != nullptr, "connector creation failed");
    auto original = state(owner);

    if (test_case == "current")
    {
        backend.m_InputThread.RunHotkey(KEY_F);
        require(original.lock()->load(), "current connector did not enter fullscreen");
        g_bFullscreen.value = true;
        backend.m_InputThread.RunHotkey(KEY_F);
        require(!original.lock()->load(), "current connector did not leave fullscreen");
    }
    else if (test_case == "expired")
    {
        owner.reset();
        require(original.expired(), "input retained the removed connector state");
        require(backend.m_pFocusConnector == nullptr, "removed connector remains focused");
        const unsigned reads = g_bFullscreen.reads;
        backend.m_InputThread.RunHotkey(KEY_F);
        require(g_bFullscreen.reads == reads, "hotkey reached an expired target");
    }
    else if (test_case == "pinned" || test_case == "replacement")
    {
        g_bFullscreen.block = true;
        auto callback = std::async(std::launch::async, [&] { backend.m_InputThread.RunHotkey(KEY_F); });
        g_bFullscreen.entered.wait(false);
        owner.reset();
        require(destroyed_connectors == 1, "hotkey moved connector destruction onto the input thread");
        require(original.use_count() == 1, "hotkey did not retain exactly its request flag");
        auto pinned = original.lock();
        require(pinned != nullptr, "request flag died while the hotkey was using it");

        std::shared_ptr<gamescope::IBackendConnector> replacement;
        std::weak_ptr<std::atomic<bool>> next;
        if (test_case == "replacement")
        {
            replacement = backend.CreateVirtualConnector(1);
            require(replacement != nullptr, "replacement connector creation failed");
            next = state(replacement);
        }
        g_bFullscreen.release = true;
        g_bFullscreen.release.notify_all();
        callback.get();
        require(pinned->load(), "pinned hotkey did not finish its request");
        if (replacement)
        {
            require(!next.lock()->load(), "orphaned request changed the new connector");
            g_bFullscreen.block = false;
            backend.m_InputThread.RunHotkey(KEY_F);
            require(next.lock()->load(), "next hotkey did not reach the new connector");
        }
        pinned.reset();
        require(original.expired(), "orphaned request state leaked after the callback");
    }
    else if (test_case == "failed-replacement")
    {
        fail_connector_init = true;
        require(!backend.CreateVirtualConnector(1), "failed initialization returned a connector");
        require(backend.m_pFocusConnector == nullptr, "failed connector remains focused");
        const unsigned reads = g_bFullscreen.reads;
        backend.m_InputThread.RunHotkey(KEY_F);
        require(g_bFullscreen.reads == reads, "failed focus reused a previous target");
        require(!original.lock()->load(), "failed focus changed the old connector");
    }
    else if (test_case == "old-destroy")
    {
        auto replacement = backend.CreateVirtualConnector(1);
        require(replacement != nullptr, "replacement connector creation failed");
        auto next = state(replacement);
        owner.reset();
        require(original.expired(), "old connector state survived its owner");
        require(backend.m_pFocusConnector == replacement.get(), "old destructor cleared the new focus");
        backend.m_InputThread.RunHotkey(KEY_F);
        require(next.lock()->load(), "old destructor cleared the new hotkey target");
    }
    else
        require(false, "unknown case");

    return 0;
}
