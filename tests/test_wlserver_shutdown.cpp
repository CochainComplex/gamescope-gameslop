// External Wayland/input dependencies are fixtures; loop and stop are production code.
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <mutex>
#include <poll.h>
#include <pthread.h>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std::chrono_literals;

static void require(bool condition, const char *message)
{
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

struct Listener { int link; };
struct Entry { int present_timing; };
struct FakeServer
{
    void *display = this;
    void *event_loop = this;
    std::atomic<bool> bWaylandServerRunning{false};
    bool physical_cursor_move = false;
    std::mutex xdg_commit_lock;
    std::vector<Entry> xdg_commit_queue;
    std::vector<int> xdg_wins;
    struct { std::vector<int> xwayland_servers; } wlr;
    Listener new_pointer_constraint, new_xdg_surface, new_xdg_toplevel, new_layer_shell_surface;
} wlserver;
static Listener new_surface_listener, new_input_listener;
std::mutex g_SteamCompMgrXWaylandServerMutex, g_wlserver_xdg_shell_windows_lock;
static std::mutex waylock;
static int event_pipe[2];
static std::string_view test_case;
static std::atomic<bool> callback_entered{false}, release_callback{false}, backend_alive{true};
static std::atomic<bool> requested{false};
struct { template<class... T> void errorf(T...) {} template<class... T> void errorf_errno(T...) {} } wl_log;

void wlserver_lock() { waylock.lock(); }
void wlserver_unlock(bool = true) { waylock.unlock(); }
void ShutdownGamescope() { requested = true; }
[[noreturn]] void gamescope_exit(int status) { std::exit(status); }
void wlserver_shutdown_input() {}
void wlserver_present_timing_discard(int &) {}
void wl_display_flush_clients(void *) {}
void wl_display_destroy_clients(void *) {}
void wl_display_destroy(void *) {}
void wl_list_remove(int *) {}
int wl_event_loop_get_fd(void *) { return event_pipe[0]; }

static void blocked_callback()
{
    callback_entered = true;
    callback_entered.notify_all();
    release_callback.wait(false);
    require(backend_alive, "backend deleted during callback");
}

namespace gamescope { enum class InputType { Mouse }; }
struct Backend
{
    void NotifyPhysicalInput(gamescope::InputType)
    {
        require(backend_alive, "notification after backend deletion");
        if (test_case == "notification") blocked_callback();
    }
} backend;
Backend *GetBackend() { return backend_alive ? &backend : nullptr; }

int wl_event_loop_dispatch(void *, int)
{
    char byte;
    require(read(event_pipe[0], &byte, 1) == 1, "event fixture read failed");
    if (test_case == "error") return -1;
    if (test_case == "dispatch") blocked_callback();
    wlserver.physical_cursor_move = true;
    return 0;
}

#include "wlserver_loop.inc"

int main(int argc, char **argv)
{
    require(argc == 2, "expected case");
    test_case = argv[1];
    require(pipe2(event_pipe, O_CLOEXEC) == 0, "event pipe failed");
    // As in steamcompmgr_exit: held throughout stop and backend destruction.
    std::unique_lock compositor_guard(g_SteamCompMgrXWaylandServerMutex);
    std::thread server;
    if (test_case != "startup")
    {
        server = std::thread(wlserver_run);
        wlserver.bWaylandServerRunning.wait(false);
    }
    if (test_case == "dispatch" || test_case == "notification" || test_case == "error")
    {
        require(write(event_pipe[1], "x", 1) == 1, "event fixture write failed");
        if (test_case == "error")
            wlserver.bWaylandServerRunning.wait(true);
        else
            callback_entered.wait(false);
    }
    auto stop = std::async(std::launch::async, wlserver_shutdown);
    if (test_case == "startup" || test_case == "dispatch" || test_case == "notification")
    {
        require(stop.wait_for(50ms) == std::future_status::timeout,
                "stop returned before dispatch finished");
        if (test_case == "startup")
            server = std::thread(wlserver_run);
        else
        {
            release_callback = true;
            release_callback.notify_all();
        }
    }
    require(stop.wait_for(2s) == std::future_status::ready,
            "stop deadlocked on Wayland lock, idle poll or Xwayland mutex");
    stop.get();
    require(g_bWaylandDispatchStopped, "missing stop acknowledgement");
    if (test_case == "error") require(requested, "loop error did not request compositor exit");
    backend_alive = false;
    compositor_guard.unlock();
    server.join();
    require(wlserver.display == nullptr, "final Wayland teardown did not finish");
    return 0;
}
