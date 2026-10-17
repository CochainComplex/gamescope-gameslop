// A real private Wayland connection using the production server handlers.
// No Vulkan instance, EGL context, display socket, GPU or NPU is required.
#include "WaylandServer/GpuDiagnostics.h"
#include "control-client.h"
#include <cassert>
#include <csignal>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace gamescope::framegen;
struct Client {
    gamescope_gpu_diagnostics_v1 *protocol=nullptr;
    GpuDiagnostics result;
    bool done=false;
};
static const gamescope_gpu_diagnostics_v1_listener listener{
    [](void *data,gamescope_gpu_diagnostics_v1 *,uint32_t index,uint32_t vendor,uint32_t device,uint32_t software,
       const char *name,const char *pci,const char *node,const char *driver) {
        auto &client=*static_cast<Client *>(data);
        assert(index==client.result.cards.size());
        client.result.cards.push_back({name,pci,node,driver,vendor,device,software!=0});
    },
    [](void *data,gamescope_gpu_diagnostics_v1 *,uint32_t presenter,uint32_t renderer,const char *evidence,
       const char *transport,const char *scope,const char *build) {
        auto &value=static_cast<Client *>(data)->result;
        value.presenter=presenter; value.renderer=renderer; value.evidence=evidence;
        value.transport=transport; value.scope=scope; value.build=build;
    },
    [](void *data,gamescope_gpu_diagnostics_v1 *) { static_cast<Client *>(data)->done=true; },
};
static const wl_registry_listener registryListener{
    [](void *data,wl_registry *registry,uint32_t id,const char *name,uint32_t) {
        if (std::strcmp(name,"gamescope_gpu_diagnostics_v1")) return;
        auto &client=*static_cast<Client *>(data);
        client.protocol=static_cast<gamescope_gpu_diagnostics_v1 *>(
            wl_registry_bind(registry,id,&gamescope_gpu_diagnostics_v1_interface,1));
        gamescope_gpu_diagnostics_v1_add_listener(client.protocol,&listener,&client);
    },
    [](void *,wl_registry *,uint32_t) {},
};
int main(int argc, char **argv) {
    // The global is opt-in: present only when GAMESCOPE_GPU_DIAGNOSTICS is on.
    assert(argc==2 && (!std::strcmp(argv[1],"--expect-present") || !std::strcmp(argv[1],"--expect-absent")));
    const bool expectPresent=!std::strcmp(argv[1],"--expect-present");
    alarm(15);
    for (int scenario=0;scenario<4;++scenario) {
        GpuDiagnostics inventory;
        inventory.cards={{"AMD compositor","0000:66:00.0","/dev/dri/renderD128","RADV",0x1002,0x150e,false},
            {"NVIDIA candidate","0000:01:00.0","/dev/dri/renderD129","NVIDIA",0x10de,0x2db9,false}};
        inventory.presenter=0; inventory.scope="Nested: outer presenter unknown"; inventory.build="test+abc123+";
        if (scenario==1) inventory.cards.push_back(inventory.cards.back());
        gpu_diagnostics_store().initialize(inventory);
        gpu_diagnostics_store().observe(scenario==2 ? "linear" : "NVIDIA",scenario!=2);
        int sockets[2]; assert(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,sockets)==0);
        const pid_t child=fork(); assert(child>=0);
        if (child==0) {
            close(sockets[1]);
            auto *server=wl_display_create(); assert(server);
            if (scenario!=3) {
                const bool created=gamescope::create_gpu_diagnostics_global_if_requested(
                    server,std::getenv("GAMESCOPE_GPU_DIAGNOSTICS"));
                assert(created==expectPresent);
            }
            assert(wl_client_create(server,sockets[0]));
            wl_display_run(server);
            _exit(0);
        }
        close(sockets[0]);
        auto *display=wl_display_connect_to_fd(sockets[1]); assert(display);
        auto *registry=wl_display_get_registry(display);
        Client client;
        wl_registry_add_listener(registry,&registryListener,&client);
        assert(wl_display_roundtrip(display)>=0);
        // Older hosts and hosts without the opt-in: no global, no invented data.
        if (scenario==3 || !expectPresent) assert(!client.protocol);
        else {
            assert(client.protocol);
            for (int refresh=0;refresh<2;++refresh) {
                client.result={}; client.done=false;
                gamescope_gpu_diagnostics_v1_get_snapshot(client.protocol);
                assert(wl_display_roundtrip(display)>=0);
                assert(client.done);
                assert(client.result.cards.size()==inventory.cards.size());
                assert(client.result.presenter==0);
                assert(client.result.cards[0].pci=="0000:66:00.0");
                assert(client.result.cards[1].node=="/dev/dri/renderD129");
                assert(client.result.cards[1].driver=="NVIDIA");
                assert(client.result.cards[1].vendor==0x10de);
                assert(client.result.build=="test+abc123+");
                assert(client.result.scope==inventory.scope);
                assert(client.result.renderer==(scenario==0 ? 1u : kUnknownGpu));
                assert(client.result.evidence.find(scenario==0 ? "Inferred" : scenario==1 ? "Ambiguous" : "Unknown")!=std::string::npos);
            }
            gamescope_gpu_diagnostics_v1_destroy(client.protocol);
            assert(wl_display_roundtrip(display)>=0);
        }
        wl_registry_destroy(registry); wl_display_disconnect(display);
        kill(child,SIGTERM); assert(waitpid(child,nullptr,0)==child);
    }
    if (expectPresent)
        std::puts("GPU diagnostics Wayland IPC: inventory, inferred/ambiguous/unknown, refresh, destroy and older host PASS");
    else
        std::puts("GPU diagnostics Wayland IPC: global absent without GAMESCOPE_GPU_DIAGNOSTICS PASS");
}
