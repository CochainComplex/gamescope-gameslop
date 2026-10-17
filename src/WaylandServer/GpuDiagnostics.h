#pragma once
#include <algorithm>
#include <wayland-server.h>
#include "gamescope-control-protocol.h"
#include "framegen/gpu_diagnostics.hpp"
#include "framegen/settings.hpp"

// Read-only CPU snapshot transport; independent of renderer and backend APIs.
namespace gamescope {
static void gpu_diagnostics_destroy(wl_client *, wl_resource *resource) { wl_resource_destroy(resource); }
static void gpu_diagnostics_snapshot(wl_client *, wl_resource *resource)
{
	const auto snapshot=gamescope::framegen::gpu_diagnostics_store().snapshot();
	for (uint32_t i=0; i<snapshot.cards.size(); ++i) {
		const auto &card=snapshot.cards[i];
		gamescope_gpu_diagnostics_v1_send_device(resource,i,card.vendor,card.device,card.software,
			card.name.c_str(),card.pci.c_str(),card.node.c_str(),card.driver.c_str());
	}
	gamescope_gpu_diagnostics_v1_send_route(resource,snapshot.presenter,snapshot.renderer,
		snapshot.evidence.c_str(),snapshot.transport.c_str(),snapshot.scope.c_str(),snapshot.build.c_str());
	gamescope_gpu_diagnostics_v1_send_done(resource);
}
static const struct gamescope_gpu_diagnostics_v1_interface gpu_diagnostics_impl={
	.destroy=gpu_diagnostics_destroy, .get_snapshot=gpu_diagnostics_snapshot,
};
static void gpu_diagnostics_bind(wl_client *client, void *, uint32_t version, uint32_t id)
{
	// Honour the client's requested version, capped at the one this host speaks.
	auto *resource=wl_resource_create(client,&gamescope_gpu_diagnostics_v1_interface,std::min(version,1u),id);
	if (!resource) { wl_client_post_no_memory(client); return; }
	wl_resource_set_implementation(resource,&gpu_diagnostics_impl,nullptr,nullptr);
}

inline void create_gpu_diagnostics_global(wl_display *display) {
    wl_global_create(display,&gamescope_gpu_diagnostics_v1_interface,1,nullptr,gpu_diagnostics_bind);
}

// Opt-in: the global is visible to every Wayland client (games included) and
// names GPUs, PCI addresses, render nodes, drivers and the build. It is
// advertised only when GAMESCOPE_GPU_DIAGNOSTICS is on (1/true/yes/on); unset,
// off or an invalid value keeps it hidden. Returns whether it was created.
inline bool create_gpu_diagnostics_global_if_requested(wl_display *display, const char *value) {
    if (!gamescope::framegen::parse_bool_setting(value).value_or(false)) return false;
    create_gpu_diagnostics_global(display);
    return true;
}
} // namespace gamescope
