#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include <utility>

namespace gamescope::framegen {
inline constexpr uint32_t kUnknownGpu = UINT32_MAX;
struct GpuCard {
    std::string name, pci, node, driver;
    uint32_t vendor = 0, device = 0;
    bool software = false;
};
struct GpuDiagnostics {
    std::vector<GpuCard> cards;
    uint32_t presenter = kUnknownGpu, renderer = kUnknownGpu;
    std::string evidence = "No composited client buffer observed";
    std::string transport = "not observed", scope, build;
};

// Buffer layout is evidence, never proof of a client's Vulkan device UUID.
inline uint32_t gpu_buffer_observation(const char *origin, bool staged) {
    uint32_t kind = 0;
    if (origin && *origin) {
        kind = !std::strcmp(origin,"cpu") ? 1 : !std::strcmp(origin,"linear") ? 2
             : !std::strcmp(origin,"AMD") ? 3 : !std::strcmp(origin,"NVIDIA") ? 4
             : !std::strcmp(origin,"Intel") ? 5 : 6;
    }
    return kind | (staged ? 256u : 0u);
}
inline GpuDiagnostics resolve_gpu_diagnostics(GpuDiagnostics result, uint32_t observation) {
    result.renderer = kUnknownGpu;
    const uint32_t kind = observation & 255u;
    const bool staged = (observation & 256u) != 0;
    result.transport = kind == 0 ? "not observed" : staged ? "staged / copied" : "imported / not staged";
    if (kind == 0) { result.evidence = "No composited client buffer observed"; return result; }
    if (kind == 1) { result.evidence = "CPU/shared-memory buffer; GPU producer unknown"; return result; }
    const uint32_t vendor = kind == 3 ? 0x1002 : kind == 4 ? 0x10de : kind == 5 ? 0x8086 : 0;
    uint32_t matches = 0, candidate = kUnknownGpu;
    for (uint32_t i=0; i<result.cards.size(); ++i) {
        const auto &card = result.cards[i];
        if (card.software) continue;
        if ((vendor && card.vendor == vendor) || (kind == 2 && staged && i != result.presenter)) {
            ++matches; candidate = i;
        }
    }
    if (matches == 1) {
        result.renderer = candidate;
        result.evidence = vendor ? "Inferred: modifier vendor, one matching GPU"
                                 : "Inferred: staged linear buffer, one other GPU";
    } else if (matches > 1) result.evidence = "Ambiguous: multiple matching GPUs; no exact address";
    else result.evidence = kind == 2 ? "Unknown: linear buffer has no GPU identity"
                                    : "Unknown: buffer vendor has no matching visible GPU";
    return result;
}

// Immutable inventory is captured at Vulkan startup. The compositor publishes
// only a tiny atomic observation; Wayland requests never query Vulkan or IO.
class GpuDiagnosticsStore {
    std::mutex mutex;
    GpuDiagnostics inventory;
    std::atomic<uint32_t> observation{0};
public:
    void initialize(GpuDiagnostics value) {
        std::lock_guard lock(mutex); inventory = std::move(value);
        observation.store(0,std::memory_order_relaxed);
    }
    void observe(const char *origin, bool staged) {
        observation.store(gpu_buffer_observation(origin,staged),std::memory_order_relaxed);
    }
    GpuDiagnostics snapshot() {
        std::lock_guard lock(mutex);
        return resolve_gpu_diagnostics(inventory,observation.load(std::memory_order_relaxed));
    }
};
inline GpuDiagnosticsStore &gpu_diagnostics_store() { static GpuDiagnosticsStore store; return store; }
} // namespace gamescope::framegen
