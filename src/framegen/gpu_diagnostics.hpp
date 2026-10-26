#pragma once

#include <array>
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
enum GpuBufferKind : uint32_t {
    kGpuBufferUnobserved = 0, kGpuBufferCpu = 1, kGpuBufferLinear = 2,
    kGpuBufferAmd = 3, kGpuBufferNvidia = 4, kGpuBufferIntel = 5, kGpuBufferUnknown = 6,
};
inline uint32_t gpu_buffer_observation(const char *origin, bool staged) {
    const uint32_t staging = staged ? 256u : 0u;
    if (!origin || !*origin) return kGpuBufferUnobserved | staging;
    constexpr auto origins = [] {
        std::array<const char *, kGpuBufferUnknown - 1> names{};
        names[kGpuBufferCpu - 1] = "cpu"; names[kGpuBufferLinear - 1] = "linear";
        names[kGpuBufferAmd - 1] = "AMD"; names[kGpuBufferNvidia - 1] = "NVIDIA";
        names[kGpuBufferIntel - 1] = "Intel";
        return names;
    }();
    for (uint32_t i = 0; i < origins.size(); ++i) {
        if (!std::strcmp(origin, origins[i])) return (i + 1) | staging;
    }
    return kGpuBufferUnknown | staging;
}

inline bool gpu_buffer_matches_card(const GpuCard &card, uint32_t index,
                                    uint32_t presenter, uint32_t kind, bool staged) {
    if (card.software) return false;
    switch (kind) {
        case kGpuBufferLinear: return staged && index != presenter;
        case kGpuBufferAmd: return card.vendor == 0x1002;
        case kGpuBufferNvidia: return card.vendor == 0x10de;
        case kGpuBufferIntel: return card.vendor == 0x8086;
        default: return false;
    }
}
inline GpuDiagnostics resolve_gpu_diagnostics(GpuDiagnostics result, uint32_t observation) {
    result.renderer = kUnknownGpu;
    const uint32_t kind = observation & 255u;
    const bool staged = (observation & 256u) != 0;
    if (kind == kGpuBufferUnobserved) {
        result.transport = "not observed";
        result.evidence = "No composited client buffer observed";
        return result;
    }
    result.transport = staged ? "staged / copied" : "imported / not staged";
    if (kind == kGpuBufferCpu) { result.evidence = "CPU/shared-memory buffer; GPU producer unknown"; return result; }
    uint32_t matches = 0, candidate = kUnknownGpu;
    for (uint32_t i=0; i<result.cards.size(); ++i) {
        if (!gpu_buffer_matches_card(result.cards[i], i, result.presenter, kind, staged)) continue;
        ++matches; candidate = i;
    }
    if (matches == 1) {
        result.renderer = candidate;
        result.evidence = kind == kGpuBufferLinear ? "Inferred: staged linear buffer, one other GPU"
                                   : "Inferred: modifier vendor, one matching GPU";
    } else if (matches > 1) result.evidence = "Ambiguous: multiple matching GPUs; no exact address";
    else result.evidence = kind == kGpuBufferLinear ? "Unknown: linear buffer has no GPU identity"
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
