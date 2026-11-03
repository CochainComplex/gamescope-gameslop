#ifndef GAMESCOPE_FRAMEGEN_BACKEND_CONTRACT_H
#define GAMESCOPE_FRAMEGEN_BACKEND_CONTRACT_H

#include <stdint.h>

enum gsc_backend_constants {
#define GSC_CONSTANT(name, value) name = value,
#define GSC_PROFILE(...)
#include "backend_contract.def"
#undef GSC_PROFILE
#undef GSC_CONSTANT
};

// Cold metadata only: no SDK handles, tensor storage, callback wrappers or job
// queue. Typed payloads keep their existing ABI and resource/lifetime owner.
struct gsc_backend_contract {
    uint32_t size, abi, operation, execution, payload, model_version;
    uint32_t semantics, state, transport, features, max_inflight, reserved;
};

#ifdef __cplusplus
#define GSC_CONTRACT_INLINE static constexpr
static_assert(sizeof(gsc_backend_contract) == 48);
#else
#define GSC_CONTRACT_INLINE static inline
#endif

#define GSC_CONSTANT(name, value)
#define GSC_PROFILE(name, op, exec, payload, model, semantics, state, transport, features, count) \
GSC_CONTRACT_INLINE struct gsc_backend_contract gsc_contract_##name(void) { \
    struct gsc_backend_contract result = {sizeof(struct gsc_backend_contract), GSC_BACKEND_ABI, \
        op, exec, payload, model, semantics, state, transport, features, count, 0}; \
    return result; \
}
#include "backend_contract.def"
#undef GSC_PROFILE
#undef GSC_CONSTANT

GSC_CONTRACT_INLINE int gsc_backend_contract_valid(const struct gsc_backend_contract *value) {
    struct gsc_backend_contract schema;
    uint32_t allowed;
    if (!value || value->size != sizeof(*value) || value->abi != GSC_BACKEND_ABI || value->reserved)
        return 0;
    switch (value->payload) {
    case GSC_PAYLOAD_PAIR: schema = gsc_contract_generation(); break;
    case GSC_PAYLOAD_SNAPSHOT_TRAINING: schema = gsc_contract_snapshot_training(); break;
    case GSC_PAYLOAD_RESIDENT_TRAINING: schema = gsc_contract_resident_training(); break;
    case GSC_PAYLOAD_RAW_HEAD: schema = gsc_contract_npu_raw_head(); break;
    default: return 0;
    }
    // Exact selection is independent of the runtime's inference capability.
    // Current NPU adapters do not promise it; a future typed provider may.
    allowed = schema.features | GSC_FEATURE_EXACT_DEVICE;
    return value->operation == schema.operation && value->execution == schema.execution
        && value->model_version == schema.model_version && value->semantics == schema.semantics
        && value->state == schema.state && value->transport == schema.transport
        && value->max_inflight == schema.max_inflight && !(value->features & ~allowed);
}
#undef GSC_CONTRACT_INLINE

// Optional native-plugin descriptor entry point. HIP training ABI v1 remains
// unchanged. An old, validated v1 table has the known snapshot-training schema.
typedef const struct gsc_backend_contract *(*gsc_compute_contract_fn)(uint32_t abi, uint32_t size, uint32_t payload);

#endif
