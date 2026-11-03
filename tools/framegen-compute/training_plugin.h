#ifndef GAMESCOPE_COMPUTE_TRAINING_PLUGIN_H
#define GAMESCOPE_COMPUTE_TRAINING_PLUGIN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// SDK-free, same-process native C ABI. A caller explicitly loads a trusted
// plugin and negotiates this exact version. No C++/HIP/CUDA/Vulkan types cross
// the boundary. v1 targets a 64-bit little-endian host with naturally aligned
// structures; all lengths count floats. New contexts start at epoch 1.
#define GSC_TRAIN_ABI 1u
#define GSC_TRAIN_PARAMETERS 4644u
#define GSC_TRAIN_SLICES 16u
#define GSC_TRAIN_MODEL_VERSION 3u

enum gsc_train_result {
    GSC_TRAIN_OK = 0,
    GSC_TRAIN_PENDING = 1,
    GSC_TRAIN_COMPLETE = 2,
    GSC_TRAIN_DISCARDED = 3,
    GSC_TRAIN_INVALID = -1,
    GSC_TRAIN_UNAVAILABLE = -2,
    GSC_TRAIN_BUSY = -3,
    GSC_TRAIN_STALE = -4,
    GSC_TRAIN_FAILED = -5
};

struct gsc_train_device {
    uint32_t size;
    char name[128];
    char pci[32];
    uint8_t uuid[16];
};

struct gsc_train_create {
    uint32_t size;
    uint32_t abi;
    struct gsc_train_device device;
};

struct gsc_train_ticket { uint64_t epoch, serial; };

// Causal without shading (flags=0), conservative bidir (flags=4), and an
// explicit cut guard (flags|1, scene_cut) are supported. Third-frame shading
// (mask 2) is not represented by GSFD v1 and is rejected, never approximated.
// State is [fast,m,v,EMA], each row containing PARAMETERS floats. The job uses
// fast weights, fixed 8+8 hashed tiles, then the current online Adam optimizer.
// Inputs are copied before submit returns. They are never mutated. The caller
// alone commits candidate state after successful collection/health validation.
struct gsc_train_job {
    uint32_t size;
    uint32_t model_version;
    uint64_t epoch;
    uint32_t width, height;
    uint32_t step, flags, scene_cut;
    float learning_rate, ema_alpha, prior_decay;
    uint64_t luma_count, field_count, prior_count, state_count;
    const float *source, *destination, *forward, *reverse, *prior, *state;
};

struct gsc_train_output {
    uint32_t size;
    uint64_t gradient_count, state_count, served_count;
    float *gradients, *state, *served;
};

// Single-owner context. One outstanding job; discard does not retire it.
// poll is nonblocking. collect/retire require actual completion. reset refuses
// outstanding work and advances epoch. destroy drains the plugin-owned stream;
// never call it while holding a compositor lock or on a live present path.
// Failed destruction poisons admission; retain the context/module and retry
// destruction only. Never reuse partially retired allocations.
struct gsc_training_api {
    uint32_t size, abi, model_version, parameters, slices;
    const char *name;
    int (*devices)(struct gsc_train_device *, uint32_t capacity, uint32_t *count,
                   char *error, uint32_t error_size);
    int (*create)(const struct gsc_train_create *, void **context, char *error, uint32_t error_size);
    int (*submit)(void *, const struct gsc_train_job *, struct gsc_train_ticket *, char *, uint32_t);
    int (*poll)(void *, struct gsc_train_ticket, char *, uint32_t);
    int (*collect)(void *, struct gsc_train_ticket, struct gsc_train_output *, char *, uint32_t);
    int (*discard)(void *, struct gsc_train_ticket, char *, uint32_t);
    int (*retire)(void *, struct gsc_train_ticket, char *, uint32_t);
    int (*reset)(void *, uint64_t *epoch, char *, uint32_t);
    int (*destroy)(void *, char *, uint32_t);
};

// Return NULL for an unknown version or a short function-table size. The
// caller must check all layout/model fields before calling any other function.
typedef const struct gsc_training_api *(*gsc_training_get_api_fn)(uint32_t abi, uint32_t size);
const struct gsc_training_api *gsc_training_get_api(uint32_t abi, uint32_t size);

#ifdef __cplusplus
}
#endif
#endif
