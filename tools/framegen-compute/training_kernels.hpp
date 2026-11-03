#pragma once

#include <hip/hip_runtime_api.h>
#include <cstdint>

namespace gamescope::framegen::compute
{
struct TrainingKernelJob
{
    uint32_t width, height, step, flags, sceneCut;
    float learningRate, emaAlpha, priorDecay;
    const float *source, *destination, *forward, *reverse, *prior, *state;
    float *gradients, *candidate, *served;
};

// Exact current online train+optimizer operations, one stream, no host waits.
hipError_t launch_training( const TrainingKernelJob &job, hipStream_t stream );
} // namespace gamescope::framegen::compute
