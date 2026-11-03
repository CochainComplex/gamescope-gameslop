#pragma once

#include <cstddef>
#include <cstdint>

namespace gamescope::framegen::compute
{
// Push ABI for the unchanged online GLSL bodies used by standalone trainers.
struct TrainPush
{
	uint32_t seed, sliceBase;
	float mode;
	uint32_t flags;
	float timeScale;
	uint32_t width, height, sceneCut;
};
struct OptPush { float lr, ema, decay; uint32_t step; };
static_assert( sizeof( TrainPush ) == 32 && offsetof( TrainPush, width ) == 20 );
static_assert( sizeof( OptPush ) == 16 && offsetof( OptPush, step ) == 12 );
} // namespace gamescope::framegen::compute
