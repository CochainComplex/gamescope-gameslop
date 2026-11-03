#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gamescope::framegen::compute
{
// Offline GSCT arithmetic fixture. Input packs lumas/fields then the prior;
// state packs fast,m,v,EMA and an initially empty served-output row.
struct Fixture
{
	uint32_t width, height, step, flags, sceneCut;
	float lr, ema, decay;
	std::vector<float> input, state;
};
[[nodiscard]] Fixture readFixture( const std::string &path );
} // namespace gamescope::framegen::compute
