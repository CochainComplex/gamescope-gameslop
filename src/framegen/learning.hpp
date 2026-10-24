#pragma once

#include <algorithm>
#include <cstdint>

namespace gamescope::framegen
{

// Foreground and idle work share one decision per real pair, including deliberate
// NET_EVERY skips. A cancelled recording restores this state and may retry.
struct NetTrainingCadence
{
	uint64_t consideredPair = 0;
	uint32_t counter = 0;

	[[nodiscard]] constexpr bool due( uint64_t pair, uint32_t every ) const
	{
		return pair != 0u && pair != consideredPair && every != 0u
			&& counter % every == 0u;
	}

	constexpr bool consider( uint64_t pair, uint32_t every )
	{
		if ( pair == 0u || pair == consideredPair || every == 0u )
			return false;
		const bool selected = due( pair, every );
		consideredPair = pair;
		counter++;
		return selected;
	}
};

inline constexpr uint64_t k_uBackgroundLearningIntervalNs = 250'000'000ull;

[[nodiscard]] constexpr bool background_learning_due( uint64_t lastAttemptNs,
	uint64_t nowNs, uint64_t pair, uint64_t consideredPair )
{
	return pair != 0u && pair != consideredPair
		&& ( lastAttemptNs == 0u || ( nowNs >= lastAttemptNs
			&& nowNs - lastAttemptNs >= k_uBackgroundLearningIntervalNs ) );
}

// Unknown cost cannot bootstrap optional work. CPU preparation must fit before
// recording, and the GPU cost is checked again against a fresh submission clock.
// Subtract instead of summing costs so even hostile/overflowing inputs fail closed.
[[nodiscard]] constexpr bool background_learning_fits( uint64_t nowNs,
	uint64_t deadlineNs, uint64_t gpuCostNs, uint64_t cpuCostNs = 0u )
{
	if ( gpuCostNs == 0u || deadlineNs <= nowNs )
		return false;
	const uint64_t available = deadlineNs - nowNs;
	if ( gpuCostNs >= available || cpuCostNs >= available - gpuCostNs )
		return false;
	const uint64_t margin = std::max<uint64_t>( 1'000'000u, gpuCostNs / 2u );
	return margin < available - gpuCostNs - cpuCostNs;
}

} // namespace gamescope::framegen
