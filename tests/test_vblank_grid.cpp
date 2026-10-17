#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include "refresh_rate.h"
#include "vblankmanager.hpp"

using gamescope::NestedRefreshFollowsHost;
using gamescope::VBlankAtOrBefore;

static constexpr uint64_t kInterval = 16'666'666;
static constexpr uint64_t kAnchor = 1'000'000'000;

TEST_CASE( "A vblank time is its own grid point" )
{
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, kAnchor + 3 * kInterval ) == kAnchor + 3 * kInterval );
}

TEST_CASE( "An off-grid time snaps back to the grid through the anchor" )
{
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, kAnchor + 3 * kInterval + 2'500'000 ) == kAnchor + 3 * kInterval );
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, kAnchor + kInterval - 1 ) == kAnchor );
}

TEST_CASE( "A time at or before the anchor never moves the clock back" )
{
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, kAnchor ) == kAnchor );
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, kAnchor - 5'000'000 ) == kAnchor );
	CHECK( VBlankAtOrBefore( kAnchor, kInterval, 0 ) == kAnchor );
}

TEST_CASE( "Off-grid presentation times keep the phase and advance monotonically" )
{
	// Host completion times spaced 17.8-19.6 ms with a 60 ms outlier, as a
	// headless host reports them.
	const uint64_t steps[] = { 60'200'000, 17'800'000, 19'600'000, 18'400'000, 2'500'000, 16'666'666, 33'000'000 };
	uint64_t ulAnchor = kAnchor, ulTime = kAnchor;
	for ( uint64_t step : steps )
	{
		ulTime += step;
		const uint64_t ulNext = VBlankAtOrBefore( ulAnchor, kInterval, ulTime );
		CHECK( ( ulNext - kAnchor ) % kInterval == 0 );
		CHECK( ulNext >= ulAnchor );
		CHECK( ulNext <= ulTime );
		CHECK( ulTime - ulNext < kInterval );
		ulAnchor = ulNext;
	}
}

TEST_CASE( "Hours after the anchor the arithmetic stays exact" )
{
	const uint64_t ulHours = 10ull * 3600 * 1'000'000'000;
	const uint64_t ulTime = kAnchor + ulHours + 7;
	const uint64_t ulSnap = VBlankAtOrBefore( kAnchor, kInterval, ulTime );
	CHECK( ( ulSnap - kAnchor ) % kInterval == 0 );
	CHECK( ulTime - ulSnap < kInterval );
}

TEST_CASE( "A nested rate at the host's whole Hz follows host feedback" )
{
	CHECK( NestedRefreshFollowsHost( 120'000, 120'000 ) );
	CHECK( NestedRefreshFollowsHost( 120'000, 119'880 ) );
	CHECK( NestedRefreshFollowsHost( 60'000, 59'940 ) );
	CHECK_FALSE( NestedRefreshFollowsHost( 120'000, 144'000 ) );
	CHECK_FALSE( NestedRefreshFollowsHost( 60'000, 120'000 ) );
	CHECK_FALSE( NestedRefreshFollowsHost( 120'000, 119'400 ) );
	CHECK_FALSE( NestedRefreshFollowsHost( 120'000, 0 ) );
	CHECK_FALSE( NestedRefreshFollowsHost( 0, 120'000 ) );
}

namespace
{
	constexpr uint64_t kHostPeriod = 8'341'675; // 119.88 Hz VSYNC host
	constexpr uint64_t kNestedInterval = gamescope::mHzToRefreshCycle( 120'000 ); // -r 120

	// Worst distance between a scheduled target and the nearest host vblank over a session. The timer wakes on its
	// own schedule and sees the latest host vblank reported by then, as the nested backends feed it.
	uint64_t WorstHostPhaseError( bool bRateOverride, int nFrames )
	{
		gamescope::VBlankScheduleTime last{};
		uint64_t ulNow = kAnchor + 1'000'000, ulWorst = 0;
		for ( int i = 0; i < nFrames; i++ )
		{
			const uint64_t ulHostVBlank = kAnchor + ( ulNow - kAnchor ) / kHostPeriod * kHostPeriod;
			const gamescope::VBlankScheduleTime next = gamescope::VBlank::Next(
			{
				.ulLastVBlank = ulHostVBlank,
				.ulInterval = kNestedInterval,
				.ulOffset = 2'000'000,
				.ulNow = ulNow,
				.bRateOverride = bRateOverride,
			}, last );
			REQUIRE( next.ulTargetVBlank > last.ulTargetVBlank );
			const uint64_t ulPhase = ( next.ulTargetVBlank - kAnchor ) % kHostPeriod;
			ulWorst = std::max( ulWorst, std::min( ulPhase, kHostPeriod - ulPhase ) );
			last = next;
			ulNow = next.ulScheduledWakeupPoint + 50'000;
		}
		return ulWorst;
	}
}

TEST_CASE( "An explicit 120 Hz rate on a 119.88 Hz host stays on host scanout" )
{
	const int32_t nHostmHz = int32_t( gamescope::RefreshCycleTomHz( int32_t( kHostPeriod ) ) );
	const bool bRateOverride = !NestedRefreshFollowsHost( 120'000, nHostmHz );
	CHECK_FALSE( bRateOverride );
	// The lattice re-anchors on every host vblank, so the error never exceeds the period mismatch over the
	// two intervals between the latest feedback and the target.
	CHECK( WorstHostPhaseError( bRateOverride, 2000 ) <= 2 * ( kHostPeriod - kNestedInterval ) );
}

TEST_CASE( "An own cadence drifts off host scanout" )
{
	// Upstream's independent clock for a different rate: at a matched rate it would walk the commit phase
	// through the whole host interval (a full beat every ~8.3 s at 119.88 Hz).
	CHECK( WorstHostPhaseError( true, 1000 ) > kHostPeriod / 4 );
}
