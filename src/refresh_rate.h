#pragma once

#include <cstdint>

namespace gamescope
{
    // The simple CLI/X11 limit persists across paints and connector changes.
    // A combined refresh override replaces it only when it also controls FPS.
    constexpr int32_t ResolveFPSLimit( int32_t nSimpleLimit, int32_t nCombinedOverride,
                                      bool bChangeFPS )
    {
        return nCombinedOverride > 0 && bChangeFPS
            ? nCombinedOverride : ( nSimpleLimit > 0 ? nSimpleLimit : 0 );
    }

    constexpr int32_t ConvertHztomHz( int32_t nRefreshHz )
    {
        return nRefreshHz * 1'000;
    }

    constexpr int32_t ConvertmHzToHz( int32_t nRefreshmHz )
    {
        // Round to nearest when going to mHz.
        // Ceil seems to be wrong when we have 60.001 or 90.004 etc.
        // Floor seems to be bad if we have 143.99
        // So round to nearest.

        return ( nRefreshmHz + 499 ) / 1'000;
    }

    constexpr uint32_t ConvertHztomHz( uint32_t nRefreshHz )
    {
        return nRefreshHz * 1'000;
    }

    constexpr uint32_t ConvertmHzToHz( uint32_t nRefreshmHz )
    {
        return ( nRefreshmHz + 499 ) / 1'000;
    }

    constexpr float ConvertHztomHz( float flRefreshHz )
    {
        return flRefreshHz * 1000.0f;
    }

    constexpr float ConvertmHzToHz( float nRefreshmHz )
    {
        return ( nRefreshmHz ) / 1'000.0;
    }

    constexpr uint32_t RefreshCycleTomHz( int32_t nCycle )
    {
        // Round cycle to nearest.
        return ( 1'000'000'000'000ul + ( nCycle / 2 ) - 1 ) / nCycle;
    }

    constexpr uint32_t mHzToRefreshCycle( int32_t nmHz )
    {
        // Same thing.
        return RefreshCycleTomHz( nmHz );
    }

    // A nested -r at the host's own rate (same whole Hz, e.g. -r 120 on a 119.88 Hz host) stays phase-locked to
    // host presentation feedback, so commits land just before host scanout and framegen plans against it.
    // Only a different rate needs its own cadence (upstream 81146cab).
    constexpr bool NestedRefreshFollowsHost( int32_t nNestedmHz, int32_t nHostmHz )
    {
        return nNestedmHz > 0 && nHostmHz > 0 && ConvertmHzToHz( nNestedmHz ) == ConvertmHzToHz( nHostmHz );
    }
}
