#pragma once

#include <cstdint>

#include "refresh_rate.h"
#include "vblankmanager.hpp"

// steamcompmgr.hpp has no include guard; declare the one function needed.
uint64_t get_time_in_nanos();

// The next display vblank after now on the host-phase grid: the last marked
// vblank plus whole refresh intervals, exactly what CVBlankTimer::GetNextVBlank( 0 )
// returned before upstream 72ad630c removed it. Framegen plans generated slots
// against the display's real scanout phase, so this deliberately ignores the
// compositor's wake schedule (target floor, self-anchored explicit -r rates).
inline uint64_t framegen_next_vblank_ns()
{
	gamescope::CVBlankTimer &timer = GetVBlankTimer();
	return gamescope::VBlank::Next(
	{
		.ulLastVBlank = timer.GetLastVBlank(),
		.ulInterval = gamescope::mHzToRefreshCycle( timer.GetRefresh() ),
		.ulOffset = 0,
		.ulNow = get_time_in_nanos(),
	}, gamescope::VBlankScheduleTime{} ).ulTargetVBlank;
}
