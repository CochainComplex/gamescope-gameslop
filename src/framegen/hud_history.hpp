#pragma once

#include <cstdint>

namespace gamescope::framegen
{

// The framegen HUD history split in vulkan_composite, as pure decisions.
// rendervulkan.cpp supplies the facts; bHudFreeHistory, the HUD-free copy and
// the reason given for the one-time "HUD is in real-frame history" note all
// derive from the same state here, so they cannot drift apart.

// Facts vulkan_composite holds when it decides the split.
struct FramegenHudHistoryState
{
	// bCursorSplitCandidate: this composite becomes output-space framegen history.
	bool splitCandidate = false;
	// framegen_hud_level() != 0.
	bool hudOn = false;
	// The cursor split ran: history is already the owned cursor-free image,
	// which the HUD never reaches.
	bool cursorSplit = false;
	// compositeImage != nullptr.
	bool compositeImage = false;
	// compositeImage->transferSrc(); false without an image.
	bool transferSrc = false;
	// The owned HUD-free history image was acquired from the pool.
	bool hudFreeImage = false;
};

// Same candidates as the cursor split, and only with the HUD on.
[[nodiscard]] constexpr bool framegen_hud_split_candidate( const FramegenHudHistoryState &state )
{
	return state.splitCandidate && state.hudOn;
}

// Take an owned HUD-free copy of the composite: only when the cursor split did
// not already leave history HUD-free, and only from a ring image that allows
// the copy (transferSrc).
[[nodiscard]] constexpr bool framegen_hud_history_copy_wanted( const FramegenHudHistoryState &state )
{
	return framegen_hud_split_candidate( state ) && !state.cursorSplit
		&& state.compositeImage && state.transferSrc;
}

// bHudFreeHistory: the image handed to framegen carries no HUD.
[[nodiscard]] constexpr bool framegen_hud_free_history( const FramegenHudHistoryState &state )
{
	return framegen_hud_split_candidate( state ) && ( state.cursorSplit || state.hudFreeImage );
}

// Inputs the reason cascade reads only when it needs them, in this order.
// FramegenEnabled is read again at that point rather than taken from the
// split candidate: it can turn off mid-composite.
enum class FramegenHudHistoryCheck : uint32_t
{
	FramegenEnabled,     // vulkan_framegen_is_enabled()
	OwnsOutputRing,      // framegen_owns_output_ring()
	CursorSplitEnabled,  // framegen_cursor_split_enabled() (GAMESCOPE_FRAMEGEN_CURSOR)
	BidirActive,         // vulkan_framegen_bidir_active()
	ColorProbeRecording, // framegen_color_record_dir() != nullptr
};

inline constexpr const char *k_pszFramegenHudInHistorySwapchain =
	"a Vulkan-swapchain backend (nested SDL) has no compositor-owned output ring";
inline constexpr const char *k_pszFramegenHudInHistoryCursorOff =
	"GAMESCOPE_FRAMEGEN_CURSOR=0 disables the split";
inline constexpr const char *k_pszFramegenHudInHistoryBidir =
	"bidirectional interpolation flips its history images";
inline constexpr const char *k_pszFramegenHudInHistoryColorProbe =
	"the colour probe is recording";
inline constexpr const char *k_pszFramegenHudInHistoryNoTransferSrc =
	"the output image lacks transferSrc";
inline constexpr const char *k_pszFramegenHudInHistoryPoolBusy =
	"the owned history pool was busy";
// Catch-all, e.g. a base-layer scene or a repaint that records no history.
inline constexpr const char *k_pszFramegenHudInHistoryUnmet =
	"the HUD-free split conditions were not met";

// Why the HUD stays in this frame's history: nullptr when the history is
// HUD-free, framegen is off or the HUD is off; otherwise the first reason that
// holds, in a fixed order. `check` answers a FramegenHudHistoryCheck and is
// asked lazily, so nothing past the first holding reason is read.
template <typename Check>
[[nodiscard]] constexpr const char *framegen_hud_in_history_reason(
	const FramegenHudHistoryState &state, Check &&check )
{
	if ( framegen_hud_free_history( state )
		|| !check( FramegenHudHistoryCheck::FramegenEnabled ) || !state.hudOn )
		return nullptr;
	if ( !check( FramegenHudHistoryCheck::OwnsOutputRing ) )
		return k_pszFramegenHudInHistorySwapchain;
	if ( !check( FramegenHudHistoryCheck::CursorSplitEnabled ) )
		return k_pszFramegenHudInHistoryCursorOff;
	if ( check( FramegenHudHistoryCheck::BidirActive ) )
		return k_pszFramegenHudInHistoryBidir;
	if ( check( FramegenHudHistoryCheck::ColorProbeRecording ) )
		return k_pszFramegenHudInHistoryColorProbe;
	if ( framegen_hud_split_candidate( state ) && state.compositeImage && !state.transferSrc )
		return k_pszFramegenHudInHistoryNoTransferSrc;
	if ( framegen_hud_split_candidate( state ) )
		return k_pszFramegenHudInHistoryPoolBusy;
	return k_pszFramegenHudInHistoryUnmet;
}

// framegen_record_real_frame's note that the HUD is in output-space history:
// said once per process, for the first kept real frame that carries the HUD.
struct FramegenHudInHistoryNote
{
	bool logged = false;

	// The reason to log now, or nullptr. A HUD-free frame, or the HUD off,
	// does not use up the note.
	[[nodiscard]] const char *take( bool hudFreeRealFrame, bool hudOn, const char *reason )
	{
		if ( hudFreeRealFrame || !hudOn || logged )
			return nullptr;
		logged = true;
		return reason != nullptr ? reason : k_pszFramegenHudInHistoryUnmet;
	}
};

} // namespace gamescope::framegen
