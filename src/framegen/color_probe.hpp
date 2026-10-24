#pragma once

#include "numeric.hpp"

#include <cstdint>
#include <optional>

namespace gamescope::framegen
{

enum class ColorProbeFrameAction
{
	Wait,
	Reseed,
	RetainReference,
	Endpoint,
};

// Held-out B must be retained at its exact offset before C can be considered.
// The caller owns sequence continuity, texture lifetime and all state changes.
[[nodiscard]] constexpr ColorProbeFrameAction color_probe_frame_action(
	bool hasReference, uint64_t sequenceOffset,
	uint32_t referenceOffset, uint32_t endpointOffset )
{
	if ( !hasReference )
	{
		if ( sequenceOffset < referenceOffset )
			return ColorProbeFrameAction::Wait;
		if ( sequenceOffset != referenceOffset )
			return ColorProbeFrameAction::Reseed;
		return ColorProbeFrameAction::RetainReference;
	}
	if ( sequenceOffset < endpointOffset )
		return ColorProbeFrameAction::Wait;
	if ( sequenceOffset != endpointOffset )
		return ColorProbeFrameAction::Reseed;
	return ColorProbeFrameAction::Endpoint;
}

struct ColorProbePhase
{
	float phase;
	float targetPhase;
};

// Timestamp ordering and offset validity are checked by the caller. Subtract
// integer timestamps before converting to fp32, then divide in fp32 as before.
// Tolerance remains caller-owned so an invalid phase never reads that setting.
[[nodiscard]] inline std::optional<ColorProbePhase> color_probe_phase(
	uint64_t anchorTimeNs, uint64_t referenceTimeNs, uint64_t endpointTimeNs,
	uint32_t referenceOffset, uint32_t endpointOffset )
{
	const uint64_t spanNs = endpointTimeNs - anchorTimeNs;
	const float phase = (float)( referenceTimeNs - anchorTimeNs ) / (float)spanNs;
	const float targetPhase = (float)referenceOffset / (float)endpointOffset;
	if ( !is_finite_binary32( phase ) || phase <= 0.05f || phase >= 0.95f )
		return std::nullopt;
	return ColorProbePhase{ phase, targetPhase };
}

} // namespace gamescope::framegen
