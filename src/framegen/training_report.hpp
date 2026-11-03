#pragma once

#include "training_report.h"
#include "numeric.hpp"
#include "net_layout.hpp"

#include <array>
#include <bit>
#include <optional>

namespace gamescope::framegen::compute
{
static_assert( GSC_REPORT_PARAMETERS == k_uFramegenNetFloats );
using TrainingReportWords = std::array<uint32_t, GSC_REPORT_WORDS>;

struct TrainingReportIdentity
{
	uint64_t epoch;
	uint32_t key, step, attempt, previousAcceptedStep;
};

struct TrainingReport
{
	TrainingReportIdentity identity;
	uint32_t coverage, invalidState, invalidServed, changedCount;
	float maxAbsServed, maxDelta;
	uint32_t acceptedStep;
	bool accepted;
};

// Diagnostics are nonnegative finite fp32, including either representation of
// zero. Inspect signs/exponents explicitly so fast-math cannot erase admission.
[[nodiscard]] inline bool training_report_metric_valid( uint32_t bits )
{
	return is_finite_binary32( std::bit_cast<float>( bits ) )
		&& ( ( bits & 0x80000000u ) == 0u || ( bits & 0x7fffffffu ) == 0u );
}

[[nodiscard]] inline std::optional<TrainingReport> decode_training_report(
	const TrainingReportWords &words, const TrainingReportIdentity &expected )
{
	if ( expected.epoch == 0 || expected.step == 0 || expected.attempt == 0
		|| uint64_t( expected.previousAcceptedStep ) + 1u != expected.step
		|| words[GSC_REPORT_MAGIC_AT] != GSC_REPORT_MAGIC
		|| words[GSC_REPORT_VERSION_AT] != GSC_REPORT_VERSION
		|| words[GSC_REPORT_PARAMETERS_AT] != GSC_REPORT_PARAMETERS
		|| words[GSC_REPORT_COVERAGE_AT] != GSC_REPORT_PARAMETERS
		|| words[GSC_REPORT_INVALID_STATE_AT] > 4u * GSC_REPORT_PARAMETERS
		|| words[GSC_REPORT_INVALID_SERVED_AT] > GSC_REPORT_PARAMETERS
		|| words[GSC_REPORT_CHANGED_AT] > GSC_REPORT_PARAMETERS
		|| !training_report_metric_valid( words[GSC_REPORT_MAX_ABS_AT] )
		|| !training_report_metric_valid( words[GSC_REPORT_MAX_DELTA_AT] )
		|| words[GSC_REPORT_EPOCH_LO_AT] != uint32_t( expected.epoch )
		|| words[GSC_REPORT_EPOCH_HI_AT] != uint32_t( expected.epoch >> 32 )
		|| words[GSC_REPORT_KEY_AT] != expected.key
		|| words[GSC_REPORT_STEP_AT] != expected.step
		|| words[GSC_REPORT_ATTEMPT_AT] != expected.attempt
		|| words[GSC_REPORT_ACCEPTED_AT] > 1u )
		return std::nullopt;
	for ( uint32_t i = GSC_REPORT_RESERVED_AT; i < GSC_REPORT_WORDS; ++i )
		if ( words[i] != 0u ) return std::nullopt;
	const bool healthy = words[GSC_REPORT_INVALID_STATE_AT] == 0u
		&& words[GSC_REPORT_INVALID_SERVED_AT] == 0u;
	if ( ( words[GSC_REPORT_ACCEPTED_AT] != 0u ) != healthy
		|| words[GSC_REPORT_ACCEPTED_STEP_AT] != ( healthy ? expected.step : expected.previousAcceptedStep ) )
		return std::nullopt;
	return TrainingReport{ expected, words[GSC_REPORT_COVERAGE_AT], words[GSC_REPORT_INVALID_STATE_AT],
		words[GSC_REPORT_INVALID_SERVED_AT], words[GSC_REPORT_CHANGED_AT],
		std::bit_cast<float>( words[GSC_REPORT_MAX_ABS_AT] ), std::bit_cast<float>( words[GSC_REPORT_MAX_DELTA_AT] ),
		words[GSC_REPORT_ACCEPTED_STEP_AT], healthy };
}
} // namespace gamescope::framegen::compute
