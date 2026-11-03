#ifndef GAMESCOPE_FRAMEGEN_TRAINING_REPORT_H
#define GAMESCOPE_FRAMEGEN_TRAINING_REPORT_H

// Shared C++/GLSL wire layout. All offsets count 32-bit words. Binding 2 is
// float storage in train/opt and uint storage in health/publish; bytes agree.
#define GSC_REPORT_MAGIC 0x47535452u
#define GSC_REPORT_VERSION 1u
#define GSC_REPORT_PARAMETERS 4644u
#define GSC_REPORT_WORDS 20u
#define GSC_REPORT_STATE_FLOATS (5u * GSC_REPORT_PARAMETERS)
#define GSC_REPORT_CANDIDATE_OFFSET 0u
#define GSC_REPORT_BACKUP_OFFSET GSC_REPORT_STATE_FLOATS
#define GSC_REPORT_WORD_OFFSET (10u * GSC_REPORT_PARAMETERS)
#define GSC_REPORT_BUFFER_FLOATS (GSC_REPORT_WORD_OFFSET + GSC_REPORT_WORDS)
#define GSC_REPORT_MAX_FINITE_BITS 0x7f7fffffu

#define GSC_REPORT_MAGIC_AT 0u
#define GSC_REPORT_VERSION_AT 1u
#define GSC_REPORT_PARAMETERS_AT 2u
#define GSC_REPORT_COVERAGE_AT 3u
#define GSC_REPORT_INVALID_STATE_AT 4u
#define GSC_REPORT_INVALID_SERVED_AT 5u
#define GSC_REPORT_CHANGED_AT 6u
#define GSC_REPORT_MAX_ABS_AT 7u
#define GSC_REPORT_MAX_DELTA_AT 8u
#define GSC_REPORT_EPOCH_LO_AT 9u
#define GSC_REPORT_EPOCH_HI_AT 10u
#define GSC_REPORT_KEY_AT 11u
#define GSC_REPORT_STEP_AT 12u
#define GSC_REPORT_ATTEMPT_AT 13u
#define GSC_REPORT_ACCEPTED_STEP_AT 14u
#define GSC_REPORT_ACCEPTED_AT 15u
#define GSC_REPORT_RESERVED_AT 16u

#ifdef __cplusplus
#include <cstdint>

namespace gamescope::framegen::compute
{
struct TrainingReportPush
{
	uint32_t epochLo, epochHi, key, step, attempt, previousAcceptedStep;
	uint32_t reserved0 = 0, reserved1 = 0;
};
static_assert( sizeof( TrainingReportPush ) == 32 );
} // namespace gamescope::framegen::compute
#endif
#endif
