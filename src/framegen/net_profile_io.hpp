#pragma once

#include "net_profile.hpp"

namespace gamescope::framegen
{

enum class NetWeightsFileStatus
{
	Loaded,
	Unreadable,
	Profile,
	Malformed,
};

struct NetWeightsFileRead
{
	NetWeightsFileStatus status;
	uint32_t version; // Source GSFR version; meaningful only when Loaded.
};

// Cold, synchronous input only. An unreadable file leaves weights unchanged;
// an opened invalid file clears them. Valid legacy GSFR weights are migrated
// by the existing codec. Trailing bytes remain accepted for bare GSFR blobs.
[[nodiscard]] NetWeightsFileRead read_net_weights_file( const char *path, std::vector<float> &weights );

enum class NetProfileFileStatus
{
	Loaded,
	Unreadable,
	Malformed,
};

// Reads at most MaxBytes+1, retaining GSFP's exact current-format/length checks.
// Assigns sets only on success. The caller owns missing-file and logging policy.
[[nodiscard]] NetProfileFileStatus read_net_profile_file( const char *path, std::vector<NetProfileSet> &sets );

} // namespace gamescope::framegen
