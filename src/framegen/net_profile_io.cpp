#include "net_profile_io.hpp"

#include <cstdio>
#include <utility>

namespace gamescope::framegen
{

NetWeightsFileRead read_net_weights_file( const char *path, std::vector<float> &weights )
{
	FILE *file = fopen( path, "rb" );
	if ( file == nullptr )
		return { NetWeightsFileStatus::Unreadable, 0u };

	NetProfileMetadata metadata = {};
	bool ok = fread( metadata.data(), sizeof( uint32_t ), metadata.size(), file ) == metadata.size();
	const uint32_t version = ok ? net_profile_metadata_version( metadata ) : 0u;
	ok = version != 0u;
	if ( ok )
	{
		weights.resize( k_uFramegenNetFloats );
		ok = fread( weights.data(), sizeof( float ), weights.size(), file ) == weights.size();
		ok = ok && validate_and_migrate_net_profile_weights( version, weights );
	}
	fclose( file );

	if ( !ok )
	{
		weights.clear();
		return { metadata[ 0 ] == k_uFramegenNetProfileMagic
			? NetWeightsFileStatus::Profile : NetWeightsFileStatus::Malformed, 0u };
	}
	return { NetWeightsFileStatus::Loaded, version };
}

NetProfileFileStatus read_net_profile_file( const char *path, std::vector<NetProfileSet> &sets )
{
	FILE *file = fopen( path, "rb" );
	if ( file == nullptr )
		return NetProfileFileStatus::Unreadable;

	// One byte past the largest valid profile rejects oversized files without
	// reading them all. The existing decoder owns format/finite/key validation.
	std::vector<uint8_t> bytes( k_uFramegenNetProfileMaxBytes + 1u );
	bytes.resize( fread( bytes.data(), 1u, bytes.size(), file ) );
	fclose( file );
	std::optional<std::vector<NetProfileSet>> decoded = decode_net_profile( bytes );
	if ( !decoded )
		return NetProfileFileStatus::Malformed;
	sets = std::move( *decoded );
	return NetProfileFileStatus::Loaded;
}

} // namespace gamescope::framegen
