#include <catch2/catch_test_macros.hpp>

#include "framegen/net_profile_io.hpp"

#include <bit>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace gamescope::framegen;

namespace
{

struct ProfileFiles
{
	std::filesystem::path directory;
	ProfileFiles()
	{
		char pattern[] = "/tmp/gamescope-profile-input-XXXXXX";
		const char *path = mkdtemp( pattern );
		REQUIRE( path != nullptr );
		directory = path;
	}
	~ProfileFiles() { std::filesystem::remove_all( directory ); }
};

void write_bytes( const std::filesystem::path &path, const void *data, size_t size )
{
	std::ofstream file( path, std::ios::binary );
	file.write( static_cast<const char *>( data ), size );
	REQUIRE( file.good() );
}

void write_blob( const std::filesystem::path &path, uint32_t version, const std::vector<float> &weights )
{
	const NetProfileMetadata metadata = net_profile_metadata( version );
	std::ofstream file( path, std::ios::binary );
	file.write( reinterpret_cast<const char *>( metadata.data() ), sizeof( metadata ) );
	file.write( reinterpret_cast<const char *>( weights.data() ), weights.size() * sizeof( float ) );
	REQUIRE( file.good() );
}

void check_weights( const std::vector<float> &actual, const std::vector<float> &expected )
{
	REQUIRE( actual.size() == expected.size() );
	CHECK( std::memcmp( actual.data(), expected.data(), expected.size() * sizeof( float ) ) == 0 );
}

} // namespace

TEST_CASE( "GSFR files retain exact weights, source version and legacy migration", "[framegen_profile_io]" )
{
	ProfileFiles files;
	const auto path = files.directory / "weights.gsfr";
	std::vector<float> source( k_uFramegenNetFloats, 0.125f );
	source[ 0 ] = std::bit_cast<float>( 0x80000000u );
	for ( uint32_t version : { 1u, 2u, k_uFramegenNetVersion } )
	{
		write_blob( path, version, source );
		// Bare GSFR has always accepted bytes after its complete payload.
		{
			std::ofstream file( path, std::ios::binary | std::ios::app );
			file << "accepted-tail";
			REQUIRE( file.good() );
		}
		std::vector<float> actual{ 42.0f };
		const NetWeightsFileRead result = read_net_weights_file( path.c_str(), actual );
		REQUIRE( result.status == NetWeightsFileStatus::Loaded );
		CHECK( result.version == version );
		std::vector<float> expected = source;
		if ( version < k_uFramegenNetShadingVersion )
		{
			std::fill( expected.begin() + k_uFramegenNetShadingWeightBegin,
				expected.begin() + k_uFramegenNetLayer3BiasOffset, 0.0f );
			expected[ k_uFramegenNetShadingBias ] = 0.0f;
		}
		check_weights( actual, expected );
	}
}

TEST_CASE( "Unopened files leave caller snapshots intact", "[framegen_profile_io]" )
{
	ProfileFiles files;
	const auto regular = files.directory / "regular";
	const uint32_t value = 0;
	write_bytes( regular, &value, sizeof( value ) );
	for ( const auto &path : { files.directory / "missing", regular / "child" } )
	{
		std::vector<float> weights{ 42.0f };
		CHECK( read_net_weights_file( path.c_str(), weights ).status == NetWeightsFileStatus::Unreadable );
		check_weights( weights, { 42.0f } );
		std::vector<NetProfileSet> sets{ { 42, { 17.0f } } };
		CHECK( read_net_profile_file( path.c_str(), sets ) == NetProfileFileStatus::Unreadable );
		REQUIRE( sets.size() == 1 );
		CHECK( sets[ 0 ].key == 42 );
		check_weights( sets[ 0 ].weights, { 17.0f } );
	}
}

TEST_CASE( "Opened invalid GSFR clears weights and preserves wrong-kind diagnostics", "[framegen_profile_io]" )
{
	ProfileFiles files;
	const auto path = files.directory / "bad.gsfr";
	for ( uint32_t magic : { 0u, k_uFramegenNetMagic, k_uFramegenNetProfileMagic } )
	{
		// Even a four-byte GSFP header must select the profile diagnostic.
		write_bytes( path, &magic, sizeof( magic ) );
		std::vector<float> weights{ 42.0f };
		const auto result = read_net_weights_file( path.c_str(), weights );
		CHECK( result.status == ( magic == k_uFramegenNetProfileMagic
			? NetWeightsFileStatus::Profile : NetWeightsFileStatus::Malformed ) );
		CHECK( weights.empty() );
	}
	write_blob( path, k_uFramegenNetVersion, { 1.0f } );
	std::vector<float> weights{ 42.0f };
	CHECK( read_net_weights_file( path.c_str(), weights ).status == NetWeightsFileStatus::Malformed );
	CHECK( weights.empty() );

	for ( uint32_t bits : { 0x7fc00000u, 0x7f800000u, 0xff800000u } )
	{
		std::vector<float> invalid( k_uFramegenNetFloats, 0.125f );
		invalid.back() = std::bit_cast<float>( bits );
		write_blob( path, k_uFramegenNetVersion, invalid );
		weights = { 42.0f };
		CHECK( read_net_weights_file( path.c_str(), weights ).status == NetWeightsFileStatus::Malformed );
		CHECK( weights.empty() );
	}
}

TEST_CASE( "GSFP file loading preserves ordered temporal and spatial key sets", "[framegen_profile_io]" )
{
	ProfileFiles files;
	const auto path = files.directory / "profile.gsfp";
	std::vector<NetProfileSet> source;
	for ( uint32_t i = 0; i < k_uFramegenNetProfileMaxSets; i++ )
	{
		const uint32_t key = net_input_key( true, i % 4, i % 3, i >= 4 );
		source.push_back( { key, std::vector<float>( k_uFramegenNetFloats, float( i ) / 8.0f ) } );
	}
	const auto bytes = encode_net_profile( source );
	REQUIRE( bytes.has_value() );
	REQUIRE( bytes->size() == k_uFramegenNetProfileMaxBytes );
	write_bytes( path, bytes->data(), bytes->size() );
	std::vector<NetProfileSet> actual;
	REQUIRE( read_net_profile_file( path.c_str(), actual ) == NetProfileFileStatus::Loaded );
	REQUIRE( actual.size() == source.size() );
	for ( size_t i = 0; i < source.size(); i++ )
	{
		CHECK( actual[ i ].key == source[ i ].key );
		check_weights( actual[ i ].weights, source[ i ].weights );
	}
	std::vector<float> weights{ 42.0f };
	CHECK( read_net_weights_file( path.c_str(), weights ).status == NetWeightsFileStatus::Profile );
	CHECK( weights.empty() );
}

TEST_CASE( "GSFP rejects short, trailing, oversized and wrong-kind files without replacing sets", "[framegen_profile_io]" )
{
	ProfileFiles files;
	const auto path = files.directory / "invalid.gsfp";
	const std::vector<NetProfileSet> source{ { 17, std::vector<float>( k_uFramegenNetFloats, 0.125f ) } };
	const auto encoded = encode_net_profile( source );
	REQUIRE( encoded.has_value() );
	for ( int fault : { 0, 1, 2, 3 } )
	{
		auto bytes = *encoded;
		switch ( fault )
		{
			case 0: bytes.pop_back(); break;
			case 1: bytes.push_back( 0 ); break;
			case 2: bytes.resize( k_uFramegenNetProfileMaxBytes + 100, 0 ); break;
			case 3: bytes[ 0 ] = 0; break;
		}
		write_bytes( path, bytes.data(), bytes.size() );
		std::vector<NetProfileSet> actual{ { 42, { 17.0f } } };
		CHECK( read_net_profile_file( path.c_str(), actual ) == NetProfileFileStatus::Malformed );
		REQUIRE( actual.size() == 1 );
		CHECK( actual[ 0 ].key == 42 );
		check_weights( actual[ 0 ].weights, { 17.0f } );
	}
	write_blob( path, k_uFramegenNetVersion, source[ 0 ].weights );
	std::vector<NetProfileSet> actual;
	CHECK( read_net_profile_file( path.c_str(), actual ) == NetProfileFileStatus::Malformed );
	CHECK( actual.empty() );
}
