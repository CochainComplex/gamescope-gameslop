#include <catch2/catch_test_macros.hpp>

#include "framegen/atomic_file.hpp"
#include "framegen/capture_file.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace gamescope::framegen;

namespace
{

struct CaptureFiles
{
	std::filesystem::path directory;

	CaptureFiles()
	{
		char pattern[] = "/tmp/gamescope-capture-contract-XXXXXX";
		const char *path = mkdtemp( pattern );
		REQUIRE( path != nullptr );
		directory = path;
	}
	~CaptureFiles() { std::filesystem::remove_all( directory ); }
};

std::vector<uint8_t> read_file( const std::filesystem::path &path )
{
	std::ifstream file( path, std::ios::binary );
	return { std::istreambuf_iterator<char>( file ), std::istreambuf_iterator<char>() };
}

// The public little-endian file contract, independent of native struct layout.
void append_words( std::vector<uint8_t> &bytes, std::initializer_list<uint64_t> words, size_t wordSize )
{
	for ( uint64_t word : words )
		for ( size_t i = 0; i < wordSize; i++ )
			bytes.push_back( uint8_t( word >> ( 8 * i ) ) );
}

struct PaddedPlanes
{
	std::array<std::vector<uint8_t>, 4> storage;
	std::array<CapturePlane, 4> views;
	std::vector<uint8_t> payload;

	explicit PaddedPlanes( std::array<uint32_t, 4> bytesPerPixel )
	{
		for ( size_t i = 0; i < storage.size(); i++ )
		{
			const size_t rowBytes = 3 * bytesPerPixel[ i ];
			const size_t pitch = rowBytes + 5 + i;
			// A nonzero texel origin and different padding per plane must not
			// enter the file. Values also distinguish both rows and plane order.
			storage[ i ].assign( 7 + 2 * pitch, 0xee );
			for ( size_t y = 0; y < 2; y++ )
				for ( size_t x = 0; x < rowBytes; x++ )
				{
					const uint8_t value = uint8_t( 41 * i + 13 * y + x );
					storage[ i ][ 7 + y * pitch + x ] = value;
					payload.push_back( value );
				}
			views[ i ] = { storage[ i ].data() + 7, 3, 2, pitch };
		}
	}
};

} // namespace

TEST_CASE( "GSFD v1 preserves metadata and packs all four planes", "[framegen_capture]" )
{
	CaptureFiles files;
	const auto path = files.directory / "field.bin";
	for ( uint32_t lumaBytes : { 2u, 8u } )
	{
		PaddedPlanes planes( { lumaBytes, lumaBytes, 8, 8 } );
		FieldCapture capture{ 3, 2, lumaBytes, 0x123456789abcdef0ull, planes.views };
		{
			AtomicOutputFile output( path );
			REQUIRE( output.is_open() );
			REQUIRE( write_field_capture( output, capture ) );
			REQUIRE( !std::filesystem::exists( path ) );
			REQUIRE( output.commit() );
		}
		std::vector<uint8_t> expected;
		append_words( expected, { 0x44465347, 1, 3, 2, lumaBytes, 8, 1, 0 }, 4 );
		append_words( expected, { 0x123456789abcdef0ull }, 8 );
		expected.insert( expected.end(), planes.payload.begin(), planes.payload.end() );
		REQUIRE( read_file( path ) == expected );
		std::filesystem::remove( path );
	}
}

TEST_CASE( "GSCF v3 preserves paired metadata and full precision bytes", "[framegen_capture]" )
{
	CaptureFiles files;
	const auto path = files.directory / "color.gscf";
	for ( uint32_t pixelBytes : { 4u, 8u, 16u } )
	{
		PaddedPlanes planes( { pixelBytes, pixelBytes, pixelBytes, pixelBytes } );
		ColorCapture capture{
			.width = 3, .height = 2, .drmFormat = 0x48344241, .bytesPerPixel = pixelBytes,
			.eotf = 2, .sweep = 1,
			.sequence = 0x123456789abcdef0ull, .anchorId = 10, .referenceId = 11, .endpointId = 12,
			.anchorTimeNs = 0x100000001ull, .referenceTimeNs = 0x200000002ull, .endpointTimeNs = 0x300000003ull,
			.phase = 0.25f, .strengths = { 0.0f, 0.5f, 1.0f }, .planes = planes.views,
		};
		{
			AtomicOutputFile output( path );
			REQUIRE( write_color_capture( output, capture ) );
			REQUIRE( !std::filesystem::exists( path ) );
			REQUIRE( output.commit() );
		}
		std::vector<uint8_t> expected;
		append_words( expected, { 0x46435347, 3, 3, 2, 0x48344241, pixelBytes, 2, 3, 4, 1, 0, 0 }, 4 );
		append_words( expected, { 0x123456789abcdef0ull, 10, 11, 12, 0x100000001ull, 0x200000002ull, 0x300000003ull, 0 }, 8 );
		append_words( expected, { 0x3e800000, 0, 0x3f000000, 0x3f800000 }, 4 );
		expected.insert( expected.end(), planes.payload.begin(), planes.payload.end() );
		REQUIRE( read_file( path ) == expected );
		std::filesystem::remove( path );
	}
}

TEST_CASE( "Invalid capture planes abandon publication and preserve the old file", "[framegen_capture]" )
{
	CaptureFiles files;
	const auto path = files.directory / "sample.bin";
	const std::vector<uint8_t> original{ 1, 2, 3, 4 };
	{
		AtomicOutputFile output( path );
		REQUIRE( output.write( original.data(), original.size() ) );
		REQUIRE( output.commit() );
	}
	for ( size_t plane = 0; plane < 4; plane++ )
		for ( int fault = 0; fault < 4; fault++ )
		{
			PaddedPlanes planes( { 8, 8, 8, 8 } );
			switch ( fault )
			{
				case 0: planes.views[ plane ].data = nullptr; break;
				case 1: planes.views[ plane ].rowPitch = 23; break;
				case 2: planes.views[ plane ].width = 2; break;
				case 3: planes.views[ plane ].height = 1; break;
			}
			for ( bool color : { false, true } )
			{
				{
					AtomicOutputFile output( path );
					if ( color )
					{
						ColorCapture capture{};
						capture.width = 3; capture.height = 2; capture.bytesPerPixel = 8;
						capture.planes = planes.views;
						REQUIRE_FALSE( write_color_capture( output, capture ) );
					}
					else
						REQUIRE_FALSE( write_field_capture( output, { 3, 2, 8, 1, planes.views } ) );
					REQUIRE( output.error() == 0 ); // invalid view, not an I/O failure
				}
				REQUIRE( read_file( path ) == original );
				REQUIRE( std::distance( std::filesystem::directory_iterator( files.directory ),
					std::filesystem::directory_iterator() ) == 1 );
			}
		}
}

TEST_CASE( "Capture serializers propagate output failure", "[framegen_capture]" )
{
	CaptureFiles files;
	PaddedPlanes planes( { 8, 8, 8, 8 } );
	AtomicOutputFile output( files.directory / "missing/sample.bin" );
	REQUIRE_FALSE( output.is_open() );
	REQUIRE_FALSE( write_field_capture( output, { 3, 2, 8, 1, planes.views } ) );
	ColorCapture color{};
	color.width = 3; color.height = 2; color.bytesPerPixel = 8; color.planes = planes.views;
	REQUIRE_FALSE( write_color_capture( output, color ) );
	REQUIRE( output.error() != 0 );
}
