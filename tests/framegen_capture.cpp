#include <catch2/catch_test_macros.hpp>

#include "framegen/atomic_file.hpp"
#include "framegen/capture_file.hpp"
#include "framegen/color_probe.hpp"

#include <array>
#include <bit>
#include <cerrno>
#include <csignal>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

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

TEST_CASE( "Atomic replacement publishes complete bytes while an old reader keeps its snapshot", "[framegen_capture][atomic_file]" )
{
	CaptureFiles files;
	const auto path = files.directory / "profile.bin";
	const std::vector<uint8_t> original{ 1, 0, 2, 255, 3 };
	const std::vector<uint8_t> replacement{ 0, 241, 4 };
	{
		std::ofstream seed( path, std::ios::binary );
		seed.write( reinterpret_cast<const char *>( original.data() ), original.size() );
		REQUIRE( seed.good() );
	}
	// Do not pre-read: cached stream bytes could conceal an in-place rewrite.
	std::ifstream oldReader( path, std::ios::binary );
	REQUIRE( oldReader.is_open() );
	AtomicOutputFile output( path );
	REQUIRE( output.is_open() );
	REQUIRE( output.write( nullptr, 0 ) );
	REQUIRE( output.write( replacement.data(), replacement.size() ) );
	REQUIRE( read_file( path ) == original );
	REQUIRE( output.commit() );
	REQUIRE_FALSE( output.is_open() );
	REQUIRE( output.error() == 0 );
	REQUIRE( read_file( path ) == replacement );
	const std::vector<uint8_t> oldBytes{ std::istreambuf_iterator<char>( oldReader ),
		std::istreambuf_iterator<char>() };
	REQUIRE( oldBytes == original );
	REQUIRE_FALSE( output.write( replacement.data(), replacement.size() ) );
	REQUIRE_FALSE( output.commit() );
	REQUIRE( output.error() == EBADF );
	REQUIRE( read_file( path ) == replacement );
	REQUIRE( std::distance( std::filesystem::directory_iterator( files.directory ),
		std::filesystem::directory_iterator() ) == 1 );
}

TEST_CASE( "An invalid atomic write poisons publication and immediately discards staging", "[framegen_capture][atomic_file]" )
{
	CaptureFiles files;
	const auto path = files.directory / "profile.bin";
	const std::vector<uint8_t> original{ 17, 0, 42 };
	{
		std::ofstream seed( path, std::ios::binary );
		seed.write( reinterpret_cast<const char *>( original.data() ), original.size() );
		REQUIRE( seed.good() );
	}
	AtomicOutputFile output( path );
	REQUIRE( output.write( original.data(), 1 ) );
	REQUIRE_FALSE( output.write( nullptr, 1 ) );
	REQUIRE( output.error() == EINVAL );
	REQUIRE_FALSE( output.write( original.data(), original.size() ) );
	REQUIRE_FALSE( output.write( nullptr, 0 ) );
	REQUIRE_FALSE( output.commit() );
	REQUIRE_FALSE( output.is_open() );
	REQUIRE( output.error() == EINVAL );
	REQUIRE( read_file( path ) == original );
	REQUIRE( std::distance( std::filesystem::directory_iterator( files.directory ),
		std::filesystem::directory_iterator() ) == 1 );
	REQUIRE_FALSE( output.commit() );
	REQUIRE( output.error() == EINVAL ); // EBADF must not replace the first error.
}

TEST_CASE( "Atomic rename failure preserves the destination and a fresh writer can recover", "[framegen_capture][atomic_file]" )
{
	CaptureFiles files;
	const auto path = files.directory / "profile.bin";
	REQUIRE( std::filesystem::create_directory( path ) );
	{
		std::ofstream sentinel( path / "keep" );
		sentinel << "old";
		REQUIRE( sentinel.good() );
	}
	const std::vector<uint8_t> replacement{ 7, 8, 9 };
	{
		AtomicOutputFile output( path );
		REQUIRE( output.write( replacement.data(), replacement.size() ) );
		REQUIRE_FALSE( output.commit() );
		REQUIRE_FALSE( output.is_open() );
		REQUIRE( output.error() != 0 );
		REQUIRE( std::filesystem::is_directory( path ) );
		REQUIRE( read_file( path / "keep" ) == std::vector<uint8_t>{ 'o', 'l', 'd' } );
		REQUIRE( std::distance( std::filesystem::directory_iterator( files.directory ),
			std::filesystem::directory_iterator() ) == 1 );
	}
	REQUIRE( std::filesystem::remove_all( path ) == 2 );
	AtomicOutputFile recovered( path );
	REQUIRE( recovered.write( replacement.data(), replacement.size() ) );
	REQUIRE( recovered.commit() );
	REQUIRE( recovered.error() == 0 );
	REQUIRE( read_file( path ) == replacement );
}

TEST_CASE( "A real filesystem write failure cannot publish a partial atomic file", "[framegen_capture][atomic_file]" )
{
	CaptureFiles files;
	const auto path = files.directory / "profile.bin";
	const std::vector<uint8_t> original{ 21, 0, 43 };
	{
		std::ofstream seed( path, std::ios::binary );
		seed.write( reinterpret_cast<const char *>( original.data() ), original.size() );
		REQUIRE( seed.good() );
	}
	// Only the child changes its limit/signal disposition. Both a buffered close
	// failure and an immediate fwrite failure must preserve the published file.
	for ( size_t size : { size_t{ 1 }, size_t{ 65536 } } )
	{
		const pid_t child = fork();
		if ( child == 0 )
		{
			struct sigaction ignore{};
			ignore.sa_handler = SIG_IGN;
			sigemptyset( &ignore.sa_mask );
			struct rlimit limit{};
			if ( sigaction( SIGXFSZ, &ignore, nullptr ) != 0 || getrlimit( RLIMIT_FSIZE, &limit ) != 0 )
				_exit( 1 );
			limit.rlim_cur = 0;
			if ( setrlimit( RLIMIT_FSIZE, &limit ) != 0 )
				_exit( 2 );
			int result = 0;
			{
				AtomicOutputFile output( path );
				if ( !output.is_open() )
					result = 3;
				else
				{
					const std::vector<uint8_t> bytes( size, 0xab );
					(void)output.write( bytes.data(), bytes.size() );
					if ( output.commit() || output.error() != EFBIG || output.is_open() )
						result = 4;
					if ( std::distance( std::filesystem::directory_iterator( files.directory ),
						std::filesystem::directory_iterator() ) != 1 )
						result = 5;
				}
			} // Destruction must run before _exit; inherited test state must not.
			_exit( result );
		}
		REQUIRE( child >= 0 );
		int status = 0;
		pid_t waited;
		do
			waited = waitpid( child, &status, 0 );
		while ( waited < 0 && errno == EINTR );
		REQUIRE( waited == child );
		REQUIRE( WIFEXITED( status ) );
		REQUIRE( WEXITSTATUS( status ) == 0 );
		REQUIRE( read_file( path ) == original );
		REQUIRE( std::distance( std::filesystem::directory_iterator( files.directory ),
			std::filesystem::directory_iterator() ) == 1 );
	}
}

TEST_CASE( "Color probes retain the exact held-out frame before the endpoint", "[framegen_capture]" )
{
	using Action = ColorProbeFrameAction;
	REQUIRE( color_probe_frame_action( false, 0, 2, 4 ) == Action::Wait );
	REQUIRE( color_probe_frame_action( false, 1, 2, 4 ) == Action::Wait );
	REQUIRE( color_probe_frame_action( false, 2, 2, 4 ) == Action::RetainReference );
	REQUIRE( color_probe_frame_action( false, 3, 2, 4 ) == Action::Reseed );
	REQUIRE( color_probe_frame_action( false, 4, 2, 4 ) == Action::Reseed );
	REQUIRE( color_probe_frame_action( true, 2, 2, 4 ) == Action::Wait );
	REQUIRE( color_probe_frame_action( true, 3, 2, 4 ) == Action::Wait );
	REQUIRE( color_probe_frame_action( true, 4, 2, 4 ) == Action::Endpoint );
	REQUIRE( color_probe_frame_action( true, 5, 2, 4 ) == Action::Reseed );
	REQUIRE( color_probe_frame_action( true, std::numeric_limits<uint64_t>::max(), 2, 4 ) == Action::Reseed );
	// Without B, even C cannot advance the sequence. Reference selection takes
	// precedence; lexical setting validation remains outside this policy.
	REQUIRE( color_probe_frame_action( false, 4, 4, 4 ) == Action::RetainReference );
	REQUIRE( color_probe_frame_action( true, 4, 4, 4 ) == Action::Endpoint );
}

TEST_CASE( "Color probe phase uses measured intervals and strict endpoint limits", "[framegen_capture]" )
{
	// A large clock epoch must not erase a small interval through fp32 rounding.
	constexpr uint64_t epoch = uint64_t{ 1 } << 60;
	const auto phase = color_probe_phase( epoch, epoch + 25, epoch + 100, 1, 2 );
	REQUIRE( phase.has_value() );
	REQUIRE( phase->phase == 0.25f );
	REQUIRE( phase->targetPhase == 0.5f );
	REQUIRE( !color_probe_phase( 0, 5, 100, 1, 2 ) );
	REQUIRE( color_probe_phase( 0, 6, 100, 1, 2 ).has_value() );
	REQUIRE( color_probe_phase( 0, 94, 100, 1, 2 ).has_value() );
	REQUIRE( !color_probe_phase( 0, 95, 100, 1, 2 ) );
	REQUIRE( !color_probe_phase( 100, 100, 100, 1, 2 ) );
	REQUIRE( !color_probe_phase( 100, 101, 100, 1, 2 ) );

	// A power-of-two span represents each adjacent binary32 phase exactly.
	const auto atPhase = []( float value )
	{
		constexpr uint64_t span = uint64_t{ 1 } << 32;
		return color_probe_phase( 0, uint64_t( double( value ) * double( span ) ), span, 1, 2 );
	};
	REQUIRE( !atPhase( std::nextafter( 0.05f, 0.0f ) ) );
	REQUIRE( !atPhase( 0.05f ) );
	const auto first = atPhase( std::nextafter( 0.05f, 1.0f ) );
	REQUIRE( first.has_value() );
	REQUIRE( first->phase == std::nextafter( 0.05f, 1.0f ) );
	const auto last = atPhase( std::nextafter( 0.95f, 0.0f ) );
	REQUIRE( last.has_value() );
	REQUIRE( last->phase == std::nextafter( 0.95f, 0.0f ) );
	REQUIRE( !atPhase( 0.95f ) );
	REQUIRE( !atPhase( std::nextafter( 0.95f, 1.0f ) ) );
}

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

TEST_CASE( "GSCF should preserve full width frame identities and nanosecond epochs", "[framegen_capture]" )
{
	CaptureFiles files;
	const auto path = files.directory / "epoch.gscf";
	PaddedPlanes planes( { 8, 8, 8, 8 } );
	const ColorCapture capture{
		.width = 3, .height = 2, .drmFormat = 0x48344241, .bytesPerPixel = 8,
		.eotf = 2, .sweep = 1,
		.sequence = 0xfffffffffffffffeull,
		.anchorId = 0x8000000000000010ull, .referenceId = 0x8000000000000011ull,
		.endpointId = 0x8000000000000012ull,
		.anchorTimeNs = 0xfffffffffffffc00ull, .referenceTimeNs = 0xfffffffffffffd00ull,
		.endpointTimeNs = 0xffffffffffffff00ull,
		.phase = std::bit_cast<float>( 0x3eaaaaabu ),
		.strengths = { 0.0f, 0.375f, 0.875f },
		.planes = planes.views,
	};
	AtomicOutputFile output( path );
	CHECK( output.is_open() );
	CHECK( write_color_capture( output, capture ) );
	CHECK( output.commit() );
	std::vector<uint8_t> expected;
	append_words( expected, { 0x46435347, 3, 3, 2, 0x48344241, 8, 2, 3, 4, 1, 0, 0 }, 4 );
	append_words( expected, { 0xfffffffffffffffeull, 0x8000000000000010ull,
		0x8000000000000011ull, 0x8000000000000012ull, 0xfffffffffffffc00ull,
		0xfffffffffffffd00ull, 0xffffffffffffff00ull, 0 }, 8 );
	append_words( expected, { 0x3eaaaaab, 0, 0x3ec00000, 0x3f600000 }, 4 );
	expected.insert( expected.end(), planes.payload.begin(), planes.payload.end() );
	CHECK( read_file( path ) == expected );
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
