#include "vulkan_engine.hpp"
#include "framegen/net_layout.hpp"
#include "framegen/training_dispatch.hpp"
#include "training_fixture.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>

using namespace gamescope::framegen::compute;

namespace
{
constexpr size_t P = k_uFramegenNetFloats;

void require( bool value, const char *message )
{
	if ( !value ) throw std::runtime_error( message );
}

DeviceUuid parseUuid( std::string_view text )
{
	require( text.size() == 32, "UUID must contain exactly 32 hexadecimal digits" );
	auto digit = []( char c ) -> uint8_t {
		if ( c >= '0' && c <= '9' ) return uint8_t( c - '0' );
		if ( c >= 'a' && c <= 'f' ) return uint8_t( c - 'a' + 10 );
		if ( c >= 'A' && c <= 'F' ) return uint8_t( c - 'A' + 10 );
		throw std::runtime_error( "UUID contains a non-hexadecimal digit" );
	};
	DeviceUuid result{};
	for ( size_t i = 0; i < result.size(); ++i )
		result[i] = uint8_t( digit( text[2 * i] ) * 16 + digit( text[2 * i + 1] ) );
	require( result != DeviceUuid{}, "zero UUID is not a device identity" );
	return result;
}

template<typename T> std::span<const std::byte> constants( const T &value )
{
	return std::as_bytes( std::span( &value, 1 ) );
}

void writeOutput( const std::string &path, std::span<const float> gradients, std::span<const float> state )
{
	for ( auto values : { gradients, state } )
		for ( float value : values ) require( std::isfinite( value ), "nonfinite reference output" );
	std::ofstream file( path, std::ios::binary | std::ios::trunc );
	require( bool( file ), "cannot open reference output" );
	for ( auto values : { gradients, state } )
		for ( float value : values )
		{
			const uint32_t bits = std::bit_cast<uint32_t>( value );
			const std::array<char, 4> bytes{ char( bits ), char( bits >> 8 ), char( bits >> 16 ), char( bits >> 24 ) };
			file.write( bytes.data(), bytes.size() );
		}
	file.close();
	require( bool( file ), "cannot finish reference output" );
}
} // namespace

int main( int argc, char **argv )
{
	try
	{
		std::optional<DeviceUuid> uuid;
		std::string inputPath, outputPath;
		bool validation = false;
		for ( int i = 1; i < argc; ++i )
		{
			const std::string_view arg = argv[i];
			if ( arg == "--validation" && !validation ) validation = true;
			else if ( arg == "--device-uuid" )
			{
				require( !uuid && i + 1 < argc, "duplicate or missing device UUID" );
				uuid = parseUuid( argv[++i] );
			}
			else if ( arg == "--input" || arg == "--output" )
			{
				auto &path = arg == "--input" ? inputPath : outputPath;
				require( path.empty() && i + 1 < argc, "duplicate or missing file path" );
				path = argv[++i];
			}
			else throw std::runtime_error( "unknown or duplicate option: " + std::string( arg ) );
		}
		require( uuid && !inputPath.empty() && !outputPath.empty(),
			"usage: training-reference --device-uuid 32HEX --input PATH --output PATH [--validation]" );
		const Fixture fixture = readFixture( inputPath );
		VulkanEngine engine( *uuid, FRAMEGEN_TRAIN_GRAD_SHADER, validation );
		require( !engine.device().software, "software device cannot establish GPU parity" );
		const uint32_t optKernel = engine.addKernel( FRAMEGEN_TRAIN_OPT_SHADER );
		const uint32_t seed = fixture.step * 0x9E3779B9u + 0x61C88647u;
		const TrainPush forward{ seed, 0, 0.0f, fixture.flags, 1.0f, fixture.width, fixture.height, fixture.sceneCut };
		const TrainPush reverse{ seed ^ 0x55555555u, 8, 0.0f, fixture.flags, 1.0f, fixture.width, fixture.height, fixture.sceneCut };
		const OptPush optimizer{ fixture.lr, fixture.ema, fixture.decay, fixture.step };
		const std::array<BufferStep, 3> steps{{
			{ 0, constants( forward ), 8 }, { 0, constants( reverse ), 8 },
			{ optKernel, constants( optimizer ), uint32_t( ( P + 63 ) / 64 ) } }};
		std::vector<float> gradients( 16 * P, 0.0f );
		const std::array<std::span<const float>, 3> initial{ fixture.input, gradients, fixture.state };
		const Ticket ticket = engine.submitBuffers( engine.epoch(), initial, steps );
		require( ticket.serial != 0, "reference submission rejected" );
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 10 );
		while ( engine.poll( ticket ) == Status::Pending )
		{
			require( std::chrono::steady_clock::now() < deadline,
				"reference completion timed out; owner teardown must still drain GPU work" );
			std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
		}
		std::vector<float> inputCopy( fixture.input.size() ), state( 5 * P );
		const std::array<std::span<float>, 3> output{ inputCopy, gradients, state };
		require( engine.collectBuffers( ticket, output ), "reference output collection failed" );
		engine.shutdown();
		writeOutput( outputPath, gradients, state );
		std::cout << "original GLSL arithmetic transport reference complete; "
			"production image/sampler parity remains unverified\n";
		return 0;
	}
	catch ( const std::exception &error )
	{
		std::cerr << "training-reference: " << error.what() << '\n';
		return 1;
	}
}
