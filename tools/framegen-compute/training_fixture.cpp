#include "training_fixture.hpp"
#include "framegen/net_layout.hpp"
#include "framegen/numeric.hpp"

#include <array>
#include <bit>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace gamescope::framegen::compute
{
namespace
{
void require( bool condition, const char *message )
{
	if ( !condition ) throw std::runtime_error( message );
}
uint32_t word( const uint8_t *data )
{
	return uint32_t( data[0] ) | uint32_t( data[1] ) << 8
		| uint32_t( data[2] ) << 16 | uint32_t( data[3] ) << 24;
}
}

Fixture readFixture( const std::string &path )
{
	constexpr size_t P = k_uFramegenNetFloats, HeaderBytes = 44;
	std::ifstream file( path, std::ios::binary );
	require( bool( file ), "cannot open fixture" );
	std::array<uint8_t, HeaderBytes> header{};
	require( bool( file.read( reinterpret_cast<char *>( header.data() ), header.size() ) ), "truncated fixture header" );
	std::array<uint32_t, 11> values{};
	for ( size_t i = 0; i < values.size(); ++i ) values[i] = word( header.data() + 4 * i );
	require( values[0] == 0x54534347u && values[1] == 1 && values[7] == 0, "unsupported fixture header" );
	Fixture result{ values[2], values[3], values[4], values[5], values[6],
		std::bit_cast<float>( values[8] ), std::bit_cast<float>( values[9] ), std::bit_cast<float>( values[10] ), {}, {} };
	require( result.width >= 16 && result.height >= 16
		&& result.width <= INT32_MAX && result.height <= INT32_MAX, "invalid fixture dimensions" );
	require( result.step >= 1 && ( result.flags & ~5u ) == 0 && result.sceneCut <= 1,
		"unsupported training step, flags or scene-cut value (shading history is unavailable)" );
	require( is_finite_binary32( result.lr ) && result.lr >= 0.0f
		&& is_finite_binary32( result.ema ) && result.ema >= 0.0f && result.ema <= 1.0f
		&& is_finite_binary32( result.decay ) && result.decay >= 0.0f && result.decay <= 1.0f,
		"invalid optimizer scalars" );
	const uint64_t count = uint64_t( result.width ) * result.height;
	require( count <= ( uint64_t( INT32_MAX ) - P ) / 10, "fixture extent exceeds shader addressing" );
	require( count <= ( std::numeric_limits<size_t>::max() - HeaderBytes - 20 * P ) / 40,
		"fixture extent overflows host storage" );
	const size_t n = size_t( count );
	const size_t expected = HeaderBytes + ( 5 * P + 10 * n ) * sizeof( float );
	require( std::filesystem::file_size( path ) == expected, "fixture length disagrees with dimensions" );
	std::vector<float> arrays( 5 * P + 10 * n );
	std::array<uint8_t, 4> bytes{};
	for ( float &value : arrays )
	{
		require( bool( file.read( reinterpret_cast<char *>( bytes.data() ), bytes.size() ) ), "truncated fixture arrays" );
		value = std::bit_cast<float>( word( bytes.data() ) );
		require( is_finite_binary32( value ), "nonfinite fixture value" );
	}
	result.input.assign( arrays.begin() + 5 * P, arrays.end() );
	result.input.insert( result.input.end(), arrays.begin(), arrays.begin() + P );
	result.state.assign( arrays.begin() + P, arrays.begin() + 5 * P );
	result.state.resize( 5 * P, 0.0f );
	return result;
}
} // namespace gamescope::framegen::compute
