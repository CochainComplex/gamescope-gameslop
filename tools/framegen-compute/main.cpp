#include "vulkan_engine.hpp"
#include "backend_bindings.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>

using namespace gamescope::framegen::compute;

namespace
{
void require( bool condition, const char *message )
{
	if ( !condition ) throw std::runtime_error( message );
}

DeviceUuid parseUuid( std::string_view text )
{
	require( text.size() == 32, "UUID must contain exactly 32 hexadecimal digits" );
	auto digit = []( char c ) -> int {
		if ( c >= '0' && c <= '9' ) return c - '0';
		if ( c >= 'a' && c <= 'f' ) return c - 'a' + 10;
		if ( c >= 'A' && c <= 'F' ) return c - 'A' + 10;
		throw std::runtime_error( "UUID contains a non-hexadecimal digit" );
	};
	DeviceUuid uuid{};
	for ( size_t i = 0; i < uuid.size(); ++i )
		uuid[i] = uint8_t( digit( text[i * 2] ) * 16 + digit( text[i * 2 + 1] ) );
	require( uuid != DeviceUuid{}, "zero UUID is not a device identity" );
	return uuid;
}

std::string uuidString( DeviceUuid uuid )
{
	std::ostringstream stream;
	stream << std::hex << std::setfill( '0' );
	for ( uint8_t byte : uuid ) stream << std::setw( 2 ) << unsigned( byte );
	return stream.str();
}

void printDevice( const Device &device )
{
	std::cout << "name=" << device.name << " uuid=" << uuidString( device.uuid )
		<< " pci=" << ( device.pci.empty() ? "unavailable" : device.pci )
		<< " vendor=0x" << std::hex << device.vendor << " device=0x" << device.device
		<< std::dec << " software=" << ( device.software ? "yes" : "no" ) << '\n';
}

void usage()
{
	std::cout << "Usage: gamescope-framegen-compute --list\n"
		"       gamescope-framegen-compute --self-test --device-uuid 32HEX "
		"[--render-uuid 32HEX] [--validation]\n"
		"Select the presenter GPU explicitly by UUID. No device is selected by default.\n"
		"This standalone float32 conformance probe makes no quality or performance claim.\n";
}

// A material-space affine field whose slopes also change with content time.
// Different slopes make translation observably different from plain blending.
constexpr std::array<double, 4> Offset{ -0.7, 1.8, 0.3, 0.2 };
constexpr std::array<double, 4> Time{ 0.4, -0.2, 0.7, 0.1 };
constexpr std::array<double, 4> X{ 0.017, -0.009, 0.023, 0.003 };
constexpr std::array<double, 4> Y{ -0.011, 0.019, 0.007, 0.004 };
constexpr std::array<double, 4> ChangeX{ 0.008, 0.006, -0.012, 0.002 };
constexpr std::array<double, 4> ChangeY{ -0.006, 0.009, 0.013, -0.001 };

std::array<double, 4> field( double x, double y, double time )
{
	std::array<double, 4> value;
	for ( size_t c = 0; c < value.size(); ++c )
		value[c] = Offset[c] + Time[c] * time + ( X[c] + ChangeX[c] * time ) * x
			+ ( Y[c] + ChangeY[c] * time ) * y;
	return value;
}

Pixel pixel( const std::array<double, 4> &value )
{
	return { float( value[0] ), float( value[1] ), float( value[2] ), float( value[3] ) };
}

Status waitForCompletion( VulkanEngine &engine, Ticket ticket )
{
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 10 );
	for ( ;; )
	{
		const Status status = engine.poll( ticket );
		if ( status == Status::Complete || status == Status::Discarded ) return status;
		require( status != Status::Stale, "ticket became stale before GPU completion" );
		require( std::chrono::steady_clock::now() < deadline,
			"completion observation timed out; this does not cancel GPU work (owner teardown drains it)" );
		std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
	}
}

void selfTest( VulkanEngine &engine )
{
	constexpr uint32_t width = 65, height = 33;
	constexpr float dx = 2.5f, dy = -1.5f;
	std::vector<Pixel> previous( width * height ), current( width * height ), output( width * height );
	for ( uint32_t y = 0; y < height; ++y )
		for ( uint32_t x = 0; x < width; ++x )
		{
			previous[y * width + x] = pixel( field( x, y, 0.0 ) );
			current[y * width + x] = pixel( field( double( x ) - dx, double( y ) - dy, 1.0 ) );
		}
	Request admission{ { engine.epoch(), 1, 2, 1000, 1400, 1100 },
		width, height, Method::Blend, dx, dy, previous, current };
	auto reject = [&]( Request request ) {
		request.pair.epoch = engine.epoch();
		require( engine.submit( request ).serial == 0, "invalid request admitted" );
		require( engine.reset(), "invalid request retained active ownership" );
	};
	Request invalid = admission; invalid.width = 0; reject( invalid );
	invalid = admission; invalid.height = 0; reject( invalid );
	invalid = admission; invalid.width = UINT32_MAX; invalid.height = UINT32_MAX; reject( invalid );
	invalid = admission; invalid.previous = std::span<const Pixel>( previous ).first( previous.size() - 1 ); reject( invalid );
	invalid = admission; invalid.current = std::span<const Pixel>( current ).first( current.size() - 1 ); reject( invalid );
	invalid = admission; invalid.method = static_cast<Method>( UINT32_MAX ); reject( invalid );
	for ( float nonfinite : { std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity() } )
	{
		invalid = admission; invalid.dx = nonfinite; reject( invalid );
		invalid = admission; invalid.dy = nonfinite; reject( invalid );
		const float previousAlpha = previous[0].a, currentAlpha = current[0].a;
		previous[0].a = nonfinite; reject( admission ); previous[0].a = previousAlpha;
		current[0].a = nonfinite; reject( admission ); current[0].a = currentAlpha;
	}
	uint64_t id = 1;
	Ticket last;
	for ( Method method : { Method::Blend, Method::Translation } )
		for ( uint64_t quarter : { 1u, 2u, 3u } )
		{
			const double phaseValue = double( quarter ) / 4.0;
			Request request{ { engine.epoch(), id, id + 1, 1000, 1400, 1000 + quarter * 100 },
				width, height, method, dx, dy, previous, current };
			id += 2;
			const Ticket ticket = engine.submit( request );
			require( ticket.serial != 0, "valid request rejected" );
			require( engine.submit( request ).serial == 0, "second batch accepted before retirement" );
			require( !engine.reset(), "reset accepted before retirement" );
			if ( last.serial != 0 )
			{
				engine.discard( last );
				require( engine.poll( last ) == Status::Stale, "old ticket poll affected new batch" );
				Pixel canary{ -7.0f, 8.0f, -9.0f, 10.0f };
				require( !engine.collect( last, std::span<Pixel>( &canary, 1 ) ), "old ticket collected during new batch" );
				require( canary.r == -7.0f && canary.g == 8.0f && canary.b == -9.0f && canary.a == 10.0f,
					"old ticket modified destination" );
				require( !engine.retire( last ), "old ticket retired new batch" );
			}
			require( waitForCompletion( engine, ticket ) == Status::Complete, "unexpected discard" );
			require( engine.submit( request ).serial == 0, "completion released ownership before collection" );
			require( !engine.reset(), "completion allowed reset before collection" );
			std::vector<Pixel> undersized( output.size() - 1, Pixel{ -7.0f, 8.0f, -9.0f, 10.0f } );
			require( !engine.collect( ticket, undersized ), "undersized collection succeeded" );
			require( std::all_of( undersized.begin(), undersized.end(), []( const Pixel &p ) {
				return p.r == -7.0f && p.g == 8.0f && p.b == -9.0f && p.a == 10.0f;
			} ), "failed collection modified destination" );
			require( !engine.reset() && engine.submit( request ).serial == 0,
				"failed collection released batch ownership" );
			require( engine.collect( ticket, output ), "completed output could not be collected" );
			require( !engine.collect( ticket, output ), "retired output replay accepted" );
			require( engine.poll( ticket ) == Status::Stale, "collected ticket remained live" );
			double maximumError = 0.0;
			size_t checked = 0;
			for ( uint32_t y = 0; y < height; ++y )
				for ( uint32_t x = 0; x < width; ++x )
				{
					// Translation's analytic oracle applies where neither endpoint clamps.
					if ( method == Method::Translation && ( x < 3 || x >= width - 3 || y < 2 || y >= height - 2 ) ) continue;
					auto expected = field( double( x ) - ( method == Method::Translation ? phaseValue * dx : 0.0 ),
						double( y ) - ( method == Method::Translation ? phaseValue * dy : 0.0 ), phaseValue );
					if ( method == Method::Blend )
						for ( size_t c = 0; c < expected.size(); ++c )
							expected[c] -= phaseValue * ( ( X[c] + ChangeX[c] ) * dx + ( Y[c] + ChangeY[c] ) * dy );
					const Pixel &p = output[y * width + x];
					const std::array<double, 4> actual{ p.r, p.g, p.b, p.a };
					for ( size_t c = 0; c < actual.size(); ++c )
					{
						require( std::isfinite( actual[c] ), "non-finite output" );
						maximumError = std::max( maximumError, std::abs( actual[c] - expected[c] ) );
					}
					++checked;
				}
			require( checked != 0 && maximumError < 0.00002, "analytic RGBA conformance mismatch" );
			std::cout << "method=" << ( method == Method::Blend ? "blend" : "translation" )
				<< " phase=" << phaseValue << " maxerror=" << maximumError << " pixels=" << checked << '\n';
			last = ticket;
		}

	Request discarded{ { engine.epoch(), id, id + 1, 1000, 1400, 1200 },
		width, height, Method::Blend, dx, dy, previous, current };
	const Ticket ticket = engine.submit( discarded );
	require( ticket.serial != 0, "discard probe request rejected" );
	engine.discard( ticket );
	require( engine.submit( discarded ).serial == 0, "discard freed batch ownership" );
	require( !engine.reset(), "discard allowed reset before retirement" );
	require( waitForCompletion( engine, ticket ) == Status::Discarded, "discard status lost" );
	require( !engine.collect( ticket, output ), "discarded output was collected" );
	require( engine.retire( ticket ), "completed discarded batch could not retire" );
	require( !engine.retire( ticket ), "discarded ticket replay accepted" );
	require( engine.reset(), "idle reset rejected" );
	require( engine.poll( last ) == Status::Stale && engine.poll( ticket ) == Status::Stale,
		"old ticket survived epoch reset" );
	require( engine.submit( discarded ).serial == 0, "old epoch request accepted" );
	discarded.pair.epoch = engine.epoch();
	discarded.pair.timeline = Timeline::Causal;
	require( engine.submit( discarded ).serial == 0, "unsupported causal request accepted" );

	// Extreme finite displacement clamps both samples to known opposite corners.
	// A unique RGBA grid makes this an independent border oracle, including 1D
	// images and buffer/descriptor resizing after the larger affine fixtures.
	for ( const auto shape : { std::array<uint32_t, 2>{ 1, 1 }, { 1, 7 }, { 9, 1 } } )
	{
		const uint32_t w = shape[0], h = shape[1];
		std::vector<Pixel> a( w * h ), b( w * h ), result( w * h );
		for ( size_t i = 0; i < a.size(); ++i )
		{
			const float value = float( i );
			a[i] = { -2.0f + value * 0.1f, 3.0f + value * 0.2f, 0.5f + value * 0.3f, 0.1f + value * 0.01f };
			b[i] = { 4.0f + value * 0.4f, -1.0f + value * 0.3f, 2.0f + value * 0.2f, 0.7f + value * 0.02f };
		}
		Request border{ { engine.epoch(), id + 2, id + 3, 1000, 1400, 1100 },
			w, h, Method::Translation, 1e20f, 1e20f, a, b };
		id += 2;
		const Ticket fresh = engine.submit( border );
		require( fresh.serial != 0, "valid new-epoch shape-change request rejected" );
		engine.discard( ticket );
		require( engine.poll( ticket ) == Status::Stale && !engine.retire( ticket ), "stale epoch ticket affected new batch" );
		require( waitForCompletion( engine, fresh ) == Status::Complete, "stale epoch discard affected new batch" );
		require( engine.collect( fresh, result ), "border fixture collection failed" );
		const Pixel &first = a.front(), &lastPixel = b.back();
		const std::array<double, 4> expected{
			0.75 * first.r + 0.25 * lastPixel.r, 0.75 * first.g + 0.25 * lastPixel.g,
			0.75 * first.b + 0.25 * lastPixel.b, 0.75 * first.a + 0.25 * lastPixel.a };
		double maximumError = 0.0;
		for ( const Pixel &p : result )
		{
			const std::array<double, 4> actual{ p.r, p.g, p.b, p.a };
			for ( size_t c = 0; c < actual.size(); ++c )
			{
				require( std::isfinite( actual[c] ), "non-finite clamped border output" );
				maximumError = std::max( maximumError, std::abs( actual[c] - expected[c] ) );
			}
		}
		require( maximumError < 0.00002, "clamped corner RGBA mismatch" );
		std::cout << "method=translation phase=0.25 shape=" << w << 'x' << h
			<< " border=max-displacement maxerror=" << maximumError << '\n';
	}
}
} // namespace

int main( int argc, char **argv )
{
	try
	{
		if ( argc == 1 ) { usage(); return 0; }
		bool list = false, self = false, validation = false;
		std::optional<DeviceUuid> presenter, renderer;
		for ( int i = 1; i < argc; ++i )
		{
			const std::string_view arg = argv[i];
			if ( arg == "--help" && argc == 2 ) { usage(); return 0; }
			if ( arg == "--list" && !list ) list = true;
			else if ( arg == "--self-test" && !self ) self = true;
			else if ( arg == "--validation" && !validation ) validation = true;
			else if ( arg == "--device-uuid" || arg == "--render-uuid" )
			{
				auto &destination = arg == "--device-uuid" ? presenter : renderer;
				require( !destination && i + 1 < argc, "duplicate UUID option or missing UUID value" );
				destination = parseUuid( argv[++i] );
			}
			else throw std::runtime_error( "unknown or duplicate option: " + std::string( arg ) );
		}
		if ( list )
		{
			require( argc == 2, "--list must be used alone" );
			for ( const Device &device : VulkanEngine::devices() ) printDevice( device );
			return 0;
		}
		require( self && presenter.has_value(), "GPU runs require --self-test and --device-uuid 32HEX" );
		require( !renderer || *renderer != *presenter, "presenter UUID equals render UUID; select the second GPU explicitly" );
		const auto devices = VulkanEngine::devices();
		const auto selected = std::find_if( devices.begin(), devices.end(), [&]( const Device &device ) { return device.uuid == *presenter; } );
		require( selected != devices.end(), "presenter UUID was not found" );
		require( !selected->software, "software device cannot establish GPU conformance" );
		if ( renderer )
			require( std::any_of( devices.begin(), devices.end(), [&]( const Device &device ) { return device.uuid == *renderer; } ), "render UUID was not found" );
		VulkanEngine owner( *presenter, FRAMEGEN_COMPUTE_SHADER, validation );
		auto &engine = bind_backend( owner, requirements_for( gsc_contract_generation() ) );
		printDevice( engine.device() );
		selfTest( engine );
		engine.shutdown();
		std::cout << "conformance PASS (synthetic float32 RGBA; no quality or performance claim)\n";
		return 0;
	}
	catch ( const std::exception &error )
	{
		std::cerr << "framegen-compute: " << error.what() << '\n';
		return 1;
	}
}
