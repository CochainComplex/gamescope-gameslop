// Explicit GPU conformance probe; never an automatic CPU Meson test.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "vulkan_engine.hpp"
#include "framegen/training_report.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

using namespace gamescope::framegen::compute;

namespace
{
constexpr size_t P = 4644, Bank = 5 * P, ReportAt = 10 * P;
static_assert( GSC_REPORT_BUFFER_FLOATS == 10 * P + 20 );
static_assert( GSC_REPORT_WORD_OFFSET == ReportAt );

void require( bool value, const char *message )
{
	if ( !value ) throw std::runtime_error( message );
}

DeviceUuid parseUuid( std::string_view text )
{
	require( text.size() == 32, "UUID requires exactly 32 hexadecimal digits" );
	auto digit = []( char c ) -> uint8_t {
		if ( c >= '0' && c <= '9' ) return uint8_t( c - '0' );
		if ( c >= 'a' && c <= 'f' ) return uint8_t( c - 'a' + 10 );
		if ( c >= 'A' && c <= 'F' ) return uint8_t( c - 'A' + 10 );
		throw std::runtime_error( "nonhexadecimal UUID" );
	};
	DeviceUuid result{};
	for ( size_t i = 0; i < result.size(); ++i )
		result[i] = uint8_t( 16 * digit( text[2 * i] ) + digit( text[2 * i + 1] ) );
	require( result != DeviceUuid{}, "zero UUID is not a device identity" );
	return result;
}

std::vector<uint32_t> banks()
{
	std::vector<uint32_t> words( GSC_REPORT_BUFFER_FLOATS, 0xdeadbeefu );
	for ( size_t row = 0; row < 5; ++row )
		for ( size_t i = 0; i < P; ++i )
		{
			words[row * P + i] = std::bit_cast<uint32_t>( .25f + float( row ) + float( i % 31 ) / 64.0f );
			words[Bank + row * P + i] = std::bit_cast<uint32_t>( -.5f - float( row ) - float( i % 17 ) / 32.0f );
		}
	return words;
}

TrainingReportWords healthyReport( const TrainingReportIdentity &identity )
{
	return { 0x47535452u, 1u, 4644u, 4644u, 0u, 0u, 4644u,
		0x40800000u, 0x41000000u, uint32_t( identity.epoch ), uint32_t( identity.epoch >> 32 ),
		identity.key, identity.step, identity.attempt, identity.step, 1u, 0u, 0u, 0u, 0u };
}

void waitComplete( VulkanEngine &engine, Ticket ticket )
{
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 10 );
	for ( ;; )
	{
		const Status status = engine.poll( ticket );
		if ( status == Status::Complete ) return;
		require( status == Status::Pending, "unexpected health ticket status" );
		require( std::chrono::steady_clock::now() < deadline,
			"health observation timed out; owner teardown must still drain work" );
		std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
	}
}

class Probe
{
public:
	VulkanEngine engine;
	uint32_t publish;
	// Both halves resemble NaN encodings as floats: report identity is raw uint.
	TrainingReportIdentity identity{ 0x7fc000017f800001ULL, 0x100f0u, 1u, 1u, 0u };
	size_t cases = 0;

	Probe( DeviceUuid uuid, bool validation )
		: engine( uuid, FRAMEGEN_RESIDENT_HEALTH_SHADER, validation ),
		  publish( engine.addKernel( FRAMEGEN_RESIDENT_PUBLISH_SHADER ) )
	{
		require( !engine.device().software && engine.device().uuid == uuid, "explicit hardware GPU required" );
	}

	TrainingReportPush push() const
	{
		return { uint32_t( identity.epoch ), uint32_t( identity.epoch >> 32 ), identity.key,
			identity.step, identity.attempt, identity.previousAcceptedStep, 0, 0 };
	}

	std::vector<uint32_t> run( const std::vector<uint32_t> &words, bool scan, bool engineContracts = false )
	{
		const auto parameters = push();
		const auto constants = std::as_bytes( std::span( &parameters, 1 ) );
		const std::array<BufferStep, 2> allSteps{{ { 0, constants, 1 }, { publish, constants, 73 } }};
		const auto steps = scan ? std::span<const BufferStep>( allSteps ) : std::span<const BufferStep>( allSteps ).subspan( 1 );
		std::vector<float> transport( words.size() );
		// No conversion through floating arithmetic may canonicalize fault bits.
		std::memcpy( transport.data(), words.data(), words.size() * sizeof( uint32_t ) );
		const std::array<float, 1> dummy{ 0.0f };
		const std::array<std::span<const float>, 3> initial{ dummy, dummy, transport };
		if ( engineContracts )
			require( engine.updateBuffers( engine.epoch(), initial, steps ).serial == 0,
				"resident update accepted before allocation bootstrap" );
		const Ticket ticket = engine.submitBuffers( engine.epoch(), initial, steps );
		require( ticket.serial != 0, "health submission rejected" );
		std::vector<uint32_t> output( words.size(), 0x12345678u );
		const std::array<BufferReadback, 3> ranges{{ {}, {},
			{ 0, std::as_writable_bytes( std::span( output ) ) } }};
		if ( engineContracts )
		{
			require( !engine.readIdleBuffers( ranges ), "idle read accepted outstanding ticket" );
			waitComplete( engine, ticket );
			require( !engine.readIdleBuffers( ranges ), "idle read accepted completed unretired ticket" );
			uint32_t firstCanary = 0x11223344u, lastCanary = 0x55667788u;
			const std::array<BufferReadback, 3> invalid{{
				{ 0, std::as_writable_bytes( std::span( &firstCanary, 1 ) ) }, {},
				{ std::numeric_limits<size_t>::max(), std::as_writable_bytes( std::span( &lastCanary, 1 ) ) } }};
			require( !engine.collectBufferRanges( ticket, invalid ), "out-of-bounds collection accepted" );
			require( firstCanary == 0x11223344u && lastCanary == 0x55667788u,
				"failed range validation clobbered caller output" );
		}
		waitComplete( engine, ticket );
		require( engine.collectBufferRanges( ticket, ranges ), "valid ticket became uncollectible" );
		if ( engineContracts )
		{
			std::vector<float> resized( transport.size() + 1, 0.0f );
			const std::array<std::span<const float>, 3> changedShape{
				std::span<const float>{}, std::span<const float>{}, resized };
			require( engine.updateBuffers( engine.epoch(), changedShape, steps ).serial == 0,
				"preserved allocation silently changed shape" );
			std::vector<uint32_t> after( output.size() );
			const std::array<BufferReadback, 3> idle{{ {}, {},
				{ 0, std::as_writable_bytes( std::span( after ) ) } }};
			require( engine.readIdleBuffers( idle ) && after == output,
				"failed resident update changed bank bytes" );
			TrainingReportWords reportOnly{};
			const std::array<BufferReadback, 3> reportRange{{ {}, {},
				{ ReportAt * sizeof( uint32_t ), std::as_writable_bytes( std::span( reportOnly ) ) } }};
			require( engine.readIdleBuffers( reportRange )
				&& std::equal( reportOnly.begin(), reportOnly.end(), output.begin() + ReportAt ),
				"compact offset read changed report words" );
		}
		++cases;
		return output;
	}

	TrainingReport inspect( const std::vector<uint32_t> &before,
		const std::vector<uint32_t> &after, bool accepted ) const
	{
		TrainingReportWords raw{};
		std::copy_n( after.begin() + ReportAt, raw.size(), raw.begin() );
		const auto report = decode_training_report( raw, identity );
		require( report.has_value() && report->accepted == accepted, "GPU report malformed or wrong verdict" );
		require( report->coverage == P, "health scan did not cover all parameters" );
		const auto expected = accepted ? before.begin() : before.begin() + Bank;
		require( std::equal( after.begin(), after.begin() + Bank, expected ), "candidate bank publish/rollback mismatch" );
		require( std::equal( after.begin() + Bank, after.begin() + 2 * Bank, expected ),
			"accepted bank changed on rejection or did not publish full candidate" );
		return *report;
	}
};

void healthyAndFaultCases( Probe &probe )
{
	const auto baseline = banks();
	auto output = probe.run( baseline, true, true );
	auto report = probe.inspect( baseline, output, true );
	require( report.invalidState == 0 && report.invalidServed == 0 && report.changedCount == P,
		"healthy representation/count diagnostics wrong" );
	require( report.maxAbsServed == 4.71875f && report.maxDelta == 9.71875f,
		"healthy full-bank maxima wrong" );
	for ( size_t row = 0; row < 5; ++row )
		for ( size_t index : { 0u, 63u, 64u, 2047u, 2048u, 4095u, 4096u, 4643u } )
			for ( uint32_t bits : { 0x7f800000u, 0xff800000u, 0x7fc00001u, 0xff800001u } )
			{
				auto words = baseline; words[row * P + index] = bits;
				report = probe.inspect( words, probe.run( words, true ), false );
				require( report.invalidState == ( row < 4 ? 1u : 0u )
					&& report.invalidServed == ( row == 4 ? 1u : 0u ), "fault scan missed or double-counted parameter" );
			}
	// One all-corrupt candidate proves the full diagnostic count, not just tail coverage.
	auto words = baseline;
	std::fill_n( words.begin(), Bank, 0x7fc00001u );
	report = probe.inspect( words, probe.run( words, true ), false );
	require( report.invalidState == 4 * P && report.invalidServed == P, "full corruption counts wrong" );
	for ( uint32_t bits : { 0u, 0x80000000u, 1u, 0x80000001u, 0x7f7fffffu, 0xff7fffffu } )
	{
		words = baseline; std::fill_n( words.begin(), Bank, bits );
		report = probe.inspect( words, probe.run( words, true ), true );
		require( report.invalidState == 0 && report.invalidServed == 0
			&& std::bit_cast<uint32_t>( report.maxAbsServed ) == ( bits & 0x7fffffffu ),
			"finite extrema or signed/subnormal representation rejected" );
	}
	words = baseline;
	std::fill_n( words.begin(), Bank, 0x7f7fffffu );
	std::fill_n( words.begin() + Bank, Bank, 0xff7fffffu );
	report = probe.inspect( words, probe.run( words, true ), true );
	require( std::bit_cast<uint32_t>( report.maxDelta ) == 0x7f7fffffu,
		"finite subtraction overflow did not saturate diagnostics" );
	// A fresh healthy job after the corruption campaign must remain serviceable.
	probe.inspect( baseline, probe.run( baseline, true ), true );
}

void malformedPublicationCases( Probe &probe )
{
	const auto good = healthyReport( probe.identity );
	require( decode_training_report( good, probe.identity ).has_value(), "independent report fixture invalid" );
	std::vector<TrainingReportWords> invalid{ TrainingReportWords{} };
	for ( auto [index, value] : {
		std::pair{ 0u, 0u }, std::pair{ 1u, 2u }, std::pair{ 2u, 4643u },
		std::pair{ 3u, 4643u }, std::pair{ 3u, 4645u }, std::pair{ 4u, 18577u },
		std::pair{ 5u, 4645u }, std::pair{ 6u, 4645u }, std::pair{ 7u, 0x7f800000u },
		std::pair{ 8u, 0x7f800001u }, std::pair{ 9u, 1u }, std::pair{ 10u, 1u },
		std::pair{ 11u, 0xf0u }, std::pair{ 12u, 2u }, std::pair{ 13u, 2u },
		std::pair{ 14u, 0u }, std::pair{ 15u, 2u }, std::pair{ 16u, 1u },
		std::pair{ 17u, 1u }, std::pair{ 18u, 1u }, std::pair{ 19u, 1u } } )
	{
		auto raw = good; raw[index] = value; invalid.push_back( raw );
	}
	// Inconsistent zero-bad-count rejection and bad-count acceptance are both malformed.
	auto raw = good; raw[14] = 0; raw[15] = 0; invalid.push_back( raw );
	raw = good; raw[4] = 1; invalid.push_back( raw );
	for ( const auto &bad : invalid )
	{
		require( !decode_training_report( bad, probe.identity ).has_value(), "malformed fixture accepted by CPU decoder" );
		auto words = banks(); std::copy( bad.begin(), bad.end(), words.begin() + ReportAt );
		const auto output = probe.run( words, false );
		require( std::equal( output.begin(), output.begin() + Bank, words.begin() + Bank )
			&& std::equal( output.begin() + Bank, output.begin() + 2 * Bank, words.begin() + Bank ),
			"malformed report did not roll back every state row" );
	}
}
} // namespace

int main( int argc, char **argv )
{
	try
	{
		std::optional<DeviceUuid> uuid;
		bool validation = false;
		for ( int i = 1; i < argc; ++i )
		{
			const std::string_view arg = argv[i];
			if ( arg == "--device-uuid" && !uuid && i + 1 < argc ) uuid = parseUuid( argv[++i] );
			else if ( arg == "--validation" && !validation ) validation = true;
			else throw std::runtime_error( "unknown, duplicate or incomplete option" );
		}
		require( uuid.has_value(), "usage: resident-health-conformance --device-uuid 32HEX [--validation]" );
		Probe probe( *uuid, validation );
		healthyAndFaultCases( probe );
		malformedPublicationCases( probe );
		probe.engine.shutdown();
		std::cout << "resident health conformance PASS: " << probe.cases
			<< " cases; 160 single-parameter faults across all five rows; full-bank rollback/publication; "
			"malformed-report refusal; finite extrema; resident engine ownership\n";
		return 0;
	}
	catch ( const std::exception &error )
	{
		std::cerr << "resident-health-conformance: " << error.what() << '\n';
		return 1;
	}
}
