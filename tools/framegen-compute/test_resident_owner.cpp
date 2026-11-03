// Explicit GPU owner-transaction fixture, not an optimizer numerical baseline.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "resident_training.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>

using namespace gamescope::framegen::compute;

namespace
{
constexpr size_t P = 4644;
constexpr uint64_t Epoch = 0x7fc000017f800001ULL;
constexpr uint32_t Key = 0x100f0;

void require( bool condition, const char *message )
{
	if ( !condition ) throw std::runtime_error( message );
}

DeviceUuid parseUuid( std::string_view text )
{
	require( text.size() == 32, "device UUID requires 32 hexadecimal digits" );
	auto digit = []( char c ) -> uint8_t {
		if ( c >= '0' && c <= '9' ) return uint8_t( c - '0' );
		if ( c >= 'a' && c <= 'f' ) return uint8_t( c - 'a' + 10 );
		if ( c >= 'A' && c <= 'F' ) return uint8_t( c - 'A' + 10 );
		throw std::runtime_error( "nonhexadecimal UUID" );
	};
	DeviceUuid result{};
	for ( size_t i = 0; i < result.size(); ++i )
		result[i] = uint8_t( 16 * digit( text[2 * i] ) + digit( text[2 * i + 1] ) );
	require( result != DeviceUuid{}, "zero device UUID" );
	return result;
}

bool exact( const std::vector<float> &a, const std::vector<float> &b )
{
	return a.size() == b.size() && std::memcmp( a.data(), b.data(), a.size() * sizeof( float ) ) == 0;
}

template<typename Function> void refuses( Function function, const char *message )
{
	bool threw = false;
	try { function(); } catch ( const std::runtime_error & ) { threw = true; }
	require( threw, message );
}

void waitComplete( ResidentTrainer &trainer, Ticket ticket )
{
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds( 10 );
	for ( ;; )
	{
		const Status status = trainer.poll( ticket );
		if ( status == Status::Complete ) return;
		require( status == Status::Pending, "unexpected owner fixture status" );
		require( std::chrono::steady_clock::now() < deadline,
			"owner fixture timed out; GPU work still requires retirement" );
		std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
	}
}

void checkCheckpoint( const TrainingCheckpoint &snapshot, uint32_t step )
{
	require( snapshot.epoch == Epoch && snapshot.key == Key && snapshot.acceptedStep == step
		&& snapshot.values.size() == 5 * P, "checkpoint identity/progress/extent mismatch" );
}

void checkAdvance( const TrainingCheckpoint &before, const TrainingCheckpoint &after, uint32_t step )
{
	checkCheckpoint( after, step );
	for ( size_t row = 0; row < 5; ++row )
		for ( size_t i = 0; i < P; ++i )
		{
			const float expected = before.values[row * P + i] + float( step + row + 1 ) / 1024.0f;
			require( std::bit_cast<uint32_t>( after.values[row * P + i] ) == std::bit_cast<uint32_t>( expected ),
				"synthetic candidate publication used wrong accepted bank or optimizer step" );
		}
}

void checkReport( const TrainingReport &report, uint32_t step, uint32_t attempt, bool accepted )
{
	require( report.identity.epoch == Epoch && report.identity.key == Key && report.identity.step == step
		&& report.identity.attempt == attempt && report.identity.previousAcceptedStep == step - 1
		&& report.accepted == accepted && report.acceptedStep == ( accepted ? step : step - 1 )
		&& report.coverage == P, "report identity or acceptance mismatch" );
}

void run( DeviceUuid uuid, bool validation )
{
	std::vector<float> prior( P );
	for ( size_t i = 0; i < P; ++i ) prior[i] = .25f + float( i % 17 ) / 64.0f;
	std::vector<float> source( 16 * 16, .25f ), destination( 16 * 16, .375f );
	std::vector<float> forward( 4 * 16 * 16, 0.0f ), reverse( 4 * 16 * 16, 0.0f );
	for ( size_t i = 0; i < 16 * 16; ++i ) forward[4 * i + 2] = reverse[4 * i + 2] = .5f;
	source[0] = 0.0f;
	const TrainingSample sample{ source, destination, forward, reverse };
	ResidentTrainer trainer( uuid, Epoch, Key, 16, 16, prior, validation );
	const auto initial = trainer.checkpoint(); checkCheckpoint( initial, 0 );
	const Ticket first = trainer.submit( sample );
	require( first.serial != 0 && first.epoch == Epoch, "first owner submission rejected" );
	// A post-submit change must not alter the already snapshotted input.
	source[0] = 1.0f;
	require( trainer.submit( sample ).serial == 0, "busy owner accepted another job" );
	refuses( [&] { (void)trainer.checkpoint(); }, "checkpoint accepted outstanding ownership" );
	waitComplete( trainer, first );
	const auto firstReport = trainer.collect( first );
	require( firstReport.has_value(), "first report missing" ); checkReport( *firstReport, 1, 1, true );
	const auto saved = trainer.checkpoint(); checkAdvance( initial, saved, 1 );
	const auto independentlyOwned = saved.values;

	// Attempt2 deliberately contaminates v[last]. The entire candidate must
	// roll back and accepted step1 must remain checkpointable.
	const Ticket rejected = trainer.submit( sample );
	require( rejected.serial != 0, "rejection fixture submission failed" );
	require( trainer.poll( first ) == Status::Stale && !trainer.collect( first ), "stale accepted ticket changed current ownership" );
	waitComplete( trainer, rejected );
	const auto rejectedReport = trainer.collect( rejected );
	require( rejectedReport.has_value(), "valid rejected report was quarantined" );
	checkReport( *rejectedReport, 2, 2, false );
	require( rejectedReport->invalidState == 1 && rejectedReport->invalidServed == 0,
		"rejected candidate fault counts wrong" );
	const auto rolledBack = trainer.checkpoint(); checkCheckpoint( rolledBack, 1 );
	require( exact( rolledBack.values, saved.values ), "rejection changed accepted state" );
	require( exact( trainer.checkpoint().values, saved.values ), "checkpoint after rejection not stable" );

	source[0] = 0.0f;
	const Ticket third = trainer.submit( sample ); require( third.serial != 0, "rejected report poisoned healthy owner" );
	waitComplete( trainer, third );
	const auto thirdReport = trainer.collect( third );
	require( thirdReport.has_value(), "recovery report missing" ); checkReport( *thirdReport, 2, 3, true );
	const auto recovered = trainer.checkpoint(); checkAdvance( saved, recovered, 2 );

	// The test-only health transport corrupts exactly attempt4's epoch-low
	// report store. Publish refuses it; collect must quarantine the CPU owner.
	const Ticket fourth = trainer.submit( sample ); require( fourth.serial != 0, "quarantine fixture submission failed" );
	waitComplete( trainer, fourth );
	refuses( [&] { (void)trainer.collect( fourth ); }, "malformed report did not quarantine owner" );
	refuses( [&] { (void)trainer.poll( fourth ); }, "quarantined owner still polls" );
	refuses( [&] { (void)trainer.submit( sample ); }, "quarantined owner still submits" );
	refuses( [&] { (void)trainer.checkpoint(); }, "quarantined owner still exposes checkpoints" );
	require( exact( saved.values, independentlyOwned ), "earlier CPU checkpoint lost independent ownership" );
	trainer.shutdown(); // malformed report has already retired the engine ticket
	trainer.shutdown(); // finalization is idempotent
}
} // namespace

int main( int argc, char **argv )
{
	try
	{
		std::optional<DeviceUuid> uuid; bool validation = false;
		for ( int i = 1; i < argc; ++i )
		{
			const std::string_view option = argv[i];
			if ( option == "--device-uuid" && !uuid && i + 1 < argc ) uuid = parseUuid( argv[++i] );
			else if ( option == "--validation" && !validation ) validation = true;
			else throw std::runtime_error( "unknown, duplicate or incomplete option" );
		}
		require( uuid.has_value(), "usage: resident-owner-conformance --device-uuid 32HEX [--validation]" );
		run( *uuid, validation );
		std::cout << "resident owner conformance PASS: accept/reject/recover/quarantine; "
			"exact five-row checkpoint bytes; synthetic optimizer fixture, no numerical-training claim\n";
		return 0;
	}
	catch ( const std::exception &error )
	{
		std::cerr << "resident-owner-conformance: " << error.what() << '\n';
		return 1;
	}
}
