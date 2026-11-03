#include "resident_training.hpp"
#include "backend_bindings.hpp"
#include "training_fixture.hpp"
#include "framegen/net_layout.hpp"
#include "framegen/numeric.hpp"
#include "framegen/training_dispatch.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string_view>

using namespace gamescope::framegen::compute;

namespace
{
constexpr size_t P = k_uFramegenNetFloats;
constexpr uint64_t ModelEpoch = 0x7fc0123480000001ull;
using Clock = std::chrono::steady_clock;

void require( bool value, const char *message )
{
	if ( !value ) throw std::runtime_error( message );
}

uint32_t number( std::string_view text )
{
	require( !text.empty() && text.find_first_not_of( "0123456789" ) == std::string_view::npos,
		"expected unsigned decimal integer" );
	const auto value = std::stoull( std::string( text ) );
	require( value <= UINT32_MAX, "integer exceeds UINT32_MAX" );
	return uint32_t( value );
}

DeviceUuid parseUuid( std::string_view text )
{
	require( text.size() == 32, "device UUID must contain exactly 32 hexadecimal digits" );
	auto digit = []( char c ) -> uint8_t {
		if ( c >= '0' && c <= '9' ) return uint8_t( c - '0' );
		if ( c >= 'a' && c <= 'f' ) return uint8_t( c - 'a' + 10 );
		if ( c >= 'A' && c <= 'F' ) return uint8_t( c - 'A' + 10 );
		throw std::runtime_error( "malformed device UUID" );
	};
	DeviceUuid uuid{};
	for ( size_t i = 0; i < uuid.size(); ++i ) uuid[i] = uint8_t( 16 * digit( text[2 * i] ) + digit( text[2 * i + 1] ) );
	require( uuid != DeviceUuid{}, "zero UUID is not a device identity" );
	return uuid;
}

struct Options
{
	std::optional<DeviceUuid> uuid;
	std::optional<uint32_t> key;
	std::string input, directory;
	uint32_t steps = 100, warmup = 10;
	bool full = false, reference = false, validation = false, directInterface = false;
};

Options options( int argc, char **argv )
{
	Options result;
	std::vector<std::string> seen;
	for ( int i = 1; i < argc; ++i )
	{
		const std::string arg = argv[i];
		require( std::find( seen.begin(), seen.end(), arg ) == seen.end(), "duplicate option" );
		seen.push_back( arg );
		if ( arg == "--full-diagnostics" ) result.full = true;
		else if ( arg == "--check-reference" ) result.reference = true;
		else if ( arg == "--validation" ) result.validation = true;
		else if ( arg == "--direct-interface" ) result.directInterface = true;
		else
		{
			require( i + 1 < argc, "missing option value" );
			const char *value = argv[++i];
			if ( arg == "--device-uuid" ) result.uuid = parseUuid( value );
			else if ( arg == "--input" ) result.input = value;
			else if ( arg == "--output-dir" ) result.directory = value;
			else if ( arg == "--input-key" ) result.key = number( value );
			else if ( arg == "--steps" ) result.steps = number( value );
			else if ( arg == "--warmup" ) result.warmup = number( value );
			else throw std::runtime_error( "unknown option: " + arg );
		}
	}
	require( result.uuid && result.key && !result.input.empty() && !result.directory.empty() && result.steps > 0,
		"usage: resident-training --device-uuid 32HEX --input GSCT --output-dir DIR --input-key U32 "
		"[--steps 100] [--warmup 10] [--full-diagnostics] [--check-reference] [--validation] [--direct-interface]" );
	require( uint64_t( result.steps ) + result.warmup <= UINT32_MAX, "combined step/attempt count exceeds UINT32_MAX" );
	return result;
}

double milliseconds( Clock::time_point begin, Clock::time_point end )
{
	return std::chrono::duration<double, std::milli>( end - begin ).count();
}

template<typename Owner> void waitComplete( Owner &owner, Ticket ticket )
{
	const auto deadline = Clock::now() + std::chrono::seconds( 10 );
	for ( ;; )
	{
		const Status status = owner.poll( ticket );
		if ( status == Status::Complete ) return;
		require( status == Status::Pending, "unexpected completion status" );
		require( Clock::now() < deadline, "completion timed out; GPU ownership persists until teardown" );
	}
}

template<typename T> std::span<const std::byte> bytes( const T &value )
{
	return std::as_bytes( std::span( &value, 1 ) );
}

void checkCheckpoint( const TrainingCheckpoint &checkpoint, uint32_t key, uint32_t step )
{
	require( checkpoint.epoch == ModelEpoch && checkpoint.key == key && checkpoint.acceptedStep == step
		&& checkpoint.values.size() == 5 * P, "checkpoint identity/extent mismatch" );
	for ( float value : checkpoint.values )
		require( gamescope::framegen::is_finite_binary32( value ), "nonfinite checkpoint" );
}

bool exact( std::span<const float> a, std::span<const float> b )
{
	return a.size() == b.size() && std::memcmp( a.data(), b.data(), a.size_bytes() ) == 0;
}

bool special( uint32_t step )
{
	return step == 1 || step == 7 || step == 1023 || step == 1024 || step == 1025;
}

void ensureNew( const std::filesystem::path &path )
{
	require( std::filesystem::symlink_status( path ).type() == std::filesystem::file_type::not_found,
		"output artifact already exists" );
}

void writeValues( const std::filesystem::path &path, std::span<const float> values )
{
	static_assert( std::endian::native == std::endian::little );
	// The model was validated by checkpoint(); raw gradient diagnostics may
	// legitimately contain nonfinite values which the optimizer sanitized.
	FILE *file = std::fopen( path.c_str(), "wbx" );
	require( file != nullptr, "cannot exclusively create output artifact" );
	const bool written = std::fwrite( values.data(), sizeof( float ), values.size(), file ) == values.size();
	const bool closed = std::fclose( file ) == 0;
	require( written && closed, "cannot finish output artifact" );
}

void writeCheckpoint( const std::filesystem::path &path, std::span<const float> values,
	uint32_t key, uint32_t step )
{
	writeValues( path, values );
	const std::string identity = "{\"epoch\":" + std::to_string( ModelEpoch )
		+ ",\"key\":" + std::to_string( key ) + ",\"accepted_step\":" + std::to_string( step )
		+ ",\"model_version\":3,\"parameters\":" + std::to_string( P ) + "}\n";
	FILE *file = std::fopen( ( path.string() + ".json" ).c_str(), "wbx" );
	require( file != nullptr, "cannot exclusively create checkpoint identity" );
	const bool written = std::fwrite( identity.data(), 1, identity.size(), file ) == identity.size();
	const bool closed = std::fclose( file ) == 0;
	require( written && closed, "cannot finish checkpoint identity" );
}

struct Row
{
	bool warmup;
	uint32_t step;
	TrainingReport report;
	double submit, wait, collect, snapshot, total;
};
} // namespace

int main( int argc, char **argv )
{
	try
	{
		const Options option = options( argc, argv );
		const Fixture fixture = readFixture( option.input );
		const size_t n = size_t( fixture.width ) * fixture.height;
		require( fixture.step == 1 && fixture.state.size() == 5 * P && fixture.input.size() == 10 * n + P,
			"resident replay requires an initial step-1 fixture" );
		const std::span<const float> input( fixture.input );
		const auto prior = input.subspan( 10 * n, P );
		require( exact( std::span<const float>( fixture.state ).subspan( 0, P ), prior )
			&& exact( std::span<const float>( fixture.state ).subspan( 3 * P, P ), prior ),
			"initial fast/EMA must equal prior exactly" );
		for ( size_t i = P; i < 3 * P; ++i )
			require( std::bit_cast<uint32_t>( fixture.state[i] ) == 0u, "initial Adam m/v must be positive zero" );
		for ( size_t i = 4 * P; i < 5 * P; ++i )
			require( std::bit_cast<uint32_t>( fixture.state[i] ) == 0u, "initial served-output row must be empty" );
		const TrainingSample sample{ input.subspan( 0, n ), input.subspan( n, n ),
			input.subspan( 2 * n, 4 * n ), input.subspan( 6 * n, 4 * n ) };
		const uint32_t count = option.warmup + option.steps;
		const std::filesystem::path directory( option.directory );
		if ( std::filesystem::exists( directory ) ) require( std::filesystem::is_directory( directory ), "output-dir is not a directory" );
		for ( uint32_t step : { 1u, 7u, 1023u, 1024u, 1025u } )
			if ( step <= count )
			{
				const auto path = directory / ( "step-" + std::to_string( step ) + ".checkpoint" );
				ensureNew( path ); ensureNew( path.string() + ".json" );
			}
		ensureNew( directory / "final.checkpoint" );
		ensureNew( directory / "final.checkpoint.json" );
		if ( option.full ) ensureNew( directory / "final.gradients" );
		std::vector<Row> rows;
		rows.reserve( count );
		std::map<uint32_t, std::vector<float>> snapshots;
		std::vector<float> finalValues, finalGradients;
		const auto setupStart = Clock::now();
		ResidentTrainer owner( *option.uuid, ModelEpoch, *option.key, fixture.width, fixture.height, prior, option.validation );
		ResidentTrainer &trainer = option.directInterface ? owner :
			bind_backend( owner, requirements_for( gsc_contract_resident_training() ) );
		std::unique_ptr<VulkanEngine> reference;
		uint32_t referenceOptimizer = 0;
		std::vector<float> referenceState = fixture.state, referenceGradients( 16 * P ), zeroGradients( 16 * P, 0.0f );
		if ( option.reference )
		{
			reference = std::make_unique<VulkanEngine>( *option.uuid, FRAMEGEN_TRAIN_GRAD_SHADER, option.validation );
			require( !reference->device().software, "software device cannot establish GPU conformance" );
			referenceOptimizer = reference->addKernel( FRAMEGEN_TRAIN_OPT_SHADER );
		}
		const double setupMs = milliseconds( setupStart, Clock::now() );
		Ticket stale{};
		for ( uint64_t iteration = 1; iteration <= count; ++iteration )
		{
			const uint32_t step = uint32_t( iteration );
			const auto begin = Clock::now();
			const Ticket ticket = trainer.submit( sample, fixture.flags, fixture.sceneCut, fixture.lr, fixture.ema, fixture.decay );
			const auto submitted = Clock::now();
			require( ticket.epoch == ModelEpoch && ticket.serial != 0, "invalid resident public ticket" );
			if ( option.reference )
			{
				require( trainer.submit( sample, fixture.flags, fixture.sceneCut, fixture.lr, fixture.ema, fixture.decay ).serial == 0,
					"resident busy admission accepted a second submission" );
				bool checkpointRejected = false;
				try { (void)trainer.checkpoint(); } catch ( const std::runtime_error & ) { checkpointRejected = true; }
				require( checkpointRejected, "resident checkpoint accepted active ownership" );
				if ( stale.serial ) require( trainer.poll( stale ) == Status::Stale && !trainer.collect( stale ), "stale ticket affected current job" );
			}
			waitComplete( trainer, ticket );
			const auto completed = Clock::now();
			const auto report = trainer.collect( ticket );
			const auto collected = Clock::now();
			require( report && report->identity.epoch == ModelEpoch && report->identity.key == *option.key
				&& report->identity.step == step && report->identity.attempt == step,
				"missing/stale resident report" );
			require( report->accepted && report->acceptedStep == step, "normal resident fixture rejected; refusing optimizer evolution" );
			std::optional<TrainingCheckpoint> checkpoint;
			if ( option.full ) checkpoint = trainer.checkpoint( true );
			const auto snapshotted = Clock::now();
			rows.push_back( { step <= option.warmup, step, *report, milliseconds( begin, submitted ),
				milliseconds( submitted, completed ), milliseconds( completed, collected ),
				option.full ? milliseconds( collected, snapshotted ) : 0.0, milliseconds( begin, snapshotted ) } );
			// Explicit checkpoints and reference dispatches below are excluded from
			// compact resident job timings. Reference mode is conformance only.
			if ( option.reference || special( step ) || step == count )
			{
				if ( !checkpoint || ( option.reference && checkpoint->gradients.empty() ) )
					checkpoint = trainer.checkpoint( option.reference || ( option.full && step == count ) );
				checkCheckpoint( *checkpoint, *option.key, step );
				if ( special( step ) ) snapshots.emplace( step, checkpoint->values );
				if ( step == count )
				{
					finalValues = checkpoint->values;
					if ( option.full ) finalGradients = checkpoint->gradients;
				}
			}
			if ( reference )
			{
				const uint32_t seed = step * 0x9E3779B9u + 0x61C88647u;
				const TrainPush forward{ seed, 0, 0.0f, fixture.flags, 1.0f, fixture.width, fixture.height, fixture.sceneCut };
				const TrainPush reverse{ seed ^ 0x55555555u, 8, 0.0f, fixture.flags, 1.0f, fixture.width, fixture.height, fixture.sceneCut };
				const OptPush opt{ fixture.lr, fixture.ema, fixture.decay, step };
				const std::array<BufferStep, 3> steps{{ { 0, bytes( forward ), 8 }, { 0, bytes( reverse ), 8 },
					{ referenceOptimizer, bytes( opt ), uint32_t( ( P + 63 ) / 64 ) } }};
				const std::array<std::span<const float>, 3> initial{ fixture.input, zeroGradients, referenceState };
				const Ticket baseline = reference->submitBuffers( reference->epoch(), initial, steps );
				require( baseline.serial != 0, "reference submission rejected" );
				waitComplete( *reference, baseline );
				const std::array<std::span<float>, 3> output{ std::span<float>{}, referenceGradients, referenceState };
				require( reference->collectBuffers( baseline, output ), "reference collection failed" );
				require( checkpoint->gradients.size() == 16 * P && exact( checkpoint->values, referenceState )
					&& exact( checkpoint->gradients, referenceGradients ), "resident/reference bit-exact multi-step mismatch" );
			}
			stale = ticket;
		}
		const auto teardownStart = Clock::now();
		trainer.shutdown();
		if ( reference ) reference->shutdown();
		const double teardownMs = milliseconds( teardownStart, Clock::now() );
		std::filesystem::create_directories( directory );
		for ( const auto &[step, values] : snapshots )
			writeCheckpoint( directory / ( "step-" + std::to_string( step ) + ".checkpoint" ), values, *option.key, step );
		writeCheckpoint( directory / "final.checkpoint", finalValues, *option.key, count );
		if ( option.full ) writeValues( directory / "final.gradients", finalGradients );
		std::cerr << std::setprecision( 12 ) << "setup_ms=" << setupMs << " first_step_ms=" << rows.front().total
			<< " teardown_ms=" << teardownMs << " report_bytes=80 flags=" << fixture.flags
			<< " check_reference=" << option.reference << " validation=" << option.validation
			<< " direct_interface=" << option.directInterface
			<< " full_diagnostics=" << option.full << " warmup=" << option.warmup << " measured=" << option.steps
			<< " scope=" << ( option.reference ? "offline-conformance" : "offline-evolving-resident-replay" )
			<< " no_live_integration_or_quality_claim=1\n";
		std::cout << "phase,step,accepted,changed_count,invalid_state,invalid_served,max_abs,max_delta,submit_ms,wait_ms,collect_ms,snapshot_ms,total_ms,epoch,key,attempt,accepted_step\n"
			<< std::setprecision( 12 );
		for ( const Row &row : rows )
			std::cout << ( row.warmup ? "warmup" : "measured" ) << ',' << row.step << ',' << row.report.accepted << ','
				<< row.report.changedCount << ',' << row.report.invalidState << ',' << row.report.invalidServed << ','
				<< row.report.maxAbsServed << ',' << row.report.maxDelta << ',' << row.submit << ',' << row.wait << ','
				<< row.collect << ',' << row.snapshot << ',' << row.total << ',' << row.report.identity.epoch
				<< ',' << row.report.identity.key << ',' << row.report.identity.attempt << ',' << row.report.acceptedStep << '\n';
		std::cout.flush();
		require( bool( std::cout ), "CSV output failed" );
		return 0;
	}
	catch ( const std::exception &error ) { std::cerr << "resident-training: " << error.what() << '\n'; return 1; }
}
