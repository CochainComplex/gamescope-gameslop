#include "resident_training.hpp"
#include "framegen/net_layout.hpp"
#include "framegen/numeric.hpp"
#include "framegen/training_dispatch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace gamescope::framegen::compute
{
namespace
{
constexpr size_t P = k_uFramegenNetFloats;
template<typename T> std::span<const std::byte> bytes( const T &value )
{
	return std::as_bytes( std::span( &value, 1 ) );
}
void require( bool condition, const char *message )
{
	if ( !condition ) throw std::runtime_error( message );
}
bool finite( std::span<const float> values )
{
	return std::all_of( values.begin(), values.end(), []( float f ) {
		return gamescope::framegen::is_finite_binary32( f ); } );
}
}

ResidentTrainer::ResidentTrainer( DeviceUuid presenter, uint64_t modelEpoch, uint32_t key,
	uint32_t width, uint32_t height, std::span<const float> prior, bool validation )
	: m_engine( presenter, FRAMEGEN_TRAIN_GRAD_SHADER, validation ), m_epoch( modelEpoch ),
	  m_key( key ), m_width( width ), m_height( height )
{
	require( modelEpoch != 0 && width >= 16 && height >= 16
		&& uint64_t( width ) * height <= ( 1u << 20 ), "invalid resident identity/shape" );
	require( prior.size() == P && finite( prior ), "invalid resident prior" );
	m_optimizer = m_engine.addKernel( FRAMEGEN_TRAIN_OPT_SHADER );
	m_health = m_engine.addKernel( FRAMEGEN_RESIDENT_HEALTH_SHADER );
	m_publish = m_engine.addKernel( FRAMEGEN_RESIDENT_PUBLISH_SHADER );
	m_input.resize( 10 * size_t( width ) * height + P );
	std::copy( prior.begin(), prior.end(), m_input.end() - P );
	m_initial.resize( GSC_REPORT_BUFFER_FLOATS, 0.0f );
	for ( size_t row : { size_t( 0 ), size_t( 3 ), size_t( 4 ), size_t( 5 ), size_t( 8 ), size_t( 9 ) } )
		std::copy( prior.begin(), prior.end(), m_initial.begin() + row * P );
	m_zeroGradients.resize( 16 * P, 0.0f );
}

ResidentTrainer::~ResidentTrainer() = default;

void ResidentTrainer::healthy() const
{
	require( !m_closed, "resident trainer is closed" );
	require( !m_failed, "resident trainer failed; only teardown is permitted" );
}

Ticket ResidentTrainer::submit( const TrainingSample &sample, uint32_t flags,
	uint32_t sceneCut, float lr, float ema, float decay )
{
	std::lock_guard lock( m_mutex );
	healthy();
	if ( m_ticket.serial ) return {};
	const size_t n = size_t( m_width ) * m_height;
	const std::array<std::span<const float>, 4> tensors{
		sample.source, sample.destination, sample.forward, sample.reverse };
	const std::array<size_t, 4> counts{ n, n, 4 * n, 4 * n };
	for ( size_t i = 0; i < tensors.size(); ++i )
		require( tensors[i].size() == counts[i] && finite( tensors[i] ), "invalid resident input tensor" );
	require( !( flags & ~5u ) && sceneCut <= 1, "unsupported resident training mode" );
	require( gamescope::framegen::is_finite_binary32( lr ) && lr >= 0
		&& gamescope::framegen::is_finite_binary32( ema ) && ema >= 0 && ema <= 1
		&& gamescope::framegen::is_finite_binary32( decay ) && decay >= 0 && decay <= 1,
		"invalid resident optimizer controls" );
	require( m_acceptedStep != UINT32_MAX && m_attempt != UINT32_MAX, "resident step/attempt exhausted" );
	const uint32_t step = m_acceptedStep + 1, attempt = m_attempt + 1;
	size_t offset = 0;
	for ( const auto values : tensors )
	{
		std::copy( values.begin(), values.end(), m_input.begin() + offset );
		offset += values.size();
	}
	const uint32_t seed = step * 0x9E3779B9u + 0x61C88647u;
	const TrainPush forward{ seed, 0, 0.0f, flags, 1.0f, m_width, m_height, sceneCut };
	const TrainPush reverse{ seed ^ 0x55555555u, 8, 0.0f, flags, 1.0f, m_width, m_height, sceneCut };
	const OptPush optimizer{ lr, ema, decay, step };
	const TrainingReportPush health{ uint32_t( m_epoch ), uint32_t( m_epoch >> 32 ),
		m_key, step, attempt, m_acceptedStep, 0, 0 };
	const std::array<BufferStep, 5> steps{{
		{ 0, bytes( forward ), 8 }, { 0, bytes( reverse ), 8 },
		{ m_optimizer, bytes( optimizer ), uint32_t( ( P + 63 ) / 64 ) },
		{ m_health, bytes( health ), 1 },
		{ m_publish, bytes( health ), uint32_t( ( P + 63 ) / 64 ) } }};
	try
	{
		const std::array<std::span<const float>, 3> initial{ m_input, m_zeroGradients, m_initial };
		const std::array<std::span<const float>, 3> updates{ m_input, {}, {} };
		m_engineTicket = m_initialized ? m_engine.updateBuffers( m_engine.epoch(), updates, steps )
			: m_engine.submitBuffers( m_engine.epoch(), initial, steps );
		require( m_engineTicket.serial != 0, "resident submission rejected" );
		m_initialized = true;
		m_attempt = attempt;
		m_expected = { m_epoch, m_key, step, attempt, m_acceptedStep };
		m_ticket = { m_epoch, m_engineTicket.serial };
		return m_ticket;
	}
	catch ( ... ) { m_failed = true; throw; }
}

Status ResidentTrainer::poll( Ticket ticket )
{
	std::lock_guard lock( m_mutex );
	healthy();
	if ( !ticket.serial || ticket != m_ticket ) return Status::Stale;
	try { return m_engine.poll( m_engineTicket ); }
	catch ( ... ) { m_failed = true; throw; }
}

std::optional<TrainingReport> ResidentTrainer::collect( Ticket ticket )
{
	std::lock_guard lock( m_mutex );
	healthy();
	if ( !ticket.serial || ticket != m_ticket ) return std::nullopt;
	try
	{
		TrainingReportWords words{};
		const std::array<BufferReadback, 3> output{{ {}, {},
			{ GSC_REPORT_WORD_OFFSET * sizeof( float ), std::as_writable_bytes( std::span( words ) ) } }};
		if ( !m_engine.collectBufferRanges( m_engineTicket, output ) ) return std::nullopt;
		m_ticket = m_engineTicket = {};
		auto report = decode_training_report( words, m_expected );
		require( report.has_value(), "malformed/stale resident report; context quarantined" );
		m_acceptedStep = report->acceptedStep;
		return report;
	}
	catch ( ... ) { m_failed = true; throw; }
}

TrainingCheckpoint ResidentTrainer::checkpoint( bool fullDiagnostics )
{
	std::lock_guard lock( m_mutex );
	healthy();
	require( !m_ticket.serial, "checkpoint requires an exclusively idle resident owner" );
	TrainingCheckpoint result{ m_epoch, m_key, m_acceptedStep, std::vector<float>( 5 * P ), {} };
	if ( fullDiagnostics ) result.gradients.resize( 16 * P );
	if ( !m_initialized )
	{
		std::copy_n( m_initial.begin() + GSC_REPORT_BACKUP_OFFSET, 5 * P, result.values.begin() );
		return result;
	}
	try
	{
		const std::array<BufferReadback, 3> output{{ {},
			{ 0, std::as_writable_bytes( std::span( result.gradients ) ) },
			{ GSC_REPORT_BACKUP_OFFSET * sizeof( float ), std::as_writable_bytes( std::span( result.values ) ) } }};
		require( m_engine.readIdleBuffers( output ), "resident checkpoint read rejected" );
		require( finite( result.values ), "nonfinite resident checkpoint" );
		return result;
	}
	catch ( ... ) { m_failed = true; throw; }
}

void ResidentTrainer::shutdown()
{
	std::lock_guard lock( m_mutex );
	if ( m_closed ) return;
	require( !m_ticket.serial, "collect resident work before explicit shutdown" );
	try { m_engine.shutdown(); m_closed = true; }
	catch ( ... ) { m_failed = true; throw; }
}
} // namespace gamescope::framegen::compute
