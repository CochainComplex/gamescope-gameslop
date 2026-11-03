#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace gamescope::framegen::compute
{

using DeviceUuid = std::array<uint8_t, 16>;

struct Pixel
{
	float r, g, b, a;
};
static_assert( sizeof( Pixel ) == 16 );

enum class Method : uint32_t { Blend, Translation };
enum class Timeline : uint32_t { Bidir, Causal };

// Times describe content, not submission or completion. Integer subtraction
// precedes conversion, preserving short intervals at large monotonic times.
struct Pair
{
	uint64_t epoch;
	uint64_t previousId;
	uint64_t currentId;
	uint64_t previousNs;
	uint64_t currentNs;
	uint64_t targetNs;
	Timeline timeline = Timeline::Bidir;
};

inline std::optional<float> phase( const Pair &pair )
{
	if ( pair.timeline != Timeline::Bidir || pair.epoch == 0
		|| pair.previousId == 0 || pair.currentId <= pair.previousId
		|| pair.currentNs <= pair.previousNs || pair.targetNs <= pair.previousNs
		|| pair.targetNs >= pair.currentNs )
		return std::nullopt;
	const float value = float( double( pair.targetNs - pair.previousNs )
		/ double( pair.currentNs - pair.previousNs ) );
	// An integer-interior target can round to the endpoint in the shader ABI.
	if ( !( value > 0.0f && value < 1.0f ) ) return std::nullopt;
	return value;
}

// std430/push-constant ABI shared with tools/framegen-compute/generate.comp.
struct Parameters
{
	uint32_t width;
	uint32_t height;
	uint32_t method;
	float phase;
	float dx;
	float dy;
};
static_assert( sizeof( Parameters ) == 24 );
static_assert( offsetof( Parameters, phase ) == 12 );
static_assert( offsetof( Parameters, dx ) == 16 );
static_assert( offsetof( Parameters, dy ) == 20 );

struct Ticket
{
	uint64_t epoch = 0;
	uint64_t serial = 0;
	bool operator==( const Ticket & ) const = default;
};

// Deadline expiry cancels usefulness, not ownership. The GPU resource owner
// alone calls complete(), after observing actual completion. Neither discard
// nor reset may make an incomplete batch's storage reusable.
class Batch
{
public:
	[[nodiscard]] Ticket begin( uint64_t epoch )
	{
		if ( busy() || epoch == 0 || epoch != m_epoch || m_serial == UINT64_MAX )
			return {};
		m_ticket = { m_epoch, ++m_serial };
		m_complete = false;
		m_discarded = false;
		return m_ticket;
	}
	[[nodiscard]] bool owns( Ticket ticket ) const
	{
		return ticket.serial != 0 && ticket == m_ticket;
	}
	[[nodiscard]] bool busy() const { return m_ticket.serial != 0; }
	[[nodiscard]] bool ready( Ticket ticket ) const { return owns( ticket ) && m_complete; }
	[[nodiscard]] bool discarded( Ticket ticket ) const { return owns( ticket ) && m_discarded; }
	[[nodiscard]] uint64_t epoch() const { return m_epoch; }
	void complete( Ticket ticket ) { if ( owns( ticket ) ) m_complete = true; }
	void discard( Ticket ticket ) { if ( owns( ticket ) ) m_discarded = true; }
	[[nodiscard]] bool retire( Ticket ticket )
	{
		if ( !ready( ticket ) ) return false;
		m_ticket = {};
		return true;
	}
	[[nodiscard]] bool reset()
	{
		if ( busy() || m_epoch == UINT64_MAX ) return false;
		++m_epoch;
		return true;
	}
private:
	uint64_t m_epoch = 1;
	uint64_t m_serial = 0;
	Ticket m_ticket{};
	bool m_complete = false;
	bool m_discarded = false;
};

} // namespace gamescope::framegen::compute
