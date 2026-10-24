#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace gamescope::framegen
{

enum class NetLearningSkip : uint8_t
{
	NoSkip, NoBatch, History, Busy, Cadence, NoSlot, Budget, OutputPool, ColorProbe,
	Reuse, Pipeline, Motion, NetUnavailable, Resources, Every, Count
};

inline const char *net_learning_skip_name( NetLearningSkip reason )
{
	static constexpr std::array names = { "none", "no_batch", "history", "gpu_busy",
		"cadence", "no_slot", "budget", "output_pool", "color_probe", "reuse",
		"pipeline", "motion", "net_unavailable", "resources", "net_every" };
	static_assert( names.size() == static_cast<size_t>( NetLearningSkip::Count ) );
	const auto index = static_cast<size_t>( reason );
	return index < names.size() ? names[index] : "unknown";
}

struct NetLearningSkips
{
	std::array<uint64_t, static_cast<size_t>( NetLearningSkip::Count )> counts = {};
	void add( NetLearningSkip reason )
	{
		if ( reason != NetLearningSkip::NoSkip && reason != NetLearningSkip::Count )
			counts[ static_cast<size_t>( reason ) ]++;
	}
	NetLearningSkips &operator+=( const NetLearningSkips &other )
	{
		for ( size_t i = 0; i < counts.size(); i++ )
			counts[i] += other.counts[i];
		return *this;
	}
	NetLearningSkip dominant() const
	{
		// Stable enum-order tie break; an empty window has no skip reason.
		return static_cast<NetLearningSkip>( std::max_element( counts.begin(), counts.end() ) - counts.begin() );
	}
};

// One skip per recorded real, finalized at the next real or shutdown.
// Deepest gate (enum order) wins; a step overrides every skip/refill.
struct NetLearningPair
{
	uint64_t realId = 0;
	bool stepped = false;
	NetLearningSkip reason = NetLearningSkip::NoBatch;

	NetLearningSkip finish()
	{
		const auto result = realId != 0 && !stepped ? reason : NetLearningSkip::NoSkip;
		realId = 0;
		return result;
	}
	NetLearningSkip begin( uint64_t id )
	{
		if ( id == 0 || id == realId )
			return NetLearningSkip::NoSkip;
		const auto result = finish();
		realId = id;
		stepped = false;
		reason = NetLearningSkip::NoBatch;
		return result;
	}
	void skip( NetLearningSkip value )
	{
		if ( value > reason && value < NetLearningSkip::Count )
			reason = value;
	}
	void step() { stepped = true; }
};

// Fixed-width histogram over nanosecond samples, reported in milliseconds.
//
// StepNs is the bucket width and Buckets the bucket count, so full scale is
// StepNs * Buckets. add() clamps out-of-range samples into the last bucket and
// percentile() returns the upper edge of the bucket the rank lands in, which
// means a distribution fed samples past full scale reports exactly full scale
// for its high percentiles with no way to tell that apart from a real value.
// Size the range for what the field measures: sub-vblank timing accuracy is
// fine at 16 ms, wall-clock flip intervals at 60 Hz are >= one vblank by
// construction and need 64 ms, and real-frame -> flip latency includes a
// delayed real's whole bidir delay and needs 256 ms (the three ranges below).
template < uint64_t StepNs, size_t Buckets >
struct MetricsDistribution_t
{
	static_assert( StepNs != 0u, "bucket width must be nonzero" );
	static_assert( Buckets != 0u, "bucket count must be nonzero" );

	static constexpr uint64_t k_ulStepNs = StepNs;
	static constexpr size_t k_nBuckets = Buckets;
	// Upper edge of the last bucket, i.e. what a saturating percentile prints.
	static constexpr double k_flFullScaleMs =
		(double)( StepNs * (uint64_t)Buckets ) / 1.0e6;

	uint64_t n = 0;
	double sum = 0.0;
	double sumSquares = 0.0;
	double min = std::numeric_limits<double>::max();
	double max = 0.0;
	std::array<uint64_t, Buckets> histogram = {};

	void add( uint64_t ulNs )
	{
		const double flMs = ulNs / 1.0e6;
		n++;
		sum += flMs;
		sumSquares += flMs * flMs;
		min = std::min( min, flMs );
		max = std::max( max, flMs );
		const size_t nBucket = std::min<size_t>( ulNs / StepNs, Buckets - 1 );
		histogram[ nBucket ]++;
	}

	double average() const { return n != 0 ? sum / n : 0.0; }
	double stddev() const
	{
		return n != 0 ? std::sqrt( std::max( 0.0,
			sumSquares / n - average() * average() ) ) : 0.0;
	}
	double percentile( uint32_t nPercent ) const
	{
		if ( n == 0 )
			return 0.0;
		const uint64_t nRank = ( n * nPercent + 99u ) / 100u;
		uint64_t nSeen = 0;
		for ( size_t i = 0; i < histogram.size(); i++ )
		{
			nSeen += histogram[ i ];
			if ( nSeen >= nRank )
				return ( i + 1 ) * StepNs / 1.0e6;
		}
		return k_flFullScaleMs;
	}
	double p50() const { return percentile( 50u ); }
	double p95() const { return percentile( 95u ); }
};

// Bucket width of the timing and flip-interval distributions: 250 us resolves
// the sub-millisecond structure of GPU pass and deadline-error timings.
inline constexpr uint64_t k_ulMetricsHistogramStepNs = 250'000ull;
// 16 ms full scale. Only for quantities that cannot plausibly run past it.
inline constexpr size_t k_nMetricsHistogramBuckets = 64;
// 64 ms full scale at the same resolution, for wall-clock flip intervals: a
// 60 Hz flip interval alone is 16.67 ms.
inline constexpr size_t k_nMetricsLatencyHistogramBuckets = 256;
// 256 ms full scale at 1 ms, for real-frame -> flip latency: a delayed real
// waits one source interval plus any backlog before its flip, which passes 64 ms
// at 30-45 fps, and 1 ms resolves that delay against a source interval.
inline constexpr uint64_t k_ulMetricsSourceLatencyStepNs = 1'000'000ull;
inline constexpr size_t k_nMetricsSourceLatencyBuckets = 256;

using MetricsDistribution =
	MetricsDistribution_t< k_ulMetricsHistogramStepNs, k_nMetricsHistogramBuckets >;
using MetricsLatencyDistribution =
	MetricsDistribution_t< k_ulMetricsHistogramStepNs, k_nMetricsLatencyHistogramBuckets >;
using MetricsSourceLatencyDistribution =
	MetricsDistribution_t< k_ulMetricsSourceLatencyStepNs, k_nMetricsSourceLatencyBuckets >;

} // namespace gamescope::framegen
