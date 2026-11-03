// Keep this standalone contract active in release builds too.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "framegen/training_report.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

using namespace gamescope::framegen::compute;

namespace
{
// Independent public-wire fixture: no decoder indexes/constants construct it.
constexpr TrainingReportIdentity Identity{ 0x123456789abcdef0ULL, 0x100f0u, 17u, 44u, 16u };

TrainingReportWords healthy()
{
	return { 0x47535452u, 1u, 4644u, 4644u, 0u, 0u, 7u,
		0x3f800000u, 0x3e800000u, 0x9abcdef0u, 0x12345678u,
		0x100f0u, 17u, 44u, 17u, 1u, 0u, 0u, 0u, 0u };
}

TrainingReportWords rejected()
{
	auto words = healthy();
	words[4] = 1;
	words[14] = 16;
	words[15] = 0;
	return words;
}

void expectRejected( const TrainingReportWords &words, const TrainingReportIdentity &identity = Identity )
{
	assert( !decode_training_report( words, identity ).has_value() );
}

void wireContract()
{
	static_assert( sizeof( TrainingReportWords ) == 80 );
	static_assert( std::is_standard_layout_v<TrainingReportPush> );
	static_assert( sizeof( TrainingReportPush ) == 32 );
	static_assert( offsetof( TrainingReportPush, epochLo ) == 0 );
	static_assert( offsetof( TrainingReportPush, epochHi ) == 4 );
	static_assert( offsetof( TrainingReportPush, key ) == 8 );
	static_assert( offsetof( TrainingReportPush, step ) == 12 );
	static_assert( offsetof( TrainingReportPush, attempt ) == 16 );
	static_assert( offsetof( TrainingReportPush, previousAcceptedStep ) == 20 );
	static_assert( offsetof( TrainingReportPush, reserved0 ) == 24 );
	static_assert( offsetof( TrainingReportPush, reserved1 ) == 28 );

	const auto result = decode_training_report( healthy(), Identity );
	assert( result && result->accepted );
	assert( result->identity.epoch == Identity.epoch && result->identity.key == Identity.key );
	assert( result->identity.step == 17 && result->identity.attempt == 44 );
	assert( result->identity.previousAcceptedStep == 16 );
	assert( result->coverage == 4644 && result->invalidState == 0 && result->invalidServed == 0 );
	assert( result->changedCount == 7 && result->acceptedStep == 17 );
	assert( result->maxAbsServed == 1.0f && result->maxDelta == 0.25f );

	const auto badCandidate = decode_training_report( rejected(), Identity );
	assert( badCandidate && !badCandidate->accepted && badCandidate->acceptedStep == 16 );
	assert( badCandidate->invalidState == 1 );
}

void missingPartialAndUnknownReport()
{
	expectRejected( TrainingReportWords{} );
	for ( uint32_t coverage : { 0u, 1u, 4643u, 4645u, UINT32_MAX } )
	{
		auto words = healthy(); words[3] = coverage; expectRejected( words );
	}
	for ( auto replacement : { std::pair{ 0u, 0u }, std::pair{ 1u, 0u },
		std::pair{ 1u, 2u }, std::pair{ 2u, 4643u }, std::pair{ 2u, 4645u } } )
	{
		auto words = healthy(); words[replacement.first] = replacement.second; expectRejected( words );
	}
	for ( size_t reserved = 16; reserved < 20; ++reserved )
	{
		auto words = healthy(); words[reserved] = 1; expectRejected( words );
	}
}

void identityAndProgress()
{
	// Each identity component independently prevents replay under another job.
	for ( size_t index : { 9u, 10u, 11u, 12u, 13u } )
	{
		auto words = healthy(); words[index] ^= 1u; expectRejected( words );
	}
	for ( auto identity : {
		TrainingReportIdentity{ 0u, Identity.key, 17u, 44u, 16u },
		TrainingReportIdentity{ Identity.epoch, Identity.key, 0u, 44u, 16u },
		TrainingReportIdentity{ Identity.epoch, Identity.key, 17u, 0u, 16u },
		TrainingReportIdentity{ Identity.epoch, Identity.key, 17u, 44u, 15u },
		TrainingReportIdentity{ Identity.epoch, Identity.key, 17u, 44u, UINT32_MAX } } )
	{
		auto words = healthy();
		words[9] = uint32_t( identity.epoch ); words[10] = uint32_t( identity.epoch >> 32 );
		words[12] = words[14] = identity.step; words[13] = identity.attempt;
		expectRejected( words, identity );
	}

	for ( uint32_t previous : { 0u, UINT32_MAX - 1u } )
	{
		TrainingReportIdentity identity = Identity;
		identity.previousAcceptedStep = previous;
		identity.step = previous + 1u;
		auto words = healthy(); words[12] = words[14] = identity.step;
		assert( decode_training_report( words, identity ).has_value() );
	}
	for ( uint32_t accepted : { 0u, 16u, 18u, UINT32_MAX } )
	{
		auto words = healthy(); words[14] = accepted; expectRejected( words );
	}
	for ( uint32_t accepted : { 0u, 15u, 17u, UINT32_MAX } )
	{
		auto words = rejected(); words[14] = accepted; expectRejected( words );
	}
}

void candidateVerdictAndCountBounds()
{
	for ( size_t field : { 4u, 5u } )
	{
		auto words = healthy(); words[field] = 1; expectRejected( words );
		words[14] = 16; words[15] = 0;
		const auto result = decode_training_report( words, Identity );
		assert( result && !result->accepted );
	}
	auto words = rejected(); words[4] = 0; expectRejected( words );
	words = healthy(); words[15] = 2; expectRejected( words );
	words = rejected(); words[4] = 4u * 4644u; words[5] = 4644u;
	assert( decode_training_report( words, Identity ).has_value() );
	words[4]++; expectRejected( words );
	words = rejected(); words[5] = 4645u; expectRejected( words );
	for ( uint32_t changed : { 0u, 4644u } )
	{
		words = healthy(); words[6] = changed;
		assert( decode_training_report( words, Identity ).has_value() );
	}
	words = healthy(); words[6] = 4645u; expectRejected( words );
	words = healthy(); words[4] = UINT32_MAX; expectRejected( words );
}

void ieeeMetricBits()
{
	for ( size_t field : { 7u, 8u } )
	{
		// Signed zeros, subnormal and max finite are not corruption. In
		// particular diagnostic max-delta saturation may produce FLT_MAX.
		for ( uint32_t bits : { 0u, 0x80000000u, 1u, 0x007fffffu, 0x00800000u, 0x7f7fffffu } )
		{
			auto words = healthy(); words[field] = bits;
			assert( decode_training_report( words, Identity ).has_value() );
		}
		// Explicit encodings avoid host floating-point classification and
		// exercise the same contract under -ffast-math.
		for ( uint32_t bits : { 0x7f800000u, 0xff800000u, 0x7fc00001u,
			0xffc00001u, 0x7f800001u, 0xff800001u, 0xbf800000u, 0x80000001u } )
		{
			auto words = healthy(); words[field] = bits; expectRejected( words );
		}
	}
}
} // namespace

int main()
{
	wireContract();
	missingPartialAndUnknownReport();
	identityAndProgress();
	candidateVerdictAndCountBounds();
	ieeeMetricBits();
	std::cout << "training report CPU contracts PASS\n";
	return 0;
}
