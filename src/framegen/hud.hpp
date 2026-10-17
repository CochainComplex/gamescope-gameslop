#pragma once

#include "types.hpp"
#include "gpu_diagnostics.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace gamescope::framegen
{

// Canonical high-intensity CGA cyan/magenta, mixed as visible ordered pixels.
// Keep this Bayer order in sync with cs_framegen_hud.comp.
inline constexpr std::array<uint32_t, 16> k_uFramegenHudCgaBayer = {
	0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5
};
[[nodiscard]] constexpr uint32_t framegen_hud_cga_color(
	uint32_t x, uint32_t y, uint32_t height )
{
	const uint32_t step = std::min( 16u, y * 16u / ( height > 1u ? height - 1u : 1u ) );
	return k_uFramegenHudCgaBayer[( y % 4u ) * 4u + x % 4u] < step
		? 0xff55ffu : 0x55ffffu;
}

inline constexpr uint32_t k_uFramegenHudMaxLines = 8u;
inline constexpr uint32_t k_uFramegenHudMaxColumns = 72u;
inline constexpr uint32_t k_uFramegenHudPackedTextWords =
	( k_uFramegenHudMaxLines * k_uFramegenHudMaxColumns ) / 4u;
inline constexpr uint32_t k_uFramegenHudDeviceNameColumns = 19u;
inline constexpr size_t k_nFramegenHudSparklineSamples = 12u;

[[nodiscard]] constexpr uint64_t font8x8_glyph(
	uint8_t row0, uint8_t row1, uint8_t row2, uint8_t row3,
	uint8_t row4, uint8_t row5, uint8_t row6, uint8_t row7 )
{
	return static_cast<uint64_t>( row0 )
		| static_cast<uint64_t>( row1 ) << 8u
		| static_cast<uint64_t>( row2 ) << 16u
		| static_cast<uint64_t>( row3 ) << 24u
		| static_cast<uint64_t>( row4 ) << 32u
		| static_cast<uint64_t>( row5 ) << 40u
		| static_cast<uint64_t>( row6 ) << 48u
		| static_cast<uint64_t>( row7 ) << 56u;
}

// Daniel Hepper's font8x8_basic, based on IBM's public-domain VGA font.
// Public domain; source rows are stored bottom-to-top and flipped by the HUD
// shader. https://github.com/dhepper/font8x8/blob/master/font8x8_basic.h
inline constexpr uint64_t k_uFramegenHudFont8x8[128] = {
	0,
	font8x8_glyph( 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ), // spark 1/8
	font8x8_glyph( 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ), // spark 2/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00 ), // spark 3/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00 ), // spark 4/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00 ), // spark 5/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00 ), // spark 6/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00 ), // spark 7/8
	font8x8_glyph( 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF ), // spark 8/8
	0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, // U+0020 space
	font8x8_glyph( 0x00, 0x18, 0x00, 0x18, 0x18, 0x3C, 0x3C, 0x18 ), // !
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x36 ), // "
	font8x8_glyph( 0x00, 0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36 ), // #
	font8x8_glyph( 0x00, 0x0C, 0x1F, 0x30, 0x1E, 0x03, 0x3E, 0x0C ), // $
	font8x8_glyph( 0x00, 0x63, 0x66, 0x0C, 0x18, 0x33, 0x63, 0x00 ), // %
	font8x8_glyph( 0x00, 0x6E, 0x33, 0x3B, 0x6E, 0x1C, 0x36, 0x1C ), // &
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x06, 0x06 ), // '
	font8x8_glyph( 0x00, 0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18 ), // (
	font8x8_glyph( 0x00, 0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06 ), // )
	font8x8_glyph( 0x00, 0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00 ), // *
	font8x8_glyph( 0x00, 0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00 ), // +
	font8x8_glyph( 0x06, 0x0C, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00 ), // ,
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00 ), // -
	font8x8_glyph( 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00 ), // .
	font8x8_glyph( 0x00, 0x01, 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60 ), // /
	font8x8_glyph( 0x00, 0x3E, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x3E ), // 0
	font8x8_glyph( 0x00, 0x3F, 0x0C, 0x0C, 0x0C, 0x0C, 0x0E, 0x0C ), // 1
	font8x8_glyph( 0x00, 0x3F, 0x33, 0x06, 0x1C, 0x30, 0x33, 0x1E ), // 2
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E ), // 3
	font8x8_glyph( 0x00, 0x78, 0x30, 0x7F, 0x33, 0x36, 0x3C, 0x38 ), // 4
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x30, 0x30, 0x1F, 0x03, 0x3F ), // 5
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x33, 0x1F, 0x03, 0x06, 0x1C ), // 6
	font8x8_glyph( 0x00, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x33, 0x3F ), // 7
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E ), // 8
	font8x8_glyph( 0x00, 0x0E, 0x18, 0x30, 0x3E, 0x33, 0x33, 0x1E ), // 9
	font8x8_glyph( 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00 ), // :
	font8x8_glyph( 0x06, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00 ), // ;
	font8x8_glyph( 0x00, 0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18 ), // <
	font8x8_glyph( 0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00 ), // =
	font8x8_glyph( 0x00, 0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06 ), // >
	font8x8_glyph( 0x00, 0x0C, 0x00, 0x0C, 0x18, 0x30, 0x33, 0x1E ), // ?
	font8x8_glyph( 0x00, 0x1E, 0x03, 0x7B, 0x7B, 0x7B, 0x63, 0x3E ), // @
	font8x8_glyph( 0x00, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x1E, 0x0C ), // A
	font8x8_glyph( 0x00, 0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F ), // B
	font8x8_glyph( 0x00, 0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C ), // C
	font8x8_glyph( 0x00, 0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F ), // D
	font8x8_glyph( 0x00, 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F ), // E
	font8x8_glyph( 0x00, 0x0F, 0x06, 0x16, 0x1E, 0x16, 0x46, 0x7F ), // F
	font8x8_glyph( 0x00, 0x7C, 0x66, 0x73, 0x03, 0x03, 0x66, 0x3C ), // G
	font8x8_glyph( 0x00, 0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33 ), // H
	font8x8_glyph( 0x00, 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E ), // I
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x33, 0x30, 0x30, 0x30, 0x78 ), // J
	font8x8_glyph( 0x00, 0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67 ), // K
	font8x8_glyph( 0x00, 0x7F, 0x66, 0x46, 0x06, 0x06, 0x06, 0x0F ), // L
	font8x8_glyph( 0x00, 0x63, 0x63, 0x6B, 0x7F, 0x7F, 0x77, 0x63 ), // M
	font8x8_glyph( 0x00, 0x63, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x63 ), // N
	font8x8_glyph( 0x00, 0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C ), // O
	font8x8_glyph( 0x00, 0x0F, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3F ), // P
	font8x8_glyph( 0x00, 0x38, 0x1E, 0x3B, 0x33, 0x33, 0x33, 0x1E ), // Q
	font8x8_glyph( 0x00, 0x67, 0x66, 0x36, 0x3E, 0x66, 0x66, 0x3F ), // R
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x38, 0x0E, 0x07, 0x33, 0x1E ), // S
	font8x8_glyph( 0x00, 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x2D, 0x3F ), // T
	font8x8_glyph( 0x00, 0x3F, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33 ), // U
	font8x8_glyph( 0x00, 0x0C, 0x1E, 0x33, 0x33, 0x33, 0x33, 0x33 ), // V
	font8x8_glyph( 0x00, 0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63 ), // W
	font8x8_glyph( 0x00, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x63 ), // X
	font8x8_glyph( 0x00, 0x1E, 0x0C, 0x0C, 0x1E, 0x33, 0x33, 0x33 ), // Y
	font8x8_glyph( 0x00, 0x7F, 0x66, 0x4C, 0x18, 0x31, 0x63, 0x7F ), // Z
	font8x8_glyph( 0x00, 0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E ), // [
	font8x8_glyph( 0x00, 0x40, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03 ), // backslash
	font8x8_glyph( 0x00, 0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E ), // ]
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x63, 0x36, 0x1C, 0x08 ), // ^
	font8x8_glyph( 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 ), // _
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x0C, 0x0C ), // `
	font8x8_glyph( 0x00, 0x6E, 0x33, 0x3E, 0x30, 0x1E, 0x00, 0x00 ), // a
	font8x8_glyph( 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x07 ), // b
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00, 0x00 ), // c
	font8x8_glyph( 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x30, 0x38 ), // d
	font8x8_glyph( 0x00, 0x1E, 0x03, 0x3F, 0x33, 0x1E, 0x00, 0x00 ), // e
	font8x8_glyph( 0x00, 0x0F, 0x06, 0x06, 0x0F, 0x06, 0x36, 0x1C ), // f
	font8x8_glyph( 0x1F, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00, 0x00 ), // g
	font8x8_glyph( 0x00, 0x67, 0x66, 0x66, 0x6E, 0x36, 0x06, 0x07 ), // h
	font8x8_glyph( 0x00, 0x1E, 0x0C, 0x0C, 0x0C, 0x0E, 0x00, 0x0C ), // i
	font8x8_glyph( 0x1E, 0x33, 0x33, 0x30, 0x30, 0x30, 0x00, 0x30 ), // j
	font8x8_glyph( 0x00, 0x67, 0x36, 0x1E, 0x36, 0x66, 0x06, 0x07 ), // k
	font8x8_glyph( 0x00, 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0E ), // l
	font8x8_glyph( 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x33, 0x00, 0x00 ), // m
	font8x8_glyph( 0x00, 0x33, 0x33, 0x33, 0x33, 0x1F, 0x00, 0x00 ), // n
	font8x8_glyph( 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00, 0x00 ), // o
	font8x8_glyph( 0x0F, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00, 0x00 ), // p
	font8x8_glyph( 0x78, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00, 0x00 ), // q
	font8x8_glyph( 0x00, 0x0F, 0x06, 0x66, 0x6E, 0x3B, 0x00, 0x00 ), // r
	font8x8_glyph( 0x00, 0x1F, 0x30, 0x1E, 0x03, 0x3E, 0x00, 0x00 ), // s
	font8x8_glyph( 0x00, 0x18, 0x2C, 0x0C, 0x0C, 0x3E, 0x0C, 0x08 ), // t
	font8x8_glyph( 0x00, 0x6E, 0x33, 0x33, 0x33, 0x33, 0x00, 0x00 ), // u
	font8x8_glyph( 0x00, 0x0C, 0x1E, 0x33, 0x33, 0x33, 0x00, 0x00 ), // v
	font8x8_glyph( 0x00, 0x36, 0x7F, 0x7F, 0x6B, 0x63, 0x00, 0x00 ), // w
	font8x8_glyph( 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00, 0x00 ), // x
	font8x8_glyph( 0x1F, 0x30, 0x3E, 0x33, 0x33, 0x33, 0x00, 0x00 ), // y
	font8x8_glyph( 0x00, 0x3F, 0x26, 0x0C, 0x19, 0x3F, 0x00, 0x00 ), // z
	font8x8_glyph( 0x00, 0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38 ), // {
	font8x8_glyph( 0x00, 0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18 ), // |
	font8x8_glyph( 0x00, 0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07 ), // }
	font8x8_glyph( 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3B, 0x6E ), // ~
	0, // U+007f delete
};
static_assert( std::size( k_uFramegenHudFont8x8 ) == 128u );

using FramegenHudSparkline_t =
	std::array<char, k_nFramegenHudSparklineSamples + 1u>;

[[nodiscard]] inline FramegenHudSparkline_t framegen_hud_sparkline(
	const std::array<double, k_nFramegenHudSparklineSamples> &values,
	double scale )
{
	FramegenHudSparkline_t result = {};
	for ( size_t i = 0u; i < values.size(); i++ )
	{
		const double bounded = scale > 0.0 && values[i] > 0.0
			? std::min( values[i], scale ) : 0.0;
		const uint32_t level = scale > 0.0
			? static_cast<uint32_t>( bounded * 7.0 / scale + 0.5 ) : 0u;
		result[i] = static_cast<char>( 0x01u + std::min( level, 7u ) );
	}
	return result;
}

struct FramegenHudSnapshot_t
{
	const char *version = "unknown";
	const char *deviceName = "unknown";
	const char *otherDeviceName = nullptr;
	const char *renderOrigin = nullptr;
	GpuDiagnostics gpu;
	GamescopeFramegenMode mode = GamescopeFramegenMode::Extrapolate;
	GamescopeFramegenPipeline pipeline = GamescopeFramegenPipeline::Warp;
	uint32_t multiplier = 2u;
	uint32_t refreshMilliHz = 0u;
	bool vrrRequested = false;
	bool vrrActive = false;
	bool bidirRequested = false;
	bool bidirActive = false;
	bool baseLayer = false;
	bool clientBuffersStaged = false;
	bool netRequested = false;
	bool netActive = false;
	bool netOnline = false;
	bool adapt = false;
	uint64_t real = 0u;
	uint64_t delayedReal = 0u;
	uint64_t generated = 0u;
	uint64_t repeats = 0u;
	uint32_t ladderSteps = 0u;
	uint32_t ladderMaxSteps = 0u;
	uint32_t ladderHold = 0u;
	uint32_t ladderRecoveryStreak = 0u;
	uint32_t ladderRecoveryBackoffDecisions = 0u;
	uint32_t ladderRecoveryProbationRemaining = 0u;
	uint32_t ladderRecoveryDecisionsSinceClimb = 0u;
	int32_t biasTenthsMs = 0;
	uint32_t deadlineHitPercent = 0u;
	uint32_t pacingSdTenthsMs = 0u;
	std::array<double, k_nFramegenHudSparklineSamples> deadlineHitHistory = {};
	std::array<double, k_nFramegenHudSparklineSamples> pacingSdMsHistory = {};
	uint64_t resets = 0u;
	uint64_t ringResets = 0u;
	uint64_t netTrainedSteps = 0u;
	bool netProfileLoaded = false;
};

struct FramegenHudText_t
{
	uint32_t lineCount = 0u;
	std::array<std::array<char, k_uFramegenHudMaxColumns + 1u>, k_uFramegenHudMaxLines> lines = {};

	bool operator==( const FramegenHudText_t &other ) const = default;
};

inline void framegen_hud_append( char *dst, size_t capacity, const char *text )
{
	const size_t used = std::strlen( dst );
	if ( used + std::strlen( text ) >= capacity )
		return;
	std::snprintf( dst + used, capacity - used, "%s", text );
}

inline void framegen_hud_trim_device_name( char *dst, size_t capacity,
	const char *deviceName )
{
	if ( capacity == 0u )
		return;
	const char *begin = deviceName != nullptr ? deviceName : "";
	while ( *begin == ' ' || *begin == '\t' )
		begin++;
	const char *end = begin + std::strlen( begin );
	while ( end > begin && ( end[-1] == ' ' || end[-1] == '\t' ) )
		end--;
	size_t length = std::min<size_t>(
		static_cast<size_t>( end - begin ),
		std::min<size_t>( capacity - 1u, k_uFramegenHudDeviceNameColumns ) );
	while ( length > 0u && ( begin[length - 1u] == ' ' || begin[length - 1u] == '\t' ) )
		length--;
	if ( length != 0u )
		std::memcpy( dst, begin, length );
	dst[length] = '\0';
	if ( length == 0u )
		std::snprintf( dst, capacity, "unknown" );
}

inline void framegen_hud_add_line( FramegenHudText_t &text, const char *line )
{
	if ( line == nullptr || *line == '\0' || text.lineCount >= k_uFramegenHudMaxLines )
		return;
	std::snprintf( text.lines[text.lineCount].data(),
		text.lines[text.lineCount].size(), "%s", line );
	text.lineCount++;
}

inline void framegen_hud_format_steps( char *dst, size_t capacity, uint64_t steps )
{
	if ( steps < 1'000u )
		std::snprintf( dst, capacity, "%llu", static_cast<unsigned long long>( steps ) );
	// Branch on the ROUNDED value: 999'950..999'999 round up to 1000.0k, which
	// belongs in the "m" branch.
	else if ( steps + 50u < 1'000'000u )
	{
		const uint64_t tenths = ( steps + 50u ) / 100u;
		std::snprintf( dst, capacity, "%llu.%lluk",
			static_cast<unsigned long long>( tenths / 10u ),
			static_cast<unsigned long long>( tenths % 10u ) );
	}
	else
	{
		const uint64_t tenths = ( steps + 50'000u ) / 100'000u;
		std::snprintf( dst, capacity, "%llu.%llum",
			static_cast<unsigned long long>( tenths / 10u ),
			static_cast<unsigned long long>( tenths % 10u ) );
	}
}

[[nodiscard]] inline const char *framegen_hud_ladder_state(
	char *dst, size_t capacity, const FramegenHudSnapshot_t &snapshot )
{
	if ( snapshot.ladderSteps == 0u )
		return "full";
	if ( snapshot.ladderHold > 0u )
		return "hold";
	if ( snapshot.ladderRecoveryProbationRemaining > 0u )
	{
		const uint32_t seconds =
			( snapshot.ladderRecoveryProbationRemaining + 44u ) / 45u;
		std::snprintf( dst, capacity, "probation %us", seconds );
		return dst;
	}
	if ( snapshot.ladderRecoveryDecisionsSinceClimb
		< snapshot.ladderRecoveryBackoffDecisions )
	{
		const uint32_t decisions = snapshot.ladderRecoveryBackoffDecisions
			- snapshot.ladderRecoveryDecisionsSinceClimb;
		const uint32_t seconds = ( decisions + 44u ) / 45u;
		std::snprintf( dst, capacity, "recover in %us", seconds );
		return dst;
	}
	return "watching";
}

[[nodiscard]] inline FramegenHudText_t format_framegen_hud_detailed(
	uint32_t level, const FramegenHudSnapshot_t &snapshot )
{
	FramegenHudText_t result;
	if ( level == 0u )
		return result;

	const uint32_t refreshHz = std::min( 999u,
		( snapshot.refreshMilliHz + 500u ) / 1'000u );
	const uint32_t gameRate = static_cast<uint32_t>( std::min<uint64_t>(
		999u, ( snapshot.real + snapshot.delayedReal ) / 5u ) );
	const uint32_t generatedRate = static_cast<uint32_t>( std::min<uint64_t>(
		999u, snapshot.generated / 5u ) );
	const uint32_t repeatRate = static_cast<uint32_t>( std::min<uint64_t>(
		999u, snapshot.repeats / 5u ) );
	// Fill rate: the share of presents that carried fresh content (real,
	// delayed-real or generated) rather than a repeat of the previous scanout.
	// Identical definition to the framegen-metrics "fill=" field so the HUD and
	// the log can be cross-checked against each other.
	const uint64_t fresh = snapshot.real + snapshot.delayedReal + snapshot.generated;
	const uint64_t presented = fresh + snapshot.repeats;
	const uint32_t fillPercent = presented != 0u
		? static_cast<uint32_t>( ( fresh * 100u + presented / 2u ) / presented ) : 0u;

	char line[128] = {};
	std::snprintf( line, sizeof( line ),
		"%-8s %-21.21s%s x%u . %s . %uHz %s", "gameslop",
		snapshot.version != nullptr ? snapshot.version : "unknown",
		mode_name( snapshot.mode ), std::clamp( snapshot.multiplier, 2u, 4u ),
		pipeline_name( snapshot.pipeline ), refreshHz,
		snapshot.vrrActive ? "VRR" : "fixed" );
	framegen_hud_add_line( result, line );

	const auto &gpu=snapshot.gpu;
	const GpuCard *present=gpu.presenter<gpu.cards.size() ? &gpu.cards[gpu.presenter] : nullptr;
	const GpuCard *render=gpu.renderer<gpu.cards.size() ? &gpu.cards[gpu.renderer] : nullptr;
	const char *renderName=render ? render->name.c_str() : "unknown";
	// Old snapshots without hardware inventory retain a clearly tentative hint.
	if (gpu.cards.empty() && snapshot.renderOrigin) {
		if (!std::strcmp(snapshot.renderOrigin,"linear"))
			renderName=snapshot.clientBuffersStaged && snapshot.otherDeviceName
				? snapshot.otherDeviceName : "unknown (linear buffer)";
		else renderName=snapshot.renderOrigin;
	}
	std::string presentName=present ? present->name : (snapshot.deviceName ? snapshot.deviceName : "unknown");
	const auto begin=presentName.find_first_not_of(" \t");
	presentName=begin==std::string::npos ? "unknown" : presentName.substr(begin,presentName.find_last_not_of(" \t")-begin+1);
	std::snprintf(line,sizeof(line),"%s: %.44s @ %.12s",
		gpu.scope.rfind("Nested",0)==0 ? "COMPOSE" : "PRESENT",
		presentName.c_str(),
		present && !present->pci.empty() ? present->pci.c_str() : "unknown");
	framegen_hud_add_line(result,line);
	std::snprintf(line,sizeof(line),"RENDER?: %.34s @ %.12s [%s]",renderName,
		render && !render->pci.empty() ? render->pci.c_str() : "unknown",
		snapshot.clientBuffersStaged ? "staged" : "import");
	framegen_hud_add_line(result,line);

	const char *bidirState = snapshot.bidirActive ? "on"
		: snapshot.bidirRequested ? "requested(OFF)" : "off";
	const char *netState = snapshot.netActive
		? ( snapshot.netOnline ? "online" : "blob" )
		: snapshot.netRequested ? "requested(OFF)" : "off";
	std::snprintf( line, sizeof( line ),
		"BIDIR: %s | BASE: %s | NET: %s | ADAPT: %s",
		bidirState, snapshot.baseLayer ? "on" : "off", netState,
		snapshot.adapt ? "on" : "off" );
	if ( snapshot.vrrRequested && !snapshot.vrrActive )
	{
		constexpr const char *pszVrrFallback = " | VRR: requested(off)";
		if ( std::strlen( line ) + std::strlen( pszVrrFallback )
			<= k_uFramegenHudMaxColumns )
		{
			framegen_hud_append( line, sizeof( line ), pszVrrFallback );
		}
		else
		{
			// Conflicting requests are the information to preserve when all four
			// ordinary state fields and the VRR fallback cannot fit together.
			std::snprintf( line, sizeof( line ),
				"%-8s bidir:%s  net:%s  vrr:requested(off)",
				"modes", bidirState, netState );
		}
	}
	framegen_hud_add_line( result, line );

	char realSteps[24] = {};
	char generatedSteps[24] = {};
	framegen_hud_format_steps( realSteps, sizeof( realSteps ), snapshot.real );
	framegen_hud_format_steps(
		generatedSteps, sizeof( generatedSteps ), snapshot.generated );
	std::snprintf( line, sizeof( line ),
		"SOURCE: %ufps[%s] | GEN: %ufps[%s] | REPEAT: %ufps | FILL: %u%%", gameRate, realSteps, generatedRate, generatedSteps,
		repeatRate, fillPercent );
	if (std::strlen(line)>k_uFramegenHudMaxColumns)
		std::snprintf(line,sizeof(line),"SOURCE: %ufps | GEN: %ufps | REPEAT: %ufps | FILL: %u%%",
			gameRate,generatedRate,repeatRate,fillPercent);
	framegen_hud_add_line( result, line );

	char ladderState[24] = {};
	const char *ladderMode = snapshot.mode == GamescopeFramegenMode::Motion
		? "motion" : "extrapolate";
	std::snprintf( line, sizeof( line ), "QUALITY: %s/%s rung %u/%u | STATE: %s",
		pipeline_name( snapshot.pipeline ), ladderMode,
		snapshot.ladderSteps, snapshot.ladderMaxSteps,
		framegen_hud_ladder_state( ladderState, sizeof( ladderState ), snapshot ) );
	framegen_hud_add_line( result, line );

	if ( level < 2u )
		return result;

	const uint32_t sdTenths = std::min( snapshot.pacingSdTenthsMs, 9'999u );
	const FramegenHudSparkline_t hitSparkline = framegen_hud_sparkline(
		snapshot.deadlineHitHistory, 100.0 );
	// Flip-interval standard deviation uses a fixed 0..4 ms scale; noisier
	// windows saturate at the full-height glyph.
	const FramegenHudSparkline_t jitterSparkline = framegen_hud_sparkline(
		snapshot.pacingSdMsHistory, 4.0 );
	std::snprintf( line, sizeof( line ),
		"HIT: %u%% %s | JITTER: +/-%u.%ums %s",
		std::min( snapshot.deadlineHitPercent, 100u ),
		hitSparkline.data(),
		sdTenths / 10u, sdTenths % 10u,
		jitterSparkline.data() );
	framegen_hud_add_line( result, line );

	if ( snapshot.netActive )
	{
		char steps[24] = {};
		framegen_hud_format_steps( steps, sizeof( steps ), snapshot.netTrainedSteps );
		std::snprintf( line, sizeof( line ),
			"LEARN: %s | STEPS: %s | PROFILE: %s",
			snapshot.netOnline ? "online" : "offline", steps,
			snapshot.netProfileLoaded ? "loaded" : "none" );
		framegen_hud_add_line( result, line );
	}

	return result;
}

// The everyday HUD keeps rates and effective quality visible without the
// device inventory. Keep the checkout identity and requested-but-inactive features visible.
[[nodiscard]] inline FramegenHudText_t format_framegen_hud(
	uint32_t level, const FramegenHudSnapshot_t &snapshot )
{
	if ( level != 1u )
		return format_framegen_hud_detailed( level, snapshot );
	FramegenHudText_t result;
	const uint32_t source = static_cast<uint32_t>( std::min<uint64_t>(
		999u, ( snapshot.real + snapshot.delayedReal ) / 5u ) );
	const uint32_t generated = static_cast<uint32_t>( std::min<uint64_t>(
		999u, snapshot.generated / 5u ) );
	const uint64_t fresh = snapshot.real + snapshot.delayedReal + snapshot.generated;
	const uint64_t total = fresh + snapshot.repeats;
	const uint32_t fill = total != 0u ? ( fresh * 100u + total / 2u ) / total : 0u;
	char line[128] = {};
	std::snprintf( line, sizeof( line ), "SOURCE: %u fps | GENERATED: %u fps | FILL: %u%%",
		source, generated, fill );
	framegen_hud_add_line( result, line );
	char steps[24] = {}, state[24] = {};
	framegen_hud_format_steps( steps, sizeof( steps ), snapshot.netTrainedSteps );
	const char *version = snapshot.version != nullptr ? snapshot.version : "unknown";
	const char *checkout = std::strchr( version, '+' );
	checkout = checkout != nullptr ? checkout + 1 : version;
	std::snprintf( line, sizeof( line ), "MODE: %s/%s%s x%u %uHz%s | GIT: %.16s",
		mode_name( snapshot.mode ), pipeline_name( snapshot.pipeline ),
		std::strcmp( pipeline_name( snapshot.pipeline ), "spatial" ) == 0 ? "[EXP]" : "",
		std::clamp( snapshot.multiplier, 2u, 4u ),
		std::min( 999u, ( snapshot.refreshMilliHz + 500u ) / 1000u ),
		snapshot.vrrActive ? " VRR" : "", checkout );
	framegen_hud_add_line( result, line );
	std::snprintf( line, sizeof( line ), "LEARN: %s%s%s | PROFILE: %s",
		snapshot.netActive ? ( snapshot.netOnline ? "online " : "frozen " ) : "off",
		snapshot.netActive ? steps : "", snapshot.netActive ? " steps" : "",
		snapshot.netProfileLoaded ? "loaded" : "not loaded" );
	framegen_hud_add_line( result, line );
	const uint32_t jitter = std::min( snapshot.pacingSdTenthsMs, 9999u );
	std::snprintf( line, sizeof( line ), "STATE: %s | HIT: %u%% | JITTER: %u.%ums",
		framegen_hud_ladder_state( state, sizeof( state ), snapshot ),
		std::min( snapshot.deadlineHitPercent, 100u ), jitter / 10u, jitter % 10u );
	framegen_hud_add_line( result, line );

	// Keep the original twelve-window plots in the everyday HUD as well.
	// Reuse the diagnostic scales: deadline hit 0..100%, jitter 0..4 ms.
	const auto hitHistory = framegen_hud_sparkline( snapshot.deadlineHitHistory, 100.0 );
	const auto jitterHistory = framegen_hud_sparkline( snapshot.pacingSdMsHistory, 4.0 );
	std::snprintf( line, sizeof( line ), "HIT: %s | JITTER: %s",
		hitHistory.data(), jitterHistory.data() );
	framegen_hud_add_line( result, line );
	line[0] = '\0';
	if ( snapshot.netRequested && !snapshot.netActive )
		framegen_hud_append( line, sizeof( line ), "net:requested(OFF) " );
	if ( snapshot.bidirRequested && !snapshot.bidirActive )
		framegen_hud_append( line, sizeof( line ), "bidir:requested(OFF) " );
	if ( snapshot.vrrRequested && !snapshot.vrrActive )
		framegen_hud_append( line, sizeof( line ), "vrr:requested(off)" );
	if ( line[0] ) framegen_hud_add_line( result, line );
	return result;
}

// CGA gains breathing room between pixel-font rows. Quiet layouts retain
// their current geometry. 101 is the CPU-side automatic-opacity sentinel.
[[nodiscard]] constexpr uint32_t framegen_hud_line_pitch( bool cga ) { return cga ? 11u : 8u; }
[[nodiscard]] constexpr uint32_t framegen_hud_height( uint32_t lines, bool cga )
{
	return lines * framegen_hud_line_pitch( cga ) + 8u;
}
[[nodiscard]] constexpr uint32_t framegen_hud_style_flags( bool cga, uint32_t opacity = 101u )
{
	return ( cga ? 1u : 0u ) | ( opacity <= 100u ? ( opacity + 1u ) << 8u : 0u );
}

// Attribute bit lives in each existing packed character byte; font remains ASCII.
// A colon separates label/value within each pipe-delimited field. Diagnostic
// rows without colons retain their first word as the label.
[[nodiscard]] inline bool framegen_hud_value_column( const char *text, uint32_t column )
{
	if (text[column]=='|' || text[column]==' ') return false;
	uint32_t begin=column, end=column;
	while (begin>0u && text[begin-1u]!='|') --begin;
	while (text[end] && text[end]!='|') ++end;
	while (begin<end && text[begin]==' ') ++begin;
	for (uint32_t i=begin; i<end; ++i)
		if (text[i]==':') return column>i;
	for (uint32_t i=begin; i<end; ++i)
		if (text[i]==' ') return column>i;
	return false;
}

// Scalar-layout UBO consumed by cs_framegen_hud.comp. Text is packed four
// attributed ASCII bytes per uint (bit 7 marks a value); the uint64_t font above is split explicitly so the GPU
// ABI is independent of host endianness and uint64 shader support.
struct alignas( 16 ) FramegenHudUniform_t
{
	uint32_t lineCount = 0u;
	uint32_t widthChars = 0u;
	uint32_t hdr = 0u;
	uint32_t style = 0u; // bit 0: CGA; bits 8..15: opacity+1 (0 = automatic)
	std::array<uint32_t, k_uFramegenHudMaxLines> lineLengths = {};
	std::array<uint32_t, k_uFramegenHudPackedTextWords> text = {};
	std::array<uint32_t, 256> font = {};
};

static_assert( offsetof( FramegenHudUniform_t, lineLengths ) == 16u );
static_assert( offsetof( FramegenHudUniform_t, text ) == 48u );
static_assert( offsetof( FramegenHudUniform_t, font ) == 624u );
static_assert( sizeof( FramegenHudUniform_t ) == 1'648u );

struct FramegenHudPush_t
{
	uint32_t offsetX;
	uint32_t offsetY;
	uint32_t scale;
	uint32_t pad = 0u;

	FramegenHudPush_t( uint32_t x, uint32_t y, uint32_t uScale )
		: offsetX( x ), offsetY( y ), scale( uScale )
	{
	}
};
static_assert( sizeof( FramegenHudPush_t ) == 16u );

[[nodiscard]] inline FramegenHudUniform_t make_framegen_hud_uniform(
	const FramegenHudText_t &text, bool hdr, bool cga = false, uint32_t opacity = 101u )
{
	FramegenHudUniform_t result;
	result.lineCount = std::min( text.lineCount, k_uFramegenHudMaxLines );
	result.hdr = hdr;
	result.style = framegen_hud_style_flags( cga, opacity );
	for ( uint32_t line = 0u; line < result.lineCount; line++ )
	{
		const uint32_t length = static_cast<uint32_t>( std::min<size_t>(
			std::strlen( text.lines[line].data() ), k_uFramegenHudMaxColumns ) );
		result.lineLengths[line] = length;
		result.widthChars = std::max( result.widthChars, length );
		for ( uint32_t column = 0u; column < length; column++ )
		{
			const uint32_t byteIndex = line * k_uFramegenHudMaxColumns + column;
			result.text[byteIndex / 4u] |=
				( static_cast<uint32_t>( static_cast<uint8_t>( text.lines[line][column] ) & 0x7fu )
				  | ( framegen_hud_value_column( text.lines[line].data(), column ) ? 0x80u : 0u ) )
				<< ( 8u * ( byteIndex % 4u ) );
		}
	}
	for ( uint32_t glyph = 0u; glyph < 128u; glyph++ )
	{
		result.font[glyph * 2u] = static_cast<uint32_t>( k_uFramegenHudFont8x8[glyph] );
		result.font[glyph * 2u + 1u] = static_cast<uint32_t>(
			k_uFramegenHudFont8x8[glyph] >> 32u );
	}
	return result;
}

} // namespace gamescope::framegen
