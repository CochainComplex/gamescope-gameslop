#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace gamescope::framegen
{

class AtomicOutputFile;

// Borrowed CPU texel origin, including any subresource offset. The caller owns
// the readback and must retire its GPU writes before passing this view.
struct CapturePlane
{
	const uint8_t *data = nullptr;
	uint32_t width = 0;
	uint32_t height = 0;
	size_t rowPitch = 0;
};

struct FieldCapture
{
	uint32_t width, height, lumaBytesPerPixel;
	uint64_t sequence;
	// Previous luma, current luma, forward field, reverse field.
	std::array<CapturePlane, 4> planes;
};

struct ColorCapture
{
	uint32_t width, height, drmFormat, bytesPerPixel, eotf, sweep;
	uint64_t sequence, anchorId, referenceId, endpointId;
	uint64_t anchorTimeNs, referenceTimeNs, endpointTimeNs;
	float phase;
	std::array<float, 3> strengths;
	// Generated candidates in strengths order, then the exact real reference.
	std::array<CapturePlane, 4> planes;
};

// Synchronously write the existing GSFD v1 / GSCF v3 payloads, tightly packing
// padded rows. No pointers are retained. The caller opens and commits the
// transaction only on success; false must abandon it even if output.error()==0
// (an invalid plane, as opposed to a filesystem failure).
bool write_field_capture( AtomicOutputFile &output, const FieldCapture &capture );
bool write_color_capture( AtomicOutputFile &output, const ColorCapture &capture );

} // namespace gamescope::framegen
