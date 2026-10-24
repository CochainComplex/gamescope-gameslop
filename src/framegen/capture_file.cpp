#include "capture_file.hpp"
#include "atomic_file.hpp"

namespace gamescope::framegen
{

static bool write_plane( AtomicOutputFile &output, const CapturePlane &plane,
	uint32_t width, uint32_t height, uint32_t bytesPerRow )
{
	if ( plane.data == nullptr || plane.rowPitch < bytesPerRow
		|| plane.width != width || plane.height != height )
		return false;
	for ( uint32_t y = 0; y < height; y++ )
	{
		if ( !output.write( plane.data + (size_t)y * plane.rowPitch, bytesPerRow ) )
			return false;
	}
	return true;
}

bool write_field_capture( AtomicOutputFile &output, const FieldCapture &capture )
{
	const uint32_t header[ 8 ] = {
		0x44465347u /* 'GSFD' */, 1u,
		capture.width, capture.height,
		capture.lumaBytesPerPixel, 8u /* field bpp */, 1u /* has reverse field */, 0u,
	};
	if ( !output.write( header, sizeof( header ) )
		|| !output.write( &capture.sequence, sizeof( capture.sequence ) ) )
		return false;
	for ( size_t i = 0; i < capture.planes.size(); i++ )
	{
		const uint32_t bytesPerPixel = i < 2 ? capture.lumaBytesPerPixel : 8u;
		if ( !write_plane( output, capture.planes[ i ], capture.width, capture.height,
			capture.width * bytesPerPixel ) )
			return false;
	}
	return true;
}

bool write_color_capture( AtomicOutputFile &output, const ColorCapture &capture )
{
	const uint32_t header[ 12 ] = {
		0x46435347u /* 'GSCF' */, 3u,
		capture.width, capture.height, capture.drmFormat, capture.bytesPerPixel, capture.eotf,
		3u /* exact reference + paired candidates */, 4u /* planes */,
		capture.sweep, 0u, 0u,
	};
	const uint64_t metadata[ 8 ] = {
		capture.sequence, capture.anchorId, capture.referenceId, capture.endpointId,
		capture.anchorTimeNs, capture.referenceTimeNs, capture.endpointTimeNs, 0u,
	};
	const float floatMetadata[ 4 ] = {
		capture.phase, capture.strengths[ 0 ], capture.strengths[ 1 ], capture.strengths[ 2 ],
	};
	if ( !output.write( header, sizeof( header ) )
		|| !output.write( metadata, sizeof( metadata ) )
		|| !output.write( floatMetadata, sizeof( floatMetadata ) ) )
		return false;
	for ( const CapturePlane &plane : capture.planes )
	{
		if ( !write_plane( output, plane, capture.width, capture.height,
			capture.width * capture.bytesPerPixel ) )
			return false;
	}
	return true;
}

} // namespace gamescope::framegen
