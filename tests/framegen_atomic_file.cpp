#include <catch2/catch_test_macros.hpp>

#include "framegen/atomic_file.hpp"
#include "framegen/capture_file.hpp"

#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <fcntl.h>

using gamescope::framegen::AtomicOutputFile;

namespace
{

struct StdioFault;
thread_local StdioFault *activeFault = nullptr;

// Only this executable wraps stdio. Each test owns its fault state, and only
// the stream opened in that scope can consume a write/close fault.
struct StdioFault
{
	int openError = 0;
	bool shortWrite = false;
	size_t writesBeforeFailure = 0;
	bool failClose = false;
	int closeError = 0;
	int fd = -1;
	FILE *stream = nullptr;
	int realCloseResult = EOF;

	StdioFault()
	{
		REQUIRE( activeFault == nullptr );
		activeFault = this;
	}
	~StdioFault() { activeFault = nullptr; }
};

struct AtomicFiles
{
	std::filesystem::path directory;
	std::filesystem::path path;

	AtomicFiles()
	{
		char pattern[] = "/tmp/gamescope-atomic-fault-XXXXXX";
		const char *created = mkdtemp( pattern );
		REQUIRE( created != nullptr );
		directory = created;
		path = directory / "profile.bin";
		std::ofstream seed( path, std::ios::binary );
		seed << "old-profile";
		REQUIRE( seed.good() );
	}
	~AtomicFiles() { std::filesystem::remove_all( directory ); }

	std::string contents() const
	{
		std::ifstream file( path, std::ios::binary );
		return { std::istreambuf_iterator<char>( file ), std::istreambuf_iterator<char>() };
	}

	void check_preserved() const
	{
		CHECK( contents() == "old-profile" );
		CHECK( std::distance( std::filesystem::directory_iterator( directory ),
			std::filesystem::directory_iterator() ) == 1 );
	}
};

bool descriptor_is_closed( int fd )
{
	errno = 0;
	return fcntl( fd, F_GETFD ) == -1 && errno == EBADF;
}

} // namespace

extern "C" FILE *__real_fdopen( int fd, const char *mode );
extern "C" size_t __real_fwrite( const void *data, size_t size, size_t count, FILE *stream );
extern "C" int __real_fclose( FILE *stream );

extern "C" FILE *__wrap_fdopen( int fd, const char *mode )
{
	if ( activeFault == nullptr )
		return __real_fdopen( fd, mode );
	activeFault->fd = fd;
	if ( activeFault->openError != 0 )
	{
		errno = activeFault->openError;
		activeFault->openError = 0;
		return nullptr;
	}
	activeFault->stream = __real_fdopen( fd, mode );
	return activeFault->stream;
}

extern "C" size_t __wrap_fwrite( const void *data, size_t size, size_t count, FILE *stream )
{
	if ( activeFault != nullptr && activeFault->stream == stream && activeFault->shortWrite && count > 0 )
	{
		if ( activeFault->writesBeforeFailure > 0 )
		{
			activeFault->writesBeforeFailure--;
			return __real_fwrite( data, size, count, stream );
		}
		activeFault->shortWrite = false;
		const size_t written = __real_fwrite( data, size, count - 1, stream );
		errno = 0;
		return written;
	}
	return __real_fwrite( data, size, count, stream );
}

extern "C" int __wrap_fclose( FILE *stream )
{
	const bool matches = activeFault != nullptr && activeFault->stream == stream;
	const int result = __real_fclose( stream );
	if ( matches )
	{
		activeFault->stream = nullptr;
		activeFault->realCloseResult = result;
		if ( activeFault->failClose )
		{
			activeFault->failClose = false;
			errno = activeFault->closeError;
			return EOF;
		}
	}
	return result;
}

TEST_CASE( "Atomic output should close and discard staging when fdopen fails", "[framegen_atomic_file]" )
{
	AtomicFiles files;
	StdioFault fault;
	fault.openError = ENOMEM;
	AtomicOutputFile output( files.path );
	// Check before assertions or other file operations could reuse a leaked fd.
	const bool closed = descriptor_is_closed( fault.fd );
	CHECK( fault.fd >= 0 );
	CHECK( closed );
	CHECK_FALSE( output.is_open() );
	CHECK( output.error() == ENOMEM );
	CHECK_FALSE( output.write( "x", 1 ) );
	CHECK_FALSE( output.commit() );
	CHECK( output.error() == ENOMEM );
	files.check_preserved();

	AtomicOutputFile recovered( files.path );
	CHECK( recovered.is_open() );
	CHECK( recovered.write( "new", 3 ) );
	CHECK( recovered.commit() );
	CHECK( recovered.error() == 0 );
	CHECK( files.contents() == "new" );
}

TEST_CASE( "Atomic output should retain EIO when a short write has no errno and cleanup also fails", "[framegen_atomic_file]" )
{
	AtomicFiles files;
	StdioFault fault;
	AtomicOutputFile output( files.path );
	CHECK( output.is_open() );
	fault.shortWrite = true;
	CHECK_FALSE( output.write( "partial", 7 ) );
	CHECK_FALSE( fault.shortWrite );
	CHECK( output.error() == EIO );
	fault.failClose = true;
	fault.closeError = ENOSPC;
	CHECK_FALSE( output.commit() );
	CHECK_FALSE( fault.failClose );
	CHECK( fault.realCloseResult == 0 );
	CHECK_FALSE( output.is_open() );
	CHECK( output.error() == EIO );
	CHECK_FALSE( output.commit() );
	CHECK( output.error() == EIO );
	files.check_preserved();
}

TEST_CASE( "Atomic output should reject publication when buffered close fails", "[framegen_atomic_file]" )
{
	for ( int error : { ENOSPC, 0 } )
	{
		INFO( "close errno=" << error );
		AtomicFiles files;
		StdioFault fault;
		AtomicOutputFile output( files.path );
		CHECK( output.is_open() );
		CHECK( output.write( "replacement", 11 ) );
		fault.failClose = true;
		fault.closeError = error;
		CHECK_FALSE( output.commit() );
		CHECK_FALSE( fault.failClose );
		CHECK( fault.realCloseResult == 0 );
		CHECK_FALSE( output.is_open() );
		CHECK( output.error() == ( error != 0 ? error : EIO ) );
		files.check_preserved();
	}
}

TEST_CASE( "Atomic output should discard abandoned staging even when close fails", "[framegen_atomic_file]" )
{
	AtomicFiles files;
	StdioFault fault;
	{
		AtomicOutputFile output( files.path );
		CHECK( output.is_open() );
		CHECK( output.write( "abandoned", 9 ) );
		fault.failClose = true;
		fault.closeError = ENOSPC;
	}
	const bool closed = descriptor_is_closed( fault.fd );
	CHECK( closed );
	CHECK_FALSE( fault.failClose );
	CHECK( fault.realCloseResult == 0 );
	files.check_preserved();
}

TEST_CASE( "Capture writers should abandon publication when a metadata or plane write is short", "[framegen_atomic_file][framegen_capture]" )
{
	// GSFD writes a header, sequence and eight rows; GSCF writes a header,
	// integer metadata, float metadata and eight rows. Fail every boundary,
	// including later rows/planes after complete records have been buffered.
	for ( bool color : { false, true } )
		for ( size_t write = 0; write < ( color ? 11u : 10u ); write++ )
		{
			INFO( "color=" << color << " write=" << write );
			AtomicFiles files;
			StdioFault fault;
			AtomicOutputFile output( files.path );
			CHECK( output.is_open() );
			const std::array<uint8_t, 16> pixels{ 1, 2, 3, 4 };
			std::array<gamescope::framegen::CapturePlane, 4> planes;
			for ( auto &plane : planes )
				plane = { pixels.data(), 1, 2, 8 };
			fault.shortWrite = true;
			fault.writesBeforeFailure = write;
			if ( color )
			{
				gamescope::framegen::ColorCapture capture{};
				capture.width = 1; capture.height = 2; capture.bytesPerPixel = 8;
				capture.phase = 0.5f; capture.planes = planes;
				CHECK_FALSE( gamescope::framegen::write_color_capture( output, capture ) );
			}
			else
				CHECK_FALSE( gamescope::framegen::write_field_capture( output, { 1, 2, 8, 9, planes } ) );
			CHECK_FALSE( fault.shortWrite );
			CHECK( output.error() == EIO );
			CHECK_FALSE( output.commit() );
			CHECK_FALSE( output.is_open() );
			files.check_preserved();
		}
}
