#include "training_plugin.h"
#include "training_kernels.hpp"
#include "framegen/compute_job.hpp"
#include "framegen/net_layout.hpp"
#include "framegen/backend_contract.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

using gamescope::framegen::compute::Batch;
using gamescope::framegen::compute::Ticket;
using gamescope::framegen::compute::TrainingKernelJob;

namespace
{
constexpr size_t P = GSC_TRAIN_PARAMETERS;
constexpr size_t GradientCount = P * GSC_TRAIN_SLICES;
constexpr size_t OutputCount = GradientCount + 5 * P;
static_assert( P == k_uFramegenNetFloats && GSC_TRAIN_MODEL_VERSION == k_uFramegenNetVersion );

struct Error : std::runtime_error
{
    int code;
    Error( int result, const char *message ) : std::runtime_error( message ), code( result ) {}
};

void require( bool condition, const char *message )
{
    if ( !condition ) throw Error( GSC_TRAIN_INVALID, message );
}

void hip( hipError_t result )
{
    if ( result != hipSuccess ) throw Error( GSC_TRAIN_FAILED, hipGetErrorString( result ) );
}

template<typename F> int guarded( char *error, uint32_t size, F function ) noexcept
{
    if ( error && size ) error[0] = 0;
    try { return function(); }
    catch ( const Error &exception )
    {
        if ( error && size ) std::snprintf( error, size, "%s", exception.what() );
        return exception.code;
    }
    catch ( const std::exception &exception )
    {
        if ( error && size ) std::snprintf( error, size, "%s", exception.what() );
        return GSC_TRAIN_FAILED;
    }
    catch ( ... ) { return GSC_TRAIN_FAILED; }
}

struct DeviceScope
{
    int previous;
    explicit DeviceScope( int selected )
    {
        hip( hipGetDevice( &previous ) );
        hip( hipSetDevice( selected ) );
    }
    ~DeviceScope() { (void)hipSetDevice( previous ); }
};

gsc_train_device describe( int ordinal )
{
    gsc_train_device result{};
    result.size = sizeof( result );
    hip( hipDeviceGetName( result.name, sizeof( result.name ), ordinal ) );
    hip( hipDeviceGetPCIBusId( result.pci, sizeof( result.pci ), ordinal ) );
    hipUUID uuid;
    hip( hipDeviceGetUuid( &uuid, ordinal ) );
    std::memcpy( result.uuid, uuid.bytes, sizeof( result.uuid ) );
    return result;
}

struct Context
{
    int ordinal;
    hipStream_t stream = nullptr;
    hipEvent_t event = nullptr;
    float *hostInput = nullptr, *hostOutput = nullptr;
    float *deviceInput = nullptr, *deviceOutput = nullptr;
    size_t inputCount = 0;
    Batch batch;
    bool failed = false;

    explicit Context( int device ) : ordinal( device ) {}
    void initialize()
    {
        DeviceScope selected( ordinal );
        hip( hipStreamCreateWithFlags( &stream, hipStreamNonBlocking ) );
        hip( hipEventCreateWithFlags( &event, hipEventDisableTiming ) );
    }
    hipError_t close() noexcept
    {
        // A failed drain must not free pinned memory still potentially in use.
        // Keep the context owned and return failure; its caller must not unload
        // the module while retirement is unresolved.
        int previous = 0;
        if ( hipGetDevice( &previous ) != hipSuccess || hipSetDevice( ordinal ) != hipSuccess )
            return hipErrorInvalidDevice;
        hipError_t result = stream ? hipStreamSynchronize( stream ) : hipSuccess;
        if ( result == hipSuccess )
        {
            if ( deviceInput ) result = hipFree( deviceInput );
            if ( result == hipSuccess ) deviceInput = nullptr;
            if ( result == hipSuccess && deviceOutput ) result = hipFree( deviceOutput );
            if ( result == hipSuccess ) deviceOutput = nullptr;
            if ( result == hipSuccess && hostInput ) result = hipHostFree( hostInput );
            if ( result == hipSuccess ) hostInput = nullptr;
            if ( result == hipSuccess && hostOutput ) result = hipHostFree( hostOutput );
            if ( result == hipSuccess ) hostOutput = nullptr;
            if ( result == hipSuccess && event ) result = hipEventDestroy( event );
            if ( result == hipSuccess ) event = nullptr;
            if ( result == hipSuccess && stream ) result = hipStreamDestroy( stream );
            if ( result == hipSuccess ) stream = nullptr;
        }
        (void)hipSetDevice( previous );
        return result;
    }
    ~Context() { (void)close(); }

    void resize( size_t count )
    {
        if ( inputCount == count ) return;
        // Called only while idle, after event completion and retirement.
        if ( deviceInput ) { hip( hipFree( deviceInput ) ); deviceInput = nullptr; }
        if ( deviceOutput ) { hip( hipFree( deviceOutput ) ); deviceOutput = nullptr; }
        if ( hostInput ) { hip( hipHostFree( hostInput ) ); hostInput = nullptr; }
        if ( hostOutput ) { hip( hipHostFree( hostOutput ) ); hostOutput = nullptr; }
        inputCount = 0;
        hip( hipHostMalloc( &hostInput, count * sizeof( float ), hipHostMallocDefault ) );
        hip( hipHostMalloc( &hostOutput, OutputCount * sizeof( float ), hipHostMallocDefault ) );
        hip( hipMalloc( &deviceInput, count * sizeof( float ) ) );
        hip( hipMalloc( &deviceOutput, OutputCount * sizeof( float ) ) );
        inputCount = count;
    }
    int poll( Ticket ticket )
    {
        if ( failed ) return GSC_TRAIN_FAILED;
        if ( !batch.owns( ticket ) ) return GSC_TRAIN_STALE;
        const hipError_t status = hipEventQuery( event );
        if ( status == hipErrorNotReady ) return GSC_TRAIN_PENDING;
        if ( status != hipSuccess ) { failed = true; hip( status ); }
        batch.complete( ticket );
        return batch.discarded( ticket ) ? GSC_TRAIN_DISCARDED : GSC_TRAIN_COMPLETE;
    }
};

Context &context( void *value )
{
    require( value != nullptr, "null training context" );
    return *static_cast<Context *>( value );
}
Ticket ticket( gsc_train_ticket value ) { return { value.epoch, value.serial }; }

int devices( gsc_train_device *output, uint32_t capacity, uint32_t *count, char *error, uint32_t size )
{
    return guarded( error, size, [&] {
        require( count && ( capacity == 0 || output ), "invalid device array" );
        int available = 0;
        const auto status = hipGetDeviceCount( &available );
        if ( status != hipSuccess ) throw Error( GSC_TRAIN_UNAVAILABLE, hipGetErrorString( status ) );
        *count = uint32_t( available );
        if ( capacity && capacity < uint32_t( available ) ) return int( GSC_TRAIN_INVALID );
        for ( int i = 0; capacity && i < available; ++i ) output[i] = describe( i );
        return int( GSC_TRAIN_OK );
    } );
}

int create( const gsc_train_create *info, void **out, char *error, uint32_t size )
{
    if ( out ) *out = nullptr;
    return guarded( error, size, [&] {
        require( info && out && info->size == sizeof( *info ) && info->abi == GSC_TRAIN_ABI
            && info->device.size == sizeof( gsc_train_device ), "incompatible creation ABI" );
        require( std::memchr( info->device.pci, 0, sizeof( info->device.pci ) )
            && info->device.pci[0], "explicit PCI identity required" );
        int count = 0, selected = -1, matches = 0;
        hip( hipGetDeviceCount( &count ) );
        for ( int i = 0; i < count; ++i )
        {
            const auto candidate = describe( i );
            if ( !std::strcmp( info->device.pci, candidate.pci )
                && !std::memcmp( info->device.uuid, candidate.uuid, sizeof( candidate.uuid ) ) )
            { selected = i; ++matches; }
        }
        require( matches == 1, "requested PCI/UUID must match exactly one HIP device" );
        auto result = std::make_unique<Context>( selected );
        result->initialize();
        *out = result.release();
        return int( GSC_TRAIN_OK );
    } );
}

int submit( void *value, const gsc_train_job *job, gsc_train_ticket *out, char *error, uint32_t size )
{
    if ( out ) *out = {};
    return guarded( error, size, [&] {
        auto &ctx = context( value );
        if ( ctx.failed ) return int( GSC_TRAIN_FAILED );
        if ( ctx.batch.busy() ) return int( GSC_TRAIN_BUSY );
        require( job && out && job->size == sizeof( *job ) && job->model_version == GSC_TRAIN_MODEL_VERSION,
            "incompatible training job ABI/model" );
        const uint64_t n = uint64_t( job->width ) * job->height;
        require( job->width >= 16 && job->height >= 16 && n <= ( 1u << 20 )
            && job->step != 0 && !( job->flags & ~5u ) && job->scene_cut <= 1
            && job->epoch == ctx.batch.epoch(), "invalid shape, mode, step or epoch" );
        require( std::isfinite( job->learning_rate ) && job->learning_rate >= 0
            && std::isfinite( job->ema_alpha ) && job->ema_alpha >= 0 && job->ema_alpha <= 1
            && std::isfinite( job->prior_decay ) && job->prior_decay >= 0 && job->prior_decay <= 1,
            "invalid optimizer controls" );
        require( job->luma_count == n && job->field_count == 4 * n && job->prior_count == P
            && job->state_count == 4 * P, "incorrect tensor lengths" );
        const std::array<const float *, 6> input{ job->source, job->destination,
            job->forward, job->reverse, job->prior, job->state };
        const std::array<size_t, 6> lengths{ size_t( n ), size_t( n ), size_t( 4 * n ), size_t( 4 * n ), P, 4 * P };
        for ( size_t i = 0; i < input.size(); ++i )
        {
            require( input[i] != nullptr, "null tensor" );
            require( std::all_of( input[i], input[i] + lengths[i], []( float f ) { return std::isfinite( f ); } ),
                "nonfinite input tensor" );
        }
        DeviceScope selected( ctx.ordinal );
        ctx.resize( size_t( 10 * n + 5 * P ) );
        size_t offset = 0;
        for ( size_t i = 0; i < input.size(); ++i )
        {
            std::memcpy( ctx.hostInput + offset, input[i], lengths[i] * sizeof( float ) );
            offset += lengths[i];
        }
        const auto owned = ctx.batch.begin( job->epoch );
        require( owned.serial != 0, "training ticket unavailable" );
        try
        {
            hip( hipMemcpyAsync( ctx.deviceInput, ctx.hostInput, ctx.inputCount * sizeof( float ), hipMemcpyHostToDevice, ctx.stream ) );
            const TrainingKernelJob kernel{ job->width, job->height, job->step, job->flags, job->scene_cut,
                job->learning_rate, job->ema_alpha, job->prior_decay,
                ctx.deviceInput, ctx.deviceInput + n, ctx.deviceInput + 2 * n, ctx.deviceInput + 6 * n,
                ctx.deviceInput + 10 * n, ctx.deviceInput + 10 * n + P,
                ctx.deviceOutput, ctx.deviceOutput + GradientCount, ctx.deviceOutput + GradientCount + 4 * P };
            hip( gamescope::framegen::compute::launch_training( kernel, ctx.stream ) );
            hip( hipMemcpyAsync( ctx.hostOutput, ctx.deviceOutput, OutputCount * sizeof( float ), hipMemcpyDeviceToHost, ctx.stream ) );
            hip( hipEventRecord( ctx.event, ctx.stream ) );
        }
        catch ( ... ) { ctx.failed = true; throw; }
        *out = { owned.epoch, owned.serial };
        return int( GSC_TRAIN_OK );
    } );
}

int poll( void *value, gsc_train_ticket owned, char *error, uint32_t size )
{
    return guarded( error, size, [&] { auto &ctx = context( value ); DeviceScope selected( ctx.ordinal ); return ctx.poll( ticket( owned ) ); } );
}

int collect( void *value, gsc_train_ticket owned, gsc_train_output *out, char *error, uint32_t size )
{
    return guarded( error, size, [&] {
        auto &ctx = context( value );
        require( out && out->size == sizeof( *out ) && out->gradient_count == GradientCount
            && out->state_count == 4 * P && out->served_count == P && out->gradients && out->state && out->served,
            "incorrect result buffers" );
        DeviceScope selected( ctx.ordinal );
        const int status = ctx.poll( ticket( owned ) );
        if ( status != GSC_TRAIN_COMPLETE ) return status;
        require( std::all_of( ctx.hostOutput, ctx.hostOutput + OutputCount, []( float f ) { return std::isfinite( f ); } ),
            "nonfinite training result; candidate not committed" );
        std::memcpy( out->gradients, ctx.hostOutput, GradientCount * sizeof( float ) );
        std::memcpy( out->state, ctx.hostOutput + GradientCount, 4 * P * sizeof( float ) );
        std::memcpy( out->served, ctx.hostOutput + GradientCount + 4 * P, P * sizeof( float ) );
        require( ctx.batch.retire( ticket( owned ) ), "completed result lost ownership" );
        return int( GSC_TRAIN_OK );
    } );
}

int discard( void *value, gsc_train_ticket owned, char *error, uint32_t size )
{
    return guarded( error, size, [&] {
        auto &ctx = context( value );
        if ( !ctx.batch.owns( ticket( owned ) ) ) return int( GSC_TRAIN_STALE );
        ctx.batch.discard( ticket( owned ) );
        return int( GSC_TRAIN_OK );
    } );
}

int retire( void *value, gsc_train_ticket owned, char *error, uint32_t size )
{
    return guarded( error, size, [&] {
        auto &ctx = context( value ); DeviceScope selected( ctx.ordinal );
        const int status = ctx.poll( ticket( owned ) );
        if ( status != GSC_TRAIN_COMPLETE && status != GSC_TRAIN_DISCARDED ) return status;
        require( ctx.batch.retire( ticket( owned ) ), "retirement lost ownership" );
        return int( GSC_TRAIN_OK );
    } );
}

int reset( void *value, uint64_t *epoch, char *error, uint32_t size )
{
    return guarded( error, size, [&] {
        auto &ctx = context( value ); require( epoch, "null epoch output" );
        if ( ctx.failed ) return int( GSC_TRAIN_FAILED );
        if ( !ctx.batch.reset() ) return int( GSC_TRAIN_BUSY );
        *epoch = ctx.batch.epoch();
        return int( GSC_TRAIN_OK );
    } );
}

int destroy( void *value, char *error, uint32_t size )
{
    return guarded( error, size, [&] {
        auto &ctx = context( value );
        // close may fail after partially releasing allocations. Only retrying
        // destruction is valid afterwards, never admission into partial state.
        ctx.failed = true;
        hip( ctx.close() );
        delete &ctx;
        return int( GSC_TRAIN_OK );
    } );
}

const gsc_training_api Api{ sizeof( gsc_training_api ), GSC_TRAIN_ABI, GSC_TRAIN_MODEL_VERSION,
    GSC_TRAIN_PARAMETERS, GSC_TRAIN_SLICES, "HIP current-framegen online trainer", devices,
    create, submit, poll, collect, discard, retire, reset, destroy };
} // namespace

extern "C" const gsc_training_api *gsc_training_get_api( uint32_t abi, uint32_t size )
{
    return abi == GSC_TRAIN_ABI && size == sizeof( Api ) ? &Api : nullptr;
}

extern "C" const gsc_backend_contract *gsc_compute_get_contract( uint32_t abi, uint32_t size, uint32_t payload )
{
    static constexpr auto contract = gsc_contract_snapshot_training();
    return abi == GSC_BACKEND_ABI && size == sizeof( contract )
        && payload == GSC_PAYLOAD_SNAPSHOT_TRAINING ? &contract : nullptr;
}
