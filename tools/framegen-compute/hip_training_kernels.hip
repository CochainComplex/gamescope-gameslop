#include "training_kernels.hpp"
#include "framegen/net_layout.hpp"

#include <hip/hip_runtime.h>
#include <cmath>

namespace gamescope::framegen::compute
{
namespace
{
// Current online GLSL layout, not the offline Python trainer's packing.
constexpr int C_IN = 12, C_MID = 16;
constexpr int LT = 4, T2 = LT + 2, T1 = LT + 4, T0 = LT + 6;
constexpr int W1_OFF = 0, B1_OFF = 1728, W2_OFF = 1744;
constexpr int W3_OFF = 4064, B3_OFF = 4640, N_PARAMS = 4644;
constexpr int N_SLICES = 16;
constexpr float CHARB_EPS = 1e-3f, FLOW_AUX = 0.05f;
constexpr float LAMBDA_DF = 0.01f, LAMBDA_DC = 0.005f;
static_assert( N_PARAMS == k_uFramegenNetFloats && W2_OFF == k_uFramegenNetLayer2Offset
    && W3_OFF == k_uFramegenNetLayer3Offset && B3_OFF == k_uFramegenNetLayer3BiasOffset );

__device__ float clampValue( float value, float low, float high )
{
    return fminf( fmaxf( value, low ), high );
}

__device__ int clampIndex( int value, int size )
{
    return value < 0 ? 0 : ( value >= size ? size - 1 : value );
}

__device__ float mixValue( float a, float b, float t )
{
    return a * ( 1.0f - t ) + b * t;
}

// Packed tensor planes are float-aligned, not necessarily float4-aligned:
// odd field dimensions put the first field after 2*width*height floats.
__device__ float4 loadField( const float *field, int at )
{
    const float *p = field + 4 * at;
    return make_float4( p[0], p[1], p[2], p[3] );
}

// framegen_evidence.h: keep the luma normalization and smoothstep polynomial.
__device__ float lumaResid( float a, float b )
{
    return fabsf( a - b ) / fmaxf( 1.0f, fmaxf( fabsf( a ), fabsf( b ) ) );
}

__device__ float evidenceGate( float refinedResid, float heldResid )
{
    const float t = clampValue( ( refinedResid - 0.03f ) / ( 0.10f - 0.03f ), 0.0f, 1.0f );
    const float predicts = 1.0f - t * t * ( 3.0f - 2.0f * t );
    const float beatsHolding = clampValue( ( heldResid - refinedResid ) / fmaxf( heldResid, 0.03f ), 0.0f, 1.0f );
    return predicts * beatsHolding;
}

// Pixel-center coordinates, clamp-to-edge texels, analytic spatial derivative.
// Clamp the floored coordinates before integer conversion; this has the same
// result in the shader's defined integer range and also handles huge finite flow.
__device__ float3 bilinearGrad( const float *source, float cx, float cy, int width, int height )
{
    const float fx = cx - 0.5f, fy = cy - 0.5f;
    const float ix = floorf( fx ), iy = floorf( fy );
    const float wx = fx - ix, wy = fy - iy;
    const int x0 = int( clampValue( ix, 0.0f, float( width - 1 ) ) );
    const int y0 = int( clampValue( iy, 0.0f, float( height - 1 ) ) );
    const int x1 = int( clampValue( ix + 1.0f, 0.0f, float( width - 1 ) ) );
    const int y1 = int( clampValue( iy + 1.0f, 0.0f, float( height - 1 ) ) );
    const float v00 = source[y0 * width + x0], v10 = source[y0 * width + x1];
    const float v01 = source[y1 * width + x0], v11 = source[y1 * width + x1];
    const float top = mixValue( v00, v10, wx ), bot = mixValue( v01, v11, wx );
    return make_float3( mixValue( top, bot, wy ), mixValue( v10 - v00, v11 - v01, wy ), bot - top );
}

// cs_framegen_motion_net_train.comp: one 64-thread block per slice, eight
// forward then eight reverse. Keep each per-parameter reduction serial and in
// source order; all cross-thread cooperation uses LDS and block barriers.
__global__ void trainingGradients( TrainingKernelJob job )
{
    __shared__ float s_feat[T0 * T0 * C_IN];
    __shared__ float s_a1[T1 * T1 * C_MID];
    __shared__ float s_a2[T2 * T2 * C_MID];
    __shared__ float s_da1[T1 * T1 * C_MID];
    __shared__ float s_da2[T2 * T2 * C_MID];
    __shared__ float s_do[LT * LT * 4];
    __shared__ float s_w[2320];

    const unsigned li = threadIdx.x, row = blockIdx.x;
    const bool backwards = row >= 8;
    const unsigned tile = row & 7u;
    const int width = int( job.width ), height = int( job.height );
    const bool conservativeBidir = ( job.flags & 4u ) != 0u;
    const float *source = backwards ? job.destination : job.source;
    const float *destination = backwards ? job.source : job.destination;
    const float *field = backwards ? job.reverse : job.forward;
    const float *opposite = backwards ? job.forward : job.reverse;
    float *gradient = job.gradients + row * N_PARAMS;

    if ( ( job.flags & 1u ) != 0u && job.sceneCut != 0u )
    {
        for ( unsigned i = li; i < unsigned( N_PARAMS ); i += 64u ) gradient[i] = 0.0f;
        return;
    }

    unsigned seed = job.step * 0x9E3779B9u + 0x61C88647u;
    if ( backwards ) seed ^= 0x55555555u;
    unsigned hash = ( seed ^ ( tile * 2654435761u ) ) * 2246822519u;
    hash ^= hash >> 15;
    const int rangeX = width - T0 + 1 - 3, rangeY = height - T0 + 1 - 3;
    const int originX = 3 + int( hash % unsigned( rangeX > 1 ? rangeX : 1 ) );
    const int originY = 3 + int( ( hash >> 16 ) % unsigned( rangeY > 1 ? rangeY : 1 ) );
    const int featX = originX - 3, featY = originY - 3;

    for ( unsigned i = li; i < 1744u; i += 64u ) s_w[i] = job.state[W1_OFF + i];
    for ( unsigned i = li; i < unsigned( T0 * T0 ); i += 64u )
    {
        const int px = clampIndex( featX + int( i ) % T0, width );
        const int py = clampIndex( featY + int( i ) / T0, height );
        const int at = py * width + px;
        const float4 x = loadField( field, at ), y = loadField( opposite, at );
        const float ld = destination[at];
        const float3 wg = bilinearGrad( source, float( px ) + 0.5f - x.x, float( py ) + 0.5f - x.y, width, height );
        const float resid = fabsf( ld - wg.x ) / fmaxf( 1.0f, fmaxf( fabsf( ld ), fabsf( wg.x ) ) );
        const unsigned b = i * unsigned( C_IN );
        s_feat[b + 0u] = x.x * 0.125f;
        s_feat[b + 1u] = x.y * 0.125f;
        s_feat[b + 2u] = x.z;
        s_feat[b + 3u] = fminf( x.w * 0.5f, 1.0f );
        s_feat[b + 4u] = y.x * 0.125f;
        s_feat[b + 5u] = y.y * 0.125f;
        s_feat[b + 6u] = y.z;
        s_feat[b + 7u] = fminf( y.w * 0.5f, 1.0f );
        s_feat[b + 8u] = ld;
        s_feat[b + 9u] = wg.x;
        s_feat[b + 10u] = resid;
        // FramegenMotionNetTrainPush_t always sets mode=0, including bidir.
        s_feat[b + 11u] = 0.0f;
    }
    __syncthreads();

    for ( unsigned i = li; i < unsigned( T1 * T1 ); i += 64u )
    {
        const int ax = int( i ) % T1, ay = int( i ) / T1;
        for ( int co = 0; co < C_MID; ++co )
        {
            float acc = s_w[B1_OFF + co];
            for ( int tap = 0; tap < 9; ++tap )
            {
                const int fb = ( ( ay + tap / 3 ) * T0 + ( ax + tap % 3 ) ) * C_IN;
                const int wb = ( co * 9 + tap ) * C_IN;
                for ( int ci = 0; ci < C_IN; ++ci ) acc += s_feat[fb + ci] * s_w[wb + ci];
            }
            s_a1[i * unsigned( C_MID ) + unsigned( co )] = fmaxf( acc, 0.0f );
        }
    }
    __syncthreads();
    for ( unsigned i = li; i < 2320u; i += 64u ) s_w[i] = job.state[W2_OFF + i];
    __syncthreads();
    for ( unsigned i = li; i < unsigned( T2 * T2 ); i += 64u )
    {
        const int bx = int( i ) % T2, by = int( i ) / T2;
        for ( int co = 0; co < C_MID; ++co )
        {
            float acc = s_w[2304 + co];
            for ( int tap = 0; tap < 9; ++tap )
            {
                const int ab = ( ( by + tap / 3 ) * T1 + ( bx + tap % 3 ) ) * C_MID;
                const int wb = ( co * 9 + tap ) * C_MID;
                for ( int ci = 0; ci < C_MID; ++ci ) acc += s_a1[ab + ci] * s_w[wb + ci];
            }
            s_a2[i * unsigned( C_MID ) + unsigned( co )] = fmaxf( acc, 0.0f );
        }
    }
    __syncthreads();
    for ( unsigned i = li; i < 580u; i += 64u ) s_w[i] = job.state[W3_OFF + i];
    __syncthreads();

    for ( unsigned i = li; i < unsigned( LT * LT ); i += 64u )
    {
        const int cx = int( i ) % LT, cy = int( i ) / LT;
        float o[4];
        for ( int c = 0; c < 4; ++c )
        {
            float acc = s_w[576 + c];
            for ( int tap = 0; tap < 9; ++tap )
            {
                const int ab = ( ( cy + tap / 3 ) * T2 + ( cx + tap % 3 ) ) * C_MID;
                const int wb = ( c * 9 + tap ) * C_MID;
                for ( int ci = 0; ci < C_MID; ++ci ) acc += s_a2[ab + ci] * s_w[wb + ci];
            }
            o[c] = acc;
        }
        const int qx = originX + cx, qy = originY + cy, at = qy * width + qx;
        const float4 x0 = loadField( field, at );
        const float thX = tanhf( o[0] ), thY = tanhf( o[1] );
        const float dfX = conservativeBidir ? 0.0f : 2.0f * thX;
        const float dfY = conservativeBidir ? 0.0f : 2.0f * thY;
        const float fhX = x0.x + dfX, fhY = x0.y + dfY;
        const float ld = destination[at], lsp = source[at];
        const float3 wg = bilinearGrad( source, float( qx ) + 0.5f - fhX, float( qy ) + 0.5f - fhY, width, height );
        const float gate = evidenceGate( lumaResid( wg.x, ld ), lumaResid( lsp, ld ) );
        const float dcChain = conservativeBidir ? ( o[2] <= 0.0f ? 1.0f : 0.0f ) : ( o[2] < 0.0f ? 1.0f : gate );
        const float dc = conservativeBidir ? fminf( o[2], 0.0f ) : o[2] * dcChain;
        const float chRaw = x0.z + dc, ch = clampValue( chRaw, 0.0f, 1.0f );
        const float ew = wg.x - ld, rw = sqrtf( ew * ew + CHARB_EPS * CHARB_EPS );
        const float efb = lsp - ld, rfb = sqrtf( efb * efb + CHARB_EPS * CHARB_EPS );
        constexpr float invN = 1.0f / float( LT * LT );
        const float dLdCh = ( rw - 0.5f * rfb ) * invN;
        const float live = chRaw > 0.0f && chRaw < 1.0f ? 1.0f : 0.0f;
        const float do2 = dLdCh * live * dcChain + ( 2.0f * LAMBDA_DC * invN ) * dc * dcChain;
        const float dLdWarp = ( ( ch + FLOW_AUX ) * invN ) * ( ew / rw );
        const float dLdFhX = -dLdWarp * wg.y, dLdFhY = -dLdWarp * wg.z;
        const float dLddFX = dLdFhX + ( 2.0f * LAMBDA_DF * invN ) * dfX;
        const float dLddFY = dLdFhY + ( 2.0f * LAMBDA_DF * invN ) * dfY;
        const unsigned b = i * 4u;
        s_do[b + 0u] = conservativeBidir ? 0.0f : dLddFX * 2.0f * ( 1.0f - thX * thX );
        s_do[b + 1u] = conservativeBidir ? 0.0f : dLddFY * 2.0f * ( 1.0f - thY * thY );
        s_do[b + 2u] = do2;
        // History flag (mask 2) is absent here and rejected by the host.
        s_do[b + 3u] = 0.0f;
    }
    __syncthreads();

    // dW3/db3 and da2, while W3 is resident.
    for ( unsigned i = li; i < 580u; i += 64u )
    {
        float g = 0.0f;
        if ( i < 576u )
        {
            const int co = int( i ) / 144, rem = int( i ) % 144;
            const int tap = rem / C_MID, ci = rem % C_MID;
            for ( int p = 0; p < LT * LT; ++p )
            {
                const int px = p % LT, py = p / LT;
                const int ap = ( ( py + tap / 3 ) * T2 + ( px + tap % 3 ) ) * C_MID + ci;
                g += s_a2[ap] * s_do[p * 4 + co];
            }
        }
        else
        {
            const int co = int( i ) - 576;
            for ( int p = 0; p < LT * LT; ++p ) g += s_do[p * 4 + co];
        }
        gradient[W3_OFF + i] = g;
    }
    for ( unsigned i = li; i < unsigned( T2 * T2 * C_MID ); i += 64u )
    {
        const int ci = int( i ) % C_MID, pos = int( i ) / C_MID;
        const int px = pos % T2, py = pos / T2;
        float g = 0.0f;
        if ( !conservativeBidir && s_a2[i] > 0.0f )
        {
            for ( int co = 0; co < 3; ++co )
                for ( int tap = 0; tap < 9; ++tap )
                {
                    const int ox = px - tap % 3, oy = py - tap / 3;
                    if ( ox >= 0 && ox < LT && oy >= 0 && oy < LT )
                        g += s_w[( co * 9 + tap ) * C_MID + ci] * s_do[( oy * LT + ox ) * 4 + co];
                }
        }
        s_da2[i] = g;
    }
    __syncthreads();

    // dW2/db2 and da1 after reloading W2.
    for ( unsigned i = li; i < 2320u; i += 64u ) s_w[i] = job.state[W2_OFF + i];
    __syncthreads();
    for ( unsigned i = li; i < 2320u; i += 64u )
    {
        float g = 0.0f;
        if ( i < 2304u )
        {
            const int co = int( i ) / 144, rem = int( i ) % 144;
            const int tap = rem / C_MID, ci = rem % C_MID;
            for ( int p = 0; p < T2 * T2; ++p )
            {
                const int px = p % T2, py = p / T2;
                const int ap = ( ( py + tap / 3 ) * T1 + ( px + tap % 3 ) ) * C_MID + ci;
                g += s_a1[ap] * s_da2[p * C_MID + co];
            }
        }
        else
        {
            const int co = int( i ) - 2304;
            for ( int p = 0; p < T2 * T2; ++p ) g += s_da2[p * C_MID + co];
        }
        gradient[W2_OFF + i] = g;
    }
    for ( unsigned i = li; i < unsigned( T1 * T1 * C_MID ); i += 64u )
    {
        const int ci = int( i ) % C_MID, pos = int( i ) / C_MID;
        const int px = pos % T1, py = pos / T1;
        float g = 0.0f;
        if ( s_a1[i] > 0.0f )
        {
            for ( int co = 0; co < C_MID; ++co )
                for ( int tap = 0; tap < 9; ++tap )
                {
                    const int ox = px - tap % 3, oy = py - tap / 3;
                    if ( ox >= 0 && ox < T2 && oy >= 0 && oy < T2 )
                        g += s_w[( co * 9 + tap ) * C_MID + ci] * s_da2[( oy * T2 + ox ) * C_MID + co];
                }
        }
        s_da1[i] = g;
    }
    __syncthreads();

    // dW1/db1. No gradient atomics or alternate reduction tree.
    for ( unsigned i = li; i < 1744u; i += 64u )
    {
        float g = 0.0f;
        if ( i < 1728u )
        {
            const int co = int( i ) / 108, rem = int( i ) % 108;
            const int tap = rem / C_IN, ci = rem % C_IN;
            for ( int p = 0; p < T1 * T1; ++p )
            {
                const int px = p % T1, py = p / T1;
                const int fp = ( ( py + tap / 3 ) * T0 + ( px + tap % 3 ) ) * C_IN + ci;
                g += s_feat[fp] * s_da1[p * C_MID + co];
            }
        }
        else
        {
            const int co = int( i ) - 1728;
            for ( int p = 0; p < T1 * T1; ++p ) g += s_da1[p * C_MID + co];
        }
        gradient[W1_OFF + i] = g;
    }
}

// cs_framegen_motion_net_opt.comp: state is read-only here; a complete
// candidate [fast,m,v,EMA] is written for the caller's later commit decision.
__global__ void trainingOptimizer( TrainingKernelJob job )
{
    const unsigned p = blockIdx.x * blockDim.x + threadIdx.x;
    if ( p >= unsigned( N_PARAMS ) ) return;
    const float prior = job.prior[p];
    if ( job.step == 0u )
    {
        job.candidate[p] = prior;
        job.candidate[N_PARAMS + p] = 0.0f;
        job.candidate[2 * N_PARAMS + p] = 0.0f;
        job.candidate[3 * N_PARAMS + p] = prior;
        job.served[p] = prior;
        return;
    }

    float g = 0.0f;
    for ( int k = 0; k < N_SLICES; ++k ) g += job.gradients[k * N_PARAMS + p];
    g /= float( N_SLICES );
    if ( isnan( g ) || isinf( g ) ) g = 0.0f;
    else g = clampValue( g, -1.0f, 1.0f );

    float w = job.state[p], m = job.state[N_PARAMS + p], v = job.state[2 * N_PARAMS + p];
    if ( isnan( w ) || isinf( w ) || isnan( m ) || isinf( m ) || isnan( v ) || isinf( v ) )
    {
        w = prior; m = 0.0f; v = 0.0f;
    }
    m = 0.9f * m + 0.1f * g;
    v = 0.999f * v + 0.001f * g * g;
    const float t = float( job.step );
    const float mh = m / ( 1.0f - powf( 0.9f, t ) );
    const float vh = v / ( 1.0f - powf( 0.999f, t ) );
    w -= job.learningRate * mh / ( sqrtf( vh ) + 1e-8f );
    w -= job.priorDecay * ( w - prior );
    w = clampValue( w, -64.0f, 64.0f );
    if ( isnan( w ) || isinf( w ) || isnan( m ) || isinf( m ) || isnan( v ) || isinf( v ) )
    {
        w = prior; m = 0.0f; v = 0.0f;
    }
    job.candidate[p] = w;
    job.candidate[N_PARAMS + p] = m;
    job.candidate[2 * N_PARAMS + p] = v;
    float ema = job.state[3 * N_PARAMS + p];
    ema += job.emaAlpha * ( w - ema );
    job.candidate[3 * N_PARAMS + p] = ema;
    job.served[p] = ema;
}
} // namespace

hipError_t launch_training( const TrainingKernelJob &job, hipStream_t stream )
{
    // Host admission owns dimensions, finite-input checks and flags. No wait:
    // stream ordering makes the optimizer depend on all sixteen gradient rows.
    hipLaunchKernelGGL( trainingGradients, dim3( N_SLICES ), dim3( 64 ), 0, stream, job );
    hipError_t result = hipGetLastError();
    if ( result != hipSuccess ) return result;
    hipLaunchKernelGGL( trainingOptimizer, dim3( ( N_PARAMS + 63 ) / 64 ), dim3( 64 ), 0, stream, job );
    return hipGetLastError();
}
} // namespace gamescope::framegen::compute
