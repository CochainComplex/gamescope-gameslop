#include "training_plugin.h"
#include "framegen/net_layout.hpp"
#include <bit>
#include <cstddef>

// v1 native plugin/ctypes consumers target the same 64-bit little-endian host.
static_assert( sizeof( void * ) == 8 && std::endian::native == std::endian::little );
static_assert( sizeof( gsc_train_device ) == 180 && sizeof( gsc_train_create ) == 188 );
static_assert( sizeof( gsc_train_ticket ) == 16 && sizeof( gsc_train_job ) == 128 );
static_assert( sizeof( gsc_train_output ) == 56 && sizeof( gsc_training_api ) == 104 );
static_assert( offsetof( gsc_train_job, epoch ) == 8 && offsetof( gsc_train_job, luma_count ) == 48 );
static_assert( offsetof( gsc_train_job, source ) == 80 && offsetof( gsc_train_job, state ) == 120 );
static_assert( offsetof( gsc_train_output, gradient_count ) == 8 && offsetof( gsc_train_output, gradients ) == 32 );
static_assert( GSC_TRAIN_PARAMETERS == k_uFramegenNetFloats && GSC_TRAIN_MODEL_VERSION == k_uFramegenNetVersion );
int main() { return 0; }
