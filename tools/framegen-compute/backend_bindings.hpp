#pragma once

#include "framegen/backend_contract.hpp"
#include "training_plugin.h"

namespace gamescope::framegen::compute
{
class ResidentTrainer;
class VulkanEngine;
[[nodiscard]] constexpr gsc_backend_contract backend_contract( const ResidentTrainer & )
{
	return gsc_contract_resident_training();
}
[[nodiscard]] constexpr gsc_backend_contract backend_contract( const VulkanEngine & )
{
	return gsc_contract_generation();
}

// Cold negotiation returns the original concrete owner. There is no façade,
// vtable, function pointer, allocation or additional call in its submit loop.
// This is a borrowed reference: the caller retains the owner through retirement.
template<typename Owner>
[[nodiscard]] Owner &bind_backend( Owner &owner, const BackendRequirements &required )
{
	require_backend_contract( backend_contract( owner ), required );
	return owner;
}

// The snapshot v1 ABI remains frozen. Binding exposes that same immutable table
// with its native context/error arguments, not a resident-state approximation.
// The caller retains the table, module and context through actual retirement.
[[nodiscard]] inline const gsc_training_api &bind_snapshot_training( const gsc_training_api &api,
	const gsc_backend_contract &contract, const BackendRequirements &required )
{
	require_backend_contract( contract, required );
	if ( contract.payload != GSC_PAYLOAD_SNAPSHOT_TRAINING || api.size != sizeof( api )
		|| api.abi != GSC_TRAIN_ABI || api.model_version != GSC_TRAIN_MODEL_VERSION
		|| api.parameters != GSC_TRAIN_PARAMETERS || api.slices != GSC_TRAIN_SLICES
		|| !api.name || !api.devices || !api.create || !api.submit || !api.poll || !api.collect
		|| !api.discard || !api.retire || !api.reset || !api.destroy )
		throw std::invalid_argument( "incompatible snapshot-training callback table" );
	return api;
}
} // namespace gamescope::framegen::compute
