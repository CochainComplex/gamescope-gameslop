#pragma once

#include "backend_contract.h"

#include <stdexcept>

namespace gamescope::framegen::compute
{
struct BackendRequirements
{
	uint32_t operation, payload, execution, model_version, semantics, state, transport, features;
};

[[nodiscard]] constexpr BackendRequirements requirements_for( const gsc_backend_contract &contract )
{
	return { contract.operation, contract.payload, contract.execution, contract.model_version,
		contract.semantics, contract.state, contract.transport, contract.features };
}

// This negotiation belongs at session creation, never in a submit/poll loop.
inline void require_backend_contract( const gsc_backend_contract &contract, const BackendRequirements &required )
{
	if ( !gsc_backend_contract_valid( &contract ) || required.operation != contract.operation
		|| required.payload != contract.payload || required.execution != contract.execution
		|| required.model_version != contract.model_version || required.semantics != contract.semantics
		|| required.state != contract.state || ( contract.transport & required.transport ) != required.transport
		|| ( contract.features & required.features ) != required.features )
		throw std::invalid_argument( "accelerator contract does not satisfy the typed operation" );
}
} // namespace gamescope::framegen::compute
