#pragma once

#include "backend_contract.h"

#include <stdexcept>
#include <tuple>

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

[[nodiscard]] constexpr bool backend_requirements_satisfied( const BackendRequirements &available,
	const BackendRequirements &required )
{
	return std::tie( required.operation, required.payload, required.execution,
		required.model_version, required.semantics, required.state )
		== std::tie( available.operation, available.payload, available.execution,
			available.model_version, available.semantics, available.state )
		&& ( available.transport & required.transport ) == required.transport
		&& ( available.features & required.features ) == required.features;
}

// This negotiation belongs at session creation, never in a submit/poll loop.
inline void require_backend_contract( const gsc_backend_contract &contract, const BackendRequirements &required )
{
	if ( !gsc_backend_contract_valid( &contract )
		|| !backend_requirements_satisfied( requirements_for( contract ), required ) )
		throw std::invalid_argument( "accelerator contract does not satisfy the typed operation" );
}
} // namespace gamescope::framegen::compute
