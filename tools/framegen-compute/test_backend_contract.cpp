#ifdef NDEBUG
#undef NDEBUG
#endif
#include "framegen/backend_contract.hpp"
#include "backend_bindings.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

using namespace gamescope::framegen::compute;

namespace
{
using Contract = gsc_backend_contract;
static_assert( std::is_standard_layout_v<Contract> && std::is_trivially_copyable_v<Contract> );
static_assert( sizeof( Contract ) == 48 );
static_assert( std::is_same_v<decltype( gsc_contract_resident_training() ), Contract> );
static_assert( offsetof( Contract, size ) == 0 && offsetof( Contract, abi ) == 4 );
static_assert( offsetof( Contract, operation ) == 8 && offsetof( Contract, execution ) == 12 );
static_assert( offsetof( Contract, payload ) == 16 && offsetof( Contract, model_version ) == 20 );
static_assert( offsetof( Contract, semantics ) == 24 && offsetof( Contract, state ) == 28 );
static_assert( offsetof( Contract, transport ) == 32 && offsetof( Contract, features ) == 36 );
static_assert( offsetof( Contract, max_inflight ) == 40 && offsetof( Contract, reserved ) == 44 );

const std::array<std::pair<const char *, Contract>, 4> Profiles{{
	{ "resident_training", gsc_contract_resident_training() },
	{ "snapshot_training", gsc_contract_snapshot_training() },
	{ "npu_raw_head", gsc_contract_npu_raw_head() },
	{ "generation", gsc_contract_generation() } }};

void rejects( const Contract &contract, const BackendRequirements &requirements )
{
	bool failed = false;
	try { require_backend_contract( contract, requirements ); }
	catch ( const std::invalid_argument & ) { failed = true; }
	assert( failed );
}

void contracts()
{
	for ( const auto &[name, contract] : Profiles )
	{
		(void)name;
		const auto requirements = requirements_for( contract );
		require_backend_contract( contract, requirements );
		auto minimal = requirements; minimal.features = 0;
		require_backend_contract( contract, minimal );
		// Independent C POD roundtrip preserves every public word.
		std::array<uint32_t, 12> words{};
		std::memcpy( words.data(), &contract, sizeof( contract ) );
		Contract restored{}; std::memcpy( &restored, words.data(), sizeof( restored ) );
		require_backend_contract( restored, requirements );
		assert( words[0] == 48 && words[11] == 0 );
		for ( size_t field = 0; field < words.size(); ++field )
		{
			auto malformed = words;
			malformed[field] = field == 0 ? 47u : UINT32_MAX;
			Contract bad{}; std::memcpy( &bad, malformed.data(), sizeof( bad ) );
			rejects( bad, requirements );
		}
		auto unavailable = requirements; unavailable.features |= 0x80000000u;
		rejects( contract, unavailable );
		for ( auto member : { &BackendRequirements::operation, &BackendRequirements::payload,
			&BackendRequirements::execution, &BackendRequirements::model_version,
			&BackendRequirements::semantics, &BackendRequirements::state, &BackendRequirements::transport } )
		{
			auto wrong = requirements; wrong.*member = UINT32_MAX;
			rejects( contract, wrong );
		}
	}
	const auto resident = gsc_contract_resident_training(), snapshot = gsc_contract_snapshot_training();
	const auto npu = gsc_contract_npu_raw_head(), generation = gsc_contract_generation();
	assert( resident.operation == snapshot.operation && resident.payload != snapshot.payload );
	assert( resident.execution == snapshot.execution && resident.model_version == 3 );
	assert( resident.state != snapshot.state );
	assert( npu.operation != resident.operation && npu.execution != resident.execution );
	assert( generation.operation != resident.operation && generation.operation != npu.operation );
	// A valid schema for another operation is still not a matching binding.
	for ( const auto &[name, actual] : Profiles )
		for ( const auto &[wantedName, wanted] : Profiles )
		{
			(void)name; (void)wantedName;
			if ( actual.operation != wanted.operation || actual.state != wanted.state )
				rejects( actual, requirements_for( wanted ) );
		}
}

void dump()
{
	std::cout << '{';
	bool first = true;
	for ( const auto &[name, contract] : Profiles )
	{
		if ( !first ) std::cout << ',';
		first = false;
		std::array<uint32_t, 12> words{}; std::memcpy( words.data(), &contract, sizeof( contract ) );
		std::cout << '"' << name << "\":[";
		for ( size_t i = 0; i < words.size(); ++i ) { if ( i ) std::cout << ','; std::cout << words[i]; }
		std::cout << ']';
	}
	std::cout << "}\n";
}

int devices( gsc_train_device *, uint32_t, uint32_t *, char *, uint32_t ) { return 0; }
int create( const gsc_train_create *, void **, char *, uint32_t ) { return 0; }
int submit( void *, const gsc_train_job *, gsc_train_ticket *, char *, uint32_t ) { return 0; }
int ticketOp( void *, gsc_train_ticket, char *, uint32_t ) { return 23; }
int collect( void *, gsc_train_ticket, gsc_train_output *, char *, uint32_t ) { return 0; }
int reset( void *, uint64_t *, char *, uint32_t ) { return 0; }
int destroy( void *, char *, uint32_t ) { return 0; }

void snapshotBinding()
{
	static_assert( std::is_same_v<decltype( bind_snapshot_training(
		std::declval<const gsc_training_api &>(), std::declval<const Contract &>(),
		std::declval<const BackendRequirements &>() ) ), const gsc_training_api &> );
	const gsc_training_api api{ sizeof( gsc_training_api ), GSC_TRAIN_ABI, GSC_TRAIN_MODEL_VERSION,
		GSC_TRAIN_PARAMETERS, GSC_TRAIN_SLICES, "CPU callback identity fixture",
		devices, create, submit, ticketOp, collect, ticketOp, ticketOp, reset, destroy };
	const auto contract = gsc_contract_snapshot_training();
	const auto required = requirements_for( contract );
	const auto &binding = bind_snapshot_training( api, contract, required );
	assert( &binding == &api && binding.submit == api.submit && binding.poll == api.poll );
	assert( binding.poll( nullptr, { 1, 2 }, nullptr, 0 ) == 23 );
	auto rejectTable = [&]( const gsc_training_api &bad ) {
		bool rejected = false;
		try { (void)bind_snapshot_training( bad, contract, required ); }
		catch ( const std::invalid_argument & ) { rejected = true; }
		assert( rejected );
	};
	for ( auto member : { &gsc_training_api::size, &gsc_training_api::abi,
		&gsc_training_api::model_version, &gsc_training_api::parameters, &gsc_training_api::slices } )
	{
		auto bad = api; bad.*member = UINT32_MAX; rejectTable( bad );
	}
	{ auto bad = api; bad.name = nullptr; rejectTable( bad ); }
#define NULL_CALLBACK_CASE( member ) { auto bad = api; bad.member = nullptr; rejectTable( bad ); }
	NULL_CALLBACK_CASE( devices )
	NULL_CALLBACK_CASE( create )
	NULL_CALLBACK_CASE( submit )
	NULL_CALLBACK_CASE( poll )
	NULL_CALLBACK_CASE( collect )
	NULL_CALLBACK_CASE( discard )
	NULL_CALLBACK_CASE( retire )
	NULL_CALLBACK_CASE( reset )
	NULL_CALLBACK_CASE( destroy )
#undef NULL_CALLBACK_CASE
	// A resident descriptor does not turn the frozen snapshot callback table
	// into a resident owner, even when the descriptor itself is coherent.
	bool rejected = false;
	const auto resident = gsc_contract_resident_training();
	try { (void)bind_snapshot_training( api, resident, requirements_for( resident ) ); }
	catch ( const std::invalid_argument & ) { rejected = true; }
	assert( rejected );
}
} // namespace

int main( int argc, char **argv )
{
	contracts();
	snapshotBinding();
	if ( argc == 2 && std::string_view( argv[1] ) == "--dump" ) dump();
	else if ( argc == 1 ) std::cout << "backend contract CPU checks PASS\n";
	else return 2;
	std::cout.flush();
	return std::cout ? 0 : 1;
}
