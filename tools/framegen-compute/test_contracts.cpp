#include "framegen/compute_job.hpp"

#include <iostream>
#include <limits>

using namespace gamescope::framegen::compute;

namespace
{
int failures = 0;
void check( bool condition, const char *message )
{
	if ( !condition ) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

void testOwnership()
{
	Batch batch;
	check( !batch.busy() && batch.epoch() == 1, "initial idle epoch" );
	check( batch.begin( 0 ).serial == 0 && batch.begin( 2 ).serial == 0, "invalid epoch rejected" );
	const Ticket first = batch.begin( batch.epoch() );
	check( first.serial != 0 && batch.owns( first ), "first ticket owned" );
	const Ticket wrongSerial{ first.epoch, first.serial + 1 };
	const Ticket wrongEpoch{ first.epoch + 1, first.serial };
	for ( Ticket wrong : { Ticket{}, wrongSerial, wrongEpoch } )
	{
		batch.complete( wrong );
		batch.discard( wrong );
		check( !batch.retire( wrong ) && !batch.owns( wrong ), "wrong ticket cannot retire or own" );
	}
	check( batch.busy() && !batch.ready( first ) && !batch.discarded( first ), "wrong ticket has no effect" );
	batch.discard( first );
	check( batch.discarded( first ) && !batch.ready( first ), "discard does not imply completion" );
	check( !batch.retire( first ) && !batch.reset(), "incomplete discard cannot retire or reset" );
	check( batch.begin( batch.epoch() ).serial == 0, "incomplete discard cannot reuse storage" );
	batch.complete( first );
	check( batch.ready( first ) && batch.busy(), "completion retains ownership" );
	check( !batch.reset() && batch.begin( batch.epoch() ).serial == 0, "completion alone cannot reset or reuse" );
	check( batch.retire( first ) && !batch.busy(), "retirement releases completed discard" );
	check( !batch.retire( first ) && !batch.ready( first ), "retired ticket cannot replay" );
	const Ticket second = batch.begin( batch.epoch() );
	check( second.serial > first.serial && !batch.discarded( second ), "serial increases and discard does not leak" );
	batch.complete( first );
	batch.discard( first );
	check( !batch.ready( second ) && !batch.discarded( second ), "old serial cannot alter new batch" );
	batch.complete( second );
	check( batch.retire( second ) && batch.reset(), "normal completion retires before reset" );
	check( batch.epoch() == first.epoch + 1 && batch.begin( first.epoch ).serial == 0, "reset rejects previous epoch" );
	const Ticket third = batch.begin( batch.epoch() );
	check( third.serial > second.serial && third.epoch != second.epoch, "reset preserves monotonic serial and changes epoch" );
	batch.complete( second );
	batch.discard( second );
	check( !batch.retire( second ) && !batch.ready( third ) && !batch.discarded( third ), "stale epoch has no effect" );
	batch.complete( third );
	check( batch.retire( third ), "final batch retired" );
}

void testPair()
{
	const Pair valid{ 1, 10, 11, 1000, 1400, 1100 };
	check( phase( valid ) == 0.25f, "valid quarter phase" );
	auto reject = []( Pair pair, const char *message ) { check( !phase( pair ), message ); };
	Pair pair = valid; pair.epoch = 0; reject( pair, "zero epoch" );
	pair = valid; pair.previousId = 0; reject( pair, "zero previous ID" );
	pair = valid; pair.currentId = pair.previousId; reject( pair, "duplicate content ID" );
	pair = valid; pair.currentId = pair.previousId - 1; reject( pair, "reversed content IDs" );
	pair = valid; pair.currentNs = pair.previousNs; reject( pair, "equal content timestamps" );
	pair = valid; pair.currentNs = pair.previousNs - 1; reject( pair, "reversed content timestamps" );
	pair = valid; pair.targetNs = pair.previousNs; reject( pair, "previous endpoint target" );
	pair = valid; pair.targetNs = pair.currentNs; reject( pair, "current endpoint target" );
	pair = valid; pair.targetNs = pair.previousNs - 1; reject( pair, "target before pair" );
	pair = valid; pair.targetNs = pair.currentNs + 1; reject( pair, "target after pair" );
	pair = valid; pair.timeline = Timeline::Causal; reject( pair, "causal timeline unsupported" );
	pair = { 1, 10, 11, 1, 100000001, 100000000 };
	reject( pair, "interior timestamp rounded to endpoint phase" );
	const uint64_t high = std::numeric_limits<uint64_t>::max();
	pair = { 1, 10, 11, high - 400, high, high - 100 };
	check( phase( pair ) == 0.75f, "large monotonic timestamps preserve short interval" );
	pair = { 1, 10, 11, high - 4, high, high - 3 };
	check( phase( pair ) == 0.25f, "single nanosecond offset at large timestamps" );
	pair = { 1, 10, 11, 0, high, high / 2 };
	check( phase( pair ) == 0.5f, "wide unsigned interval does not overflow" );
}
} // namespace

int main()
{
	testOwnership();
	testPair();
	if ( failures ) return 1;
	std::cout << "compute contracts PASS\n";
	return 0;
}
