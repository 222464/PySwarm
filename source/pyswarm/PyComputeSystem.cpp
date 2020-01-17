#include "PyComputeSystem.h"

using namespace pyswarm;

PyComputeSystem::PyComputeSystem(size_t numWorkers, unsigned long seed)
: cs(numWorkers)
{
    cs.rng.seed(seed);
}