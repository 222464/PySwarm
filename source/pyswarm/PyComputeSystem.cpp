#include "PyComputeSystem.h"

using namespace pyswarm;

PyComputeSystem::PyComputeSystem(size_t numWorkers, unsigned long seed)
: _cs(numWorkers)
{
    _cs._rng.seed(seed);
}