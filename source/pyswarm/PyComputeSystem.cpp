#include "PyComputeSystem.h"

using namespace pyswarm;

PyComputeSystem::PyComputeSystem(
    unsigned long seed
) {
    cs.rng.seed(seed);
}

void PyComputeSystem::setNumThreads(
    int numThreads
) {
    swarm::ComputeSystem::setNumThreads(numThreads);
}

int PyComputeSystem::getNumThreads() {
    return swarm::ComputeSystem::getNumThreads();
}