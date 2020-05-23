#pragma once

#include <swarm/ComputeSystem.h>
#include <random>
#include <iostream>

namespace pyswarm {
class PyComputeSystem {
private:
    swarm::ComputeSystem cs;

public:
    PyComputeSystem(
        unsigned long seed = 1234
    );

    static void setNumThreads(
        int numThreads
    );

    static int getNumThreads();

    friend class PyHierarchy;
};
} // namespace pyswarm