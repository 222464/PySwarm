#pragma once

#include <swarm/ComputeSystem.h>
#include <random>
#include <iostream>

namespace pyswarm {
    class PyComputeSystem {
    private:
        swarm::ComputeSystem cs;

    public:
        PyComputeSystem(size_t numWorkers, unsigned long seed = 1234);

        friend class PyHierarchy;
    };
}