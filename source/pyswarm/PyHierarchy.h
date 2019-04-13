#pragma once

#include "PyConstructs.h"
#include "PyComputeSystem.h"
#include <swarm/Hierarchy.h>
#include <swarm/OptimizerMAB.h>

namespace pyswarm {
    struct PyLayerDesc {
        std::string _layerType; // Can be: "conv", "pool"

        // Conv
        int _filterRadius;
        int _stride;
        int _numMaps;
        bool _recurrent;
        float _actScalar;

        // Pool
        int _poolDiv;

        PyLayerDesc()
        : _layerType("conv"), _filterRadius(1), _stride(1), _numMaps(16), _recurrent(false), _actScalar(8.0f), _poolDiv(2)
        {}

        PyLayerDesc(const PyInt3 &stateSize, const std::string &layerType, int filterRadius, int stride, int numMaps, bool recurrent, float biasScale, float actScalar, int poolDiv)
        : _layerType(layerType), _filterRadius(filterRadius), _stride(stride), _numMaps(numMaps), _recurrent(recurrent), _actScalar(actScalar), _poolDiv(poolDiv)
        {}
    };

    class PyHierarchy {
    private:
        swarm::Hierarchy _h;
        swarm::OptimizerMAB _opt;

    public:
        PyHierarchy(PyComputeSystem &cs, const PyInt3 &inputSize, const std::vector<PyLayerDesc> &layerDescs, int numArms);

        void step(PyComputeSystem &cs, const std::vector<float> &inputStates, float reward, bool learnEnabled = true);

        int getNumLayers() {
            return _h.getLayers().size();
        }

        const std::vector<float> &getOutputStates() {
            return _h.getLayers().back()->getStates();
        }

        PyInt3 getOutputSize() {
            swarm::Int3 size = _h.getLayers().back()->getStateSize();

            return PyInt3(size.x, size.y, size.z);
        }

        // Optimizer parameter getters/setters
        void setOptAlpha(float value) {
            _opt._alpha = value;
        }

        void setOptEpsilon(float value) {
            _opt._epsilon = value;
        }

        float getOptAlpha() const {
            return _opt._alpha;
        }

        float getOptEpsilon() const {
            return _opt._epsilon;
        }
    };
}