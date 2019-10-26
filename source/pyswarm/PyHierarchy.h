#pragma once

#include "PyConstructs.h"
#include "PyComputeSystem.h"
#include <swarm/Hierarchy.h>
#include <swarm/OptimizerMAB.h>

namespace pyswarm {
    struct PyLayerDesc {
        std::string _layerType; // Can be: "conv", "pool"

        // Conv
        int _spatialFilterRadius;
        int _spatialFilterStride;
        int _recurrentFilterRadius;
        int _numMaps;
        float _actScalar;
        float _recurrentScalar;

        // Pool
        int _poolDiv;

        PyLayerDesc()
        : _layerType("conv"), _spatialFilterRadius(1), _spatialFilterStride(1), _recurrentFilterRadius(1), _numMaps(16), _actScalar(4.0f), _recurrentScalar(0.5f), _poolDiv(2)
        {}

        PyLayerDesc(const PyInt3 &stateSize, const std::string &layerType, int spatialFilterRadius, int spatialFilterStride, int recurrentFilterRadius, int numMaps, float actScalar, float recurrentScalar, int poolDiv)
        : _layerType(layerType), _spatialFilterRadius(spatialFilterRadius), _spatialFilterStride(spatialFilterStride), _recurrentFilterRadius(recurrentFilterRadius), _numMaps(numMaps), _actScalar(actScalar), _recurrentScalar(recurrentScalar), _poolDiv(poolDiv)
        {}
    };

    class PyHierarchy {
    private:
        std::vector<PyLayerDesc> _layerDescs;

        swarm::Hierarchy _h;
        swarm::OptimizerMAB _opt;

    public:
        PyHierarchy(PyComputeSystem &cs, const PyInt3 &inputSize, const std::vector<PyLayerDesc> &layerDescs, int numArms);

        void step(PyComputeSystem &cs, const std::vector<float> &inputStates, float reward, bool learnEnabled = true);

        void save(const std::string &fileName);
        bool load(const std::string &fileName);

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

        void setOptPlayTime(int value) {
            _opt._playTime = value;
        }

        float getOptAlpha() const {
            return _opt._alpha;
        }
        
        float getOptEpsilon() const {
            return _opt._epsilon;
        }

        float getOptPlayTime() const {
            return _opt._playTime;
        }
    };
}