#pragma once

#include "PyConstructs.h"
#include "PyComputeSystem.h"
#include <swarm/Hierarchy.h>
#include <swarm/OptimizerMAB.h>

namespace pyswarm {
    struct PyLayerDesc {
        std::string layerType; // Can be: "conv", "pool"

        // Conv
        int spatialFilterRadius;
        int spatialFilterStride;
        int recurrentFilterRadius;
        int numMaps;
        float actScalar;
        float recurrentScalar;

        // Pool
        int poolDiv;

        PyLayerDesc()
        : layerType("conv"), spatialFilterRadius(1), spatialFilterStride(1), recurrentFilterRadius(1), numMaps(16), actScalar(5.0f), recurrentScalar(0.5f), poolDiv(2)
        {}

        PyLayerDesc(const PyInt3 &stateSize, const std::string &layerType, int spatialFilterRadius, int spatialFilterStride, int recurrentFilterRadius, int numMaps, float actScalar, float recurrentScalar, int poolDiv)
        : layerType(layerType), spatialFilterRadius(spatialFilterRadius), spatialFilterStride(spatialFilterStride), recurrentFilterRadius(recurrentFilterRadius), numMaps(numMaps), actScalar(actScalar), recurrentScalar(recurrentScalar), poolDiv(poolDiv)
        {}
    };

    class PyHierarchy {
    private:
        std::vector<PyLayerDesc> layerDescs;

        swarm::Hierarchy h;
        swarm::OptimizerMAB opt;

    public:
        PyHierarchy(PyComputeSystem &cs, const PyInt3 &inputSize, const std::vector<PyLayerDesc> &layerDescs, int numArms);

        void step(PyComputeSystem &cs, const std::vector<float> &inputStates, float reward, bool learnEnabled = true);

        void save(const std::string &fileName);
        bool load(const std::string &fileName);

        int getNumLayers() {
            return h.getLayers().size();
        }

        const std::vector<float> &getOutputStates() {
            return h.getLayers().back()->getStates();
        }

        PyInt3 getOutputSize() {
            swarm::Int3 size = h.getLayers().back()->getStateSize();

            return PyInt3(size.x, size.y, size.z);
        }

        // Optimizer parameter getters/setters
        void setOptAlpha(float value) {
            opt.alpha = value;
        }

        void setOptEpsilon(float value) {
            opt.epsilon = value;
        }

        void setOptPlayTime(int value) {
            opt.playTime = value;
        }

        float getOptAlpha() const {
            return opt.alpha;
        }
        
        float getOptEpsilon() const {
            return opt.epsilon;
        }

        float getOptPlayTime() const {
            return opt.playTime;
        }
    };
}