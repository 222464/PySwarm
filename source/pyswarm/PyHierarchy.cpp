#include "PyHierarchy.h"

#include <swarm/LayerConv.h>
#include <swarm/LayerPool.h>

using namespace pyswarm;

PyHierarchy::PyHierarchy(PyComputeSystem &cs, const PyInt3 &inputSize, const std::vector<PyLayerDesc> &layerDescs, int numArms) {
    std::vector<std::shared_ptr<swarm::Layer>> layers(layerDescs.size());

    swarm::Int3 sizePrev = swarm::Int3(inputSize.x, inputSize.y, inputSize.z);

    for (int i = 0; i < layers.size(); i++) {
        if (layerDescs[i]._layerType == "conv") {
            std::shared_ptr<swarm::LayerConv> l = std::make_shared<swarm::LayerConv>();

            l->create(cs._cs, sizePrev, layerDescs[i]._numMaps, layerDescs[i]._spatialFilterRadius, layerDescs[i]._spatialFilterStride, layerDescs[i]._recurrentFilterRadius, layerDescs[i]._recurrentFilterStride);

            l->_actScalar = layerDescs[i]._actScalar;
            
            layers[i] = std::static_pointer_cast<swarm::Layer>(l);

            sizePrev = l->getStateSize();
        }
        else if (layerDescs[i]._layerType == "pool") {
            std::shared_ptr<swarm::LayerPool> l = std::make_shared<swarm::LayerPool>();

            l->create(cs._cs, sizePrev, layerDescs[i]._poolDiv);

            layers[i] = std::static_pointer_cast<swarm::Layer>(l);

            sizePrev = l->getStateSize();
        }
        else {
            std::cerr << "Unrecognized layer type: " << layerDescs[i]._layerType << " (layer " << i << ")" << std::endl;
            abort();
        }
    }

    _h.create(layers);

    _opt.create(cs._cs, _h.getNumParameters(), numArms);
}

void PyHierarchy::step(PyComputeSystem &cs, const std::vector<float> &inputStates, float reward, bool learnEnabled) {
    _h.activate(cs._cs, inputStates);

    if (learnEnabled)
        _h.optimize(cs._cs, &_opt, reward);
}