#include "PyHierarchy.h"

#include <swarm/LayerConv.h>
#include <swarm/LayerPool.h>

#include <fstream>

using namespace pyswarm;

PyHierarchy::PyHierarchy(PyComputeSystem &cs, const PyInt3 &inputSize, const std::vector<PyLayerDesc> &layerDescs, int numArms) {
    _layerDescs = layerDescs;

    std::vector<std::shared_ptr<swarm::Layer>> layers(layerDescs.size());

    swarm::Int3 sizePrev = swarm::Int3(inputSize.x, inputSize.y, inputSize.z);

    for (int i = 0; i < layers.size(); i++) {
        if (layerDescs[i]._layerType == "conv") {
            std::shared_ptr<swarm::LayerConv> l = std::make_shared<swarm::LayerConv>();

            l->create(cs._cs, sizePrev, layerDescs[i]._numMaps, layerDescs[i]._spatialFilterRadius, layerDescs[i]._spatialFilterStride, layerDescs[i]._recurrentFilterRadius);

            l->_actScalar = layerDescs[i]._actScalar;
            l->_recurrentScalar = layerDescs[i]._recurrentScalar;
            
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

void PyHierarchy::save(const std::string &fileName) {
    std::ofstream os(fileName);

    int numLayers = _h.getNumParameters().size();
    std::vector<swarm::FloatBuffer*> params = _h.getParameters();

    for (int i = 0; i < numLayers; i++) {
        const swarm::FloatBuffer* states = &_h.getLayers()[i]->getStates();
        const swarm::FloatBuffer* statesPrev;
        
        if (_layerDescs[i]._layerType == "conv")
            statesPrev = &static_cast<swarm::LayerConv*>(_h.getLayers()[i].get())->getStatesPrev();
        else
            statesPrev = nullptr;

        os.write(reinterpret_cast<const char*>(states->data()), states->size() * sizeof(float));

        if (statesPrev != nullptr)
            os.write(reinterpret_cast<const char*>(statesPrev->data()), statesPrev->size() * sizeof(float));

        if (!params[i]->empty())
            os.write(reinterpret_cast<const char*>(params[i]->data()), params[i]->size() * sizeof(float));
    }

    for (int i = 0; i < _opt.getIndices.size(); i++) {
        os.write(reinterpret_cast<const char*>(_opt.getValues()[i].data()), _opt.getValues()[i].size() * sizeof(float));
        os.write(reinterpret_cast<const char*>(_opt.getIndices()[i].data()), _opt.getIndices()[i].size() * sizeof(int));
    }
}

bool PyHierarchy::load(const std::string &fileName) {
    std::ifstream is(fileName);

    if (!is.is_open())
        return false;

    int numLayers = _h.getNumParameters().size();
    std::vector<swarm::FloatBuffer*> params = _h.getParameters();

    for (int i = 0; i < numLayers; i++) {
        swarm::FloatBuffer* states = &_h.getLayers()[i]->getStates();
        swarm::FloatBuffer* statesPrev;
        
        if (_layerDescs[i]._layerType == "conv")
            statesPrev = &static_cast<swarm::LayerConv*>(_h.getLayers()[i].get())->getStatesPrev();
        else
            statesPrev = nullptr;

        is.read(reinterpret_cast<char*>(states->data()), states->size() * sizeof(float));

        if (statesPrev != nullptr)
            is.read(reinterpret_cast<char*>(statesPrev->data()), statesPrev->size() * sizeof(float));

        if (!params[i]->empty())
            is.read(reinterpret_cast<char*>(params[i]->data()), params[i]->size() * sizeof(float));
    }

    for (int i = 0; i < _opt.getIndices.size(); i++) {
        is.read(reinterpret_cast<char*>(_opt.getValues()[i].data()), _opt.getValues()[i].size() * sizeof(float));
        is.read(reinterpret_cast<char*>(_opt.getIndices()[i].data()), _opt.getIndices()[i].size() * sizeof(int));
    }

    return true;
}