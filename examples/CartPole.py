import pyswarm

import numpy as np
import gym

env = gym.make('CartPole-v0')

numInputs = env.observation_space.shape[0]

cs = pyswarm.PyComputeSystem(16, seed=np.random.randint(0, 99999))

inputSize = pyswarm.PyInt3(1, 1, numInputs)

lds = []

numLayers = 2

for i in range(numLayers):
    l = pyswarm.PyLayerDesc()

    l._layerType = "conv"
    l._filterRadius = 0 # Dense
    l._numMaps = 16 if i < numLayers - 1 else 1
    l._recurrent = i < numLayers - 1
    l._actScalar = 6.0

    lds.append(l)

h = pyswarm.PyHierarchy(cs, inputSize, lds, 32)

h.setOptAlpha(0.01)
h.setOptEpsilon(0.5)

action = 0
reward = 0.0
rewardNoise = 0.0

episodeCount = 20000

for episode in range(episodeCount):
    try:
        obs = env.reset()

        for t in range(1000):
            padded = inputSize.x * inputSize.y * inputSize.z * [ 0.0 ]

            for i in range(numInputs):
                padded[i] = obs[i] * 6.0 # Scale inputs to be more intense

            h.step(cs, padded, reward * (1.0 + rewardNoise * np.random.randn()))

            action = int(h.getOutputStates()[0] > 0.0)

            obs, _, done, info = env.step(action)
            
            reward = 0.0

            if done:
                reward = -1.0

            if done:
                print("Episode {} finished after {} timesteps".format(episode + 1, t + 1))

                break
        
    except KeyboardInterrupt:
        print("Stopped by user.")
        break

print("Done.")

env.close()