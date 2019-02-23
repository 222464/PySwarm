import pyswarm

import numpy as np
import gym

env = gym.make('LunarLander-v2')

numInputs = env.observation_space.shape[0]
numActions = env.action_space.n

cs = pyswarm.PyComputeSystem(16, seed=np.random.randint(0, 99999))

inputSize = pyswarm.PyInt3(1, 1, numInputs)

lds = []

numLayers = 2

for i in range(numLayers):
    l = pyswarm.PyLayerDesc()

    l._layerType = "conv"
    l._filterRadius = 0 # Dense
    l._numMaps = 24 if i < numLayers - 1 else numActions
    l._recurrent = i < numLayers - 1
    l._actScalar = 6.0

    lds.append(l)

h = pyswarm.PyHierarchy(cs, inputSize, lds, 64)

h.setOptAlpha(0.001)
h.setOptEpsilon(1.0)

assert(h.getOutputSize().x * h.getOutputSize().y * h.getOutputSize().z >= numActions)

aStart = ((h.getOutputSize().x * h.getOutputSize().y) // 2) * h.getOutputSize().z

action = 0
reward = 0.0

episodeCount = 100000

maxEpisodeReward = -99999.0
averageReward = 0.0

for episode in range(episodeCount):
    obs = env.reset()

    totalReward = 0.0

    try:
        for t in range(1000):
            if episode % 200 == 199:
                env.render()

            padded = inputSize.x * inputSize.y * inputSize.z * [ 0.0 ]

            for i in range(numInputs):
                padded[i] = obs[i] * 6.0

            h.step(cs, padded, reward)

            action = 0
            maxValue = -99999.0

            for i in range(numActions):
                value = h.getOutputStates()[aStart + i]

                if value > maxValue:
                    maxValue = value
                    action = i
            
            obs, reward, done, info = env.step(action)
            
            totalReward += reward

            if done:
                maxEpisodeReward = max(maxEpisodeReward, totalReward)

                averageReward = 0.99 * averageReward + 0.01 * totalReward

                print("Episode {} finished after {} steps and {:.2f} reward. Max: {:.2f} Average: {:.2f}".format(episode + 1, t + 1, totalReward, maxEpisodeReward, averageReward))
    
                break

    except KeyboardInterrupt:
        print("Stopped by user.")
        break

print("Done.")

env.close()
