import pyswarm

import numpy as np
import gym

env = gym.make('BipedalWalker-v2')

numInputs = env.observation_space.shape[0]
numActions = env.action_space.shape[0]

cs = pyswarm.PyComputeSystem(16, 1234)

inputSize = pyswarm.PyInt3(1, 1, numInputs)

lds = []

numLayers = 2

for i in range(numLayers):
    l = pyswarm.PyLayerDesc()

    l._layerType = "conv"
    l._filterRadius = 0 # Dense
    l._numMaps = 16 if i != numLayers - 1 else numActions
    l._recurrent = i != numLayers - 1
    l._actScalar = 6.0

    lds.append(l)

h = pyswarm.PyHierarchy(cs, inputSize, lds, 32)

h.setOptAlpha(0.001)
h.setOptEpsilon(1.0)

aStart = 0

reward = 0.0
rewardNoise = 0.0
averageReward = 0.0

episodeCount = 100000

maxEpisodeReward = -99999.0

for episode in range(episodeCount):
    try:
        obs = env.reset()

        totalReward = 0.0

        for t in range(10000):
            if episode % 200 == 199:
                env.render()

            padded = inputSize.x * inputSize.y * inputSize.z * [ 0.0 ]

            for i in range(numInputs):
                padded[i] = obs[i] * 8.0 # Scale input to be a bit more intense

            h.step(cs, padded, reward * (1.0 + rewardNoise * np.random.randn()))

            action = []

            for i in range(numActions):
                value = h.getOutputStates()[aStart + i] * 0.5 + 0.5

                action.append(value * (env.action_space.high[i] - env.action_space.low[i]) + env.action_space.low[i])

            obs, reward, done, info = env.step(np.array(action))

            totalReward += reward

            if done:
                break

        averageReward = 0.99 * averageReward + 0.01 * totalReward

        maxEpisodeReward = max(maxEpisodeReward, totalReward)

        print("Episode {} finished after {} steps and {:.3f} reward. Max: {:.3f} Average: {:.3f}".format(episode + 1, t + 1, totalReward, maxEpisodeReward, averageReward))

    except KeyboardInterrupt:
        print("Stopped by user.")
        break

print("Done.")

env.close()