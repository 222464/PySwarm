import pyswarm

import numpy as np
import gym
import pybullet
import pybullet_envs.bullet.minitaur_gym_env as e

env = e.MinitaurBulletEnv(render=True)

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
    l._numMaps = 32 if i != numLayers - 1 else numActions
    l._recurrent = i != numLayers - 1
    l._actScalar = 6.0

    lds.append(l)

h = pyswarm.PyHierarchy(cs, inputSize, lds, 32)

h.setOptAlpha(0.01)
h.setOptEpsilon(1.0)

aStart = 0

reward = 0.0
averageReward = 0.0

episodeCount = 100000

maxEpisodeReward = -99999.0

for episode in range(episodeCount):
    try:
        obs = env.reset()

        totalReward = 0.0

        for t in range(1000):
            padded = inputSize.x * inputSize.y * inputSize.z * [ 0.0 ]

            for i in range(numInputs):
                padded[i] = obs[i] * 6.0

            averageReward = 0.995 * averageReward + 0.005 * reward

            h.step(cs, padded, averageReward)

            action = []

            for i in range(numActions):
                value = h.getOutputStates()[aStart + i] * 0.5 + 0.5

                action.append(value * (env.action_space.high[i] - env.action_space.low[i]) + env.action_space.low[i])

            obs, reward, done, info = env.step(np.array(action))

            reward *= 100.0

            if done:
                reward += -10.0
            
            totalReward += reward

            if done:
                break

        maxEpisodeReward = max(maxEpisodeReward, totalReward)

        print("Episode {} finished after {} steps and {:.3f} reward. Max: {:.3f} Average: {:.3f}".format(episode + 1, t + 1, totalReward, maxEpisodeReward, averageReward))

    except KeyboardInterrupt:
        print("Stopped by user.")
        break

print("Done.")

env.close()