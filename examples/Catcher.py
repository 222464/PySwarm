import pyswarm

import numpy as np
import gym
import gym_ple
import lycon
from copy import copy

import scipy.misc

env = gym.make('Catcher-v0')

imageSize = (16, 16)
numActions = env.action_space.n

cs = pyswarm.PyComputeSystem(16)

inputSize = pyswarm.PyInt3(imageSize[0], imageSize[1], 1)

lds = []

numLayers = 5

for i in range(numLayers):
    l = pyswarm.PyLayerDesc()

    if i % 2 == 0:
        l._layerType = "conv"
        l._filterRadius = 2
        l._numMaps = 4 + i * 2 if i < numLayers - 1 else numActions
        l._recurrent = True if i == numLayers - 2 else False
        l._actScalar = 6.0
    else:
        l._layerType = "pool"
        l._poolDiv = 2

    lds.append(l)

h = pyswarm.PyHierarchy(cs, inputSize, lds, 32)

h.setOptAlpha(0.001)
h.setOptEpsilon(1.0)

minSize = min(env.observation_space.shape[0], env.observation_space.shape[1])
maxSize = max(env.observation_space.shape[0], env.observation_space.shape[1])

obsPrev = np.zeros((minSize, minSize, 3), dtype=np.float32)

print("Num actions: " + str(numActions))

print("Output size: " + str(h.getOutputSize().x) + ", " + str(h.getOutputSize().y) + ", " + str(h.getOutputSize().z))

assert(h.getOutputSize().x * h.getOutputSize().y * h.getOutputSize().z >= numActions)

aStart = ((h.getOutputSize().x * h.getOutputSize().y) // 2) * h.getOutputSize().z

reward = 0.0
averageReward = 0.0
rNoise = 0.0

episodeCount = 20000

for episode in range(episodeCount):
    try:
        obs = env.reset()

        totalReward = 0.0

        for t in range(10000):
            env.render()

            obs = obs.astype(dtype=np.float32) / 255.0#np.swapaxes(obs.astype(dtype=np.float32) / 255.0, 0, 1)

            obs = obs[:, maxSize // 2 - minSize // 2 : maxSize // 2 + minSize // 2, :]
            
            obsPrev = copy(obs)

            obs = (obs[:, :, 0] + obs[:, :, 1] + obs[:, :, 2]) / 3.0

            obs = lycon.resize(obs, width=imageSize[0], height=imageSize[1], interpolation=lycon.Interpolation.CUBIC)

            h.step(cs, (obs * 6.0).ravel().tolist(), reward)
            
            maxValue = -99999.0
            action = 0

            for i in range(numActions):
                value = h.getOutputStates()[aStart + i]

                if value > maxValue:
                    maxValue = value
                    action = i

            obs, reward, done, info = env.step(action)

            totalReward += reward

            if done:
                break

        averageReward = 0.9 * averageReward + 0.1 * totalReward
    
        print("Episode {} finished after {} timesteps, gathering {} reward. Average reward: {}".format(episode + 1, t + 1, totalReward, averageReward))
    
    except KeyboardInterrupt:
        print("Stopped by user.")
        break

print("Done.")

env.close()