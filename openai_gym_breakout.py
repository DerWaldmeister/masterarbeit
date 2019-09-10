import gym
import os
import numpy
import atari_py
from gym.envs.atari.atari_env import AtariEnv

env = gym.make('Breakout-ram-v0')

numEpisodes = 10000
maxStepsPerEpisode = 100


for episode in range(numEpisodes):
    env.reset()
    done = False

    for step in range(maxStepsPerEpisode):
        env.render()
        env.step(env.action_space.sample())



#print("Observation space:")
#print(env.observation_space)
#print(env.action_space)
