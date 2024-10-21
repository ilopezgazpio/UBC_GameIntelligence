import gymnasium as gym
import numpy as np

'''
Node class is a wrapper for an environment including additional features for observations / states
'''
class Node:

    def __init__(self,
                 env: gym.Env = None,
                 observation = None,
                 reward = 0,
                 terminated = False,
                 truncated = False,
                 info = None,
                 utility = 0,
                 cost = 0, # accumulative sum of rewards
                 action_history = [],
                 depth = 0):

        self.env = env
        self.observation = observation
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info
        self.utility = utility
        self.cost = cost
        self.action_history = action_history
        self.depth = depth