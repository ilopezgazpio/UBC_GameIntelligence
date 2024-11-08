import gymnasium as gym
import numpy as np

'''
State class is a wrapper for an environment including additional features for observations
'''
class State:

    def __init__(self,
                 observation = None,
                 info=None,
                 step = 0,
                 action = 0,
                 reward = 0,
                 cumulative_reward = 0,
                 terminated = False,
                 truncated = False,
                 utility = 0,
                 cost = 0, # accumulative sum of rewards
                 action_history = [],
                 ):

        self.observation = observation
        self.info = info
        self.step = step
        self.action = action
        self.reward = reward
        self.cumulative_reward = cumulative_reward
        self.terminated = terminated
        self.truncated = truncated
        self.utility = utility
        self.cost = cost
        self.action_history = action_history
