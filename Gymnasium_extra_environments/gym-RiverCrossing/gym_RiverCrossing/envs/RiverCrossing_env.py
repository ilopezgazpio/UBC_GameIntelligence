import gymnasium as gym
from gymnasium import error, spaces, utils
import numpy as np

import logging
logger = logging.getLogger(__name__)



class RiverCrossingEnv(gym.Env):

    metadata = {
        'render.modes': ['human']
    }


    def __init__(self):
        '''
        Observation space is defined as a tuple of 4 columns
        Column 0 --> Position of farmer
        Column 1 --> Position of wolf
        Column 2 --> Position of goat
        Column 3 --> Position of cabbage

        For each dimension value 0 means element in left side of river and value 1 means element in right side of river
        '''
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8)

        '''
        Action space is defined as a discrete number with 4 actions
        0 --> Move farmer
        1 --> Move wolf
        2 --> Move goat
        3 --> Move cabbage
        '''
        self.action_space = spaces.Discrete(4)

        '''
        Environment state
        '''
        self.state = np.array((0, 0, 0, 0), dtype=np.int8)


    def step(self, action):

        info = {}

        ''' compute new state'''
        if self._is_applicable_(action):
            observation = self._do_step_(action)
            info["message"] = "Action performed"
        else:
            observation = self.state
            info["message"] = "Action not applicable"

        ''' compute episode end / reward '''
        terminated = self._episode_terminated_()

        '''todo'''
        ''' Gymnasium requires both terminated and truncated '''
        truncated = terminated

        if terminated:
            reward = 10
            info["message"] = "Episode finished"
        else:
            reward = -1

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array((0, 0, 0, 0), dtype=np.int8)
        return self.state, {}


    def render(self):
        ''' Render the environment's state in human-readable form '''
        print("##########\nFarmer:{}\nWolf:{}\nGoat:{}\nCabbage:{}\n##########".format(self.state[0], self.state[1], self.state[2], self.state[3]))


    def close(self):
        return


    def _is_applicable_(self, action):

        if action == 0: # farmer
            possible = (self.state[1] != self.state[2]) and (self.state[2] != self.state[3])

        elif action == 1: # wolf
            possible = (self.state[2] != self.state[3]) and (self.state[0] == self.state[1])

        elif action == 2: # goat
            possible = self.state[0] == self.state[2]

        elif action == 3: # cabbage
            possible = (self.state[1] != self.state[2]) and (self.state[0] == self.state[3])

        return possible


    def _do_step_(self, action):
        # Farmer always moves
        self.state[0] -= 1

        # Move along with some other element
        if action > 0:
            self.state[action] -= 1

        self.state = np.abs(self.state)
        return self.state


    def _episode_terminated_(self):
        return np.all(self.state == 1)