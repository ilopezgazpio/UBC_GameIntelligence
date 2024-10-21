import gymnasium as gym
from gymnasium import error, spaces, utils
import numpy as np

import logging
logger = logging.getLogger(__name__)



class PuzzleEnv(gym.Env):

    metadata = {
        'render.modes': ['human']
    }


    def __init__(self):
        '''
        Observation space is defined as a 3x3 matrix of positive numbers in the range [0-8]
        '''
        self.observation_space = spaces.Box(low=0, high=8, shape=(3,3), dtype=np.int8)

        '''
        Action space is defined as a discrete number with 4 actions
        0 --> Move 0 up
        1 --> Move 0 down
        2 --> Move 0 left
        3 --> Move 0 right
        '''
        self.action_space = spaces.Discrete(4)

        '''
        Environment state
        '''
        self.state = np.arange(9).reshape((3, 3))


    def step(self, action):

        info = {}

        ''' compute new state'''
        if self.is_applicable(action):
            observation = self.do_step(action)
            info["message"] = "Action performed"
        else:
            observation = self.state
            info["message"] = "Action not applicable"

        ''' compute episode end / reward '''
        terminated = self.episode_terminated()

        ''' todo '''
        ''' Gymnasium requires truncated to be explicitly handled, assuming same as terminated '''

        truncated = terminated

        if terminated:
            reward = 10
            info["message"] = "Episode finished"
        else:
            reward = -1

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.permutation(self.state.reshape(-1)).reshape(3, 3)
        return self.state, {}

    def render(self):
        ''' Mode rendering (not required by gymnasium, just using human for this case) '''
        print("\n{}\n".format(self.state))


    def close(self):
        return


    def is_applicable(self, action):

        index_array = np.where(self.state == 0)
        row = index_array[0][0]
        col = index_array[1][0]

        if action == 0: # Move 0 up
            possible = row != 0

        elif action == 1: # Move 0 down
            possible = row != 2

        elif action == 2: # Move 0 left
            possible = col != 0

        elif action == 3: # Move 0 right
            possible = col != 2

        return possible


    def do_step(self, action):
        index_array = np.where(self.state == 0)
        row = index_array[0][0]
        col = index_array[1][0]

        aux = np.copy(self.state)

        if action == 0: # Move 0 up
            self.state[row - 1, col] = aux[row, col]
            self.state[row, col] = aux[row - 1, col]

        elif action == 1: # Move 0 down
            self.state[row + 1, col] = aux[row, col]
            self.state[row, col] = aux[row + 1, col]

        elif action == 2: # Move 0 left
            self.state[row, col - 1] = aux[row, col]
            self.state[row, col] = aux[row, col - 1]

        elif action == 3: # Move 0 right
            self.state[row, col + 1] = aux[row, col]
            self.state[row, col] = aux[row, col + 1]

        return self.state


    def episode_terminated(self):
        final_state = np.array(np.arange(0,9)).reshape(3,3)
        return np.all(self.state == final_state)