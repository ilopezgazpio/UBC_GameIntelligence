from abc import ABC, abstractmethod

import sys
import torch
import gymnasium
from gymnasium import spaces
from SimpleBaselines.states.State import State
from SimpleBaselines.reporting.RLReport import RLReport
import numpy as np


class AbstractRLAgent(ABC):

    def __init__(self, env:gym.Env):
        self.initial_state = None
        self.final_state = None
        self.current_state = None
        self.reporting = RLReport()
        self.wrapped_env = False

        '''Common Parameters for all RL agents'''
        self.env = env


    def __printlog__(self, print_every=100):
        if (self.current_state.step % print_every == 0):
            print(self.reporting.log.tail(1))


    def __play__(self, max_steps=5000, seed=None):
        '''
        This method implements the main loop and must be implemented with the particularities of each RL agent
        '''

        while not (self.current_state.terminated or self.current_state.truncated) and self.current_state.step <= max_steps:

            action = self.__action_decision_function__(self.current_state)

            # Take action
            observation, reward, terminated, truncated, info = self.env.step(action)

            # Update agent before removing old state
            # current_state is the old state (it will update next)
            # action is the performed action in self.current_state
            # observation is the new observation
            # reward is the reward obtained in self.current_state performing action
            # terminated and truncated refer to the game state after performing action on current state
            self.__update_function__(self.current_state, action, observation, reward, terminated, truncated)

            # Update current state
            self.current_state.observation = observation
            self.current_state.reward = reward
            self.current_state.terminated = terminated
            self.current_state.truncated = truncated
            self.current_state.info = info
            self.current_state.action = action
            self.current_state.action_history.append(action)
            self.current_state.cumulative_reward += reward

            self.reporting.__append__(self.current_state.terminated, self.current_state.truncated,
                                      self.current_state.reward, self.current_state.action, self.current_state.step,
                                      self.current_state.cumulative_reward)
            #self.__printlog__(print_every=10000)

            self.current_state.step += 1


        if self.current_state.terminated or self.current_state.truncated or self.current_state.step > max_steps:
            self.final_state = self.current_state

    def reset_env(self, seed=None):
        observation, info = self.env.reset(seed=seed)
        self.initial_state = State(observation=observation, info=info)
        self.current_state = self.initial_state
        self.final_state = None

    def __action_decision_function__(self, state: State):
        pass


    def __update_function__(self, old_state: State, action :gym.Space, new_observation : gym.Space, reward: float, terminated: bool, truncated: bool):
        pass

