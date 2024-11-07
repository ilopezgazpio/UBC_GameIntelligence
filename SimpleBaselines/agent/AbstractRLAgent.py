from abc import ABC, abstractmethod

import gym
import torch
import gymnasium
from gymnasium import spaces
from SimpleBaselines.states.State import State
from SimpleBaselines.reporting.RLReport import RLReport
import numpy as np


class AbstractRLAgent(ABC):

    def __init__(self):
        self.initial_state = None
        self.final_state = None
        self.current_state = None
        self.reporting = RLReport()
        self.wrapped_env = False

        ''' Parameters for the Q-Learning algorithm '''
        self.Q = None
        self.gamma = None
        self.number_states = None
        self.number_actions = None

        '''Parameters for the Stochastic Q-Learning algorithm'''
        self.learning_rate = None

        '''Parameters for the Epsilon-Greedy Q-Learning algorithm'''
        self.egreedy = None
        self.egreedy_final = None
        self.egreedy_decay = None


        '''Parameters for the Value Iteration Learning algorithm'''
        self.V = None
        self.V_init_steps = 2000
        self.policy = None


        ''' Parameters for the DQN algorithm '''
        self.hidden_layer_size = None
        self.input_layer_size = None
        self.output_layer_size = None

    def __printlog__(self, print_every=100):
        if (self.current_state.step % print_every == 0):
            print(self.reporting.log.tail(1))


    def __play__(self, max_steps=5000):
        '''
        This method implements the main loop and must be implemented with the particularities of each RL agent
        '''

        self.current_state = self.initial_state

        while not (self.current_state.terminated or self.current_state.truncated) and self.current_state.step <= max_steps:

            action = self.__action_decision_function__(self.current_state)

            # Take action
            observation, reward, terminated, truncated, info = self.current_state.env.step(action)

            # Update agent before removing old state
            # current_state is the old state (it will update next)
            # observation is the new observation
            # action is the performed action in current_state
            # reward is the reward obtained in current_state performing action
            self.__update_function__(self.current_state, observation, action, reward)

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
            self.__printlog__(print_every=10000)

            self.current_state.step += 1


        if self.current_state.terminated or self.current_state.truncated or self.current_state.step > max_steps:
            self.final_state = self.current_state


    def __action_decision_function__(self, state: State):
        pass


    def __update_function__(self, old_state: State, new_observation : gym.Space, action, reward: float):
        pass

