from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
import gymnasium as gym
import torch


class StochasticQLearning_RL_Agent(AbstractRLAgent):

    def __init__(self, env:gym.Env, seed=None, gamma=0.95, learning_rate=0.95):
        super().__init__(env)
        self.reset_env(seed=seed)

        '''Parameters for the Stochastic Q-Learning algorithm'''
        self.Q = torch.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.gamma = gamma
        self.learning_rate = learning_rate

        # Set action decision function
        # Stochastic QLearning agent plays as a Q-Based policy sampling from Q-table
        # Q table updates following the stochastic scenario Bellman equation
        self.__action_decision_function__ = self.__StochasticQLearning_decision_function__
        self.__update_function__ = self.__StochasticQLearning_bellman_update__

    def __StochasticQLearning_decision_function__(self, old_state: State):
        random_values = self.Q[old_state.observation] + torch.rand(1, self.env.action_space.n) / 1000
        ''' MAX Q(S', A') '''
        action = torch.max(random_values, 1)[1][0]
        return action.item()


    def __StochasticQLearning_bellman_update__(self, old_state: State, new_observation : gym.Space, action, reward: float):
        '''Stochastic Bellman equation update
           Current memory cell (left part of Stochastic Q-learning equation) '''
        current_memory = self.Q[old_state.observation, action]

        ''' Input gate cell (right part of Q-learning equation) '''
        input_gate = reward + self.gamma * torch.max(self.Q[new_observation])

        ''' Bellman equation for stochastic environment '''
        self.Q[old_state.observation, action] = (1 - self.learning_rate) * current_memory + self.learning_rate * input_gate


    def play(self, environment:gym.Env, max_steps=5000, seed=None):
        self.step = 0
        super().__play__(max_steps)
