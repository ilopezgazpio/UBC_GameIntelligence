from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
import gymnasium as gym
import torch


class QLearning_RL_Agent(AbstractRLAgent):

    def __init__(self, env:gym.Env, seed=None, gamma=1):
        super().__init__(env)
        self.reset_env(seed=seed)

        ''' Parameters for the Q-Learning algorithm '''
        self.Q = torch.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.gamma = gamma

        # Set action decision function
        # QLearning agent plays as a Q-Based policy sampling from Q-table
        # Q table updates following the deterministic scenario Bellman equation
        self.__action_decision_function__ = self.__QLeargning_bellman_decision_function__
        self.__update_function__ = self.__QLearning_bellman_update__


    def __QLeargning_bellman_decision_function__(self, state: State):
        ''' Environment characteristics '''
        random_values = self.Q[state.observation] + torch.rand(1, self.env.action_space.n) / 1000
        action = torch.max(random_values, 1)[1][0]
        return action.item()


    def __QLearning_bellman_update__(self, old_state: State, new_observation: gym.Space, action, reward: float):
        '''Bellman equation update'''
        self.Q[old_state.observation, action] = reward + self.gamma * torch.max(self.Q[new_observation])


    def play(self, max_steps=5000, seed=None):
        super().__play__(max_steps)

