from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
import gymnasium as gym
import torch


class QLearning_RL_Agent(AbstractRLAgent):

    def __init__(self, env:gym.Env, gamma=1):
        super().__init__()
        self.number_states = env.observation_space.n
        self.number_actions = env.action_space.n

        self.Q = torch.zeros([self.number_states, self.number_actions])
        self.gamma = gamma


    def __QLeargning_bellman_decision_function__(self, state: State):
        ''' Environment characteristics '''
        random_values = self.Q[state.observation] + torch.rand(1, self.number_actions) / 1000
        action = torch.max(random_values, 1)[1][0]
        return action.item()


    def __QLearning_bellman_update__(self, old_state: State, new_observation: gym.Space, action, reward: float):
        '''Bellman equation update'''
        self.Q[old_state.observation, action] = reward + self.gamma * torch.max(self.Q[new_observation])

    def play(self, environment:gym.Env, max_steps=5000, seed=None):
        # Set action decision function
        # QLearning agent plays as a Q-Based policy sampling from Q-table
        # Q table updates following the deterministic scenario Bellman equation
        observation, info = environment.reset(seed=seed)
        self.initial_state = State(env=environment, observation=observation, info=info)
        self.current_state = None
        self.final_state = None
        self.step = 0
        self.__action_decision_function__ = self.__QLeargning_bellman_decision_function__
        self.__update_function__ = self.__QLearning_bellman_update__
        super().__play__(max_steps)
