from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
import gymnasium as gym
import torch


class StochasticQLearning_RL_Agent(AbstractRLAgent):

    def __init__(self, env:gym.Env, gamma=0.95, learning_rate=0.95):
        super().__init__()
        self.number_states = env.observation_space.n
        self.number_actions = env.action_space.n

        self.Q = torch.zeros([self.number_states, self.number_actions])
        self.gamma = gamma
        self.learning_rate = learning_rate


    def __StochasticQLearning_decision_function__(self, old_state: State):
        random_values = self.Q[old_state.observation] + torch.rand(1, self.number_actions) / 1000
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
        # Set action decision function
        # Stochastic QLearning agent plays as a Q-Based policy sampling from Q-table
        # Q table updates following the stochastic scenario Bellman equation
        observation, info = environment.reset(seed=seed)
        self.initial_state = State(env=environment, observation=observation, info=info)
        self.current_state = None
        self.final_state = None
        self.step = 0
        self.__action_decision_function__ = self.__StochasticQLearning_decision_function__
        self.__update_function__ = self.__StochasticQLearning_bellman_update__
        super().__play__(max_steps)
