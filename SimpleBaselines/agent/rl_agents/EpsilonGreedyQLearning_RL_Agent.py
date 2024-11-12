from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
import gymnasium as gym
import torch


class EpsilonGreedyQLearning_RL_Agent(AbstractRLAgent):

    def __init__(self, env:gym.Env, seed=None, gamma=0.95, learning_rate=0.99, egreedy=0.7, egreedy_final=0.1, egreedy_decay=0.9999):
        super().__init__(env)
        self.reset_env(seed=seed)

        '''Parameters for the Epsilon-Greedy Q-Learning algorithm'''
        self.Q = torch.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.egreedy = egreedy
        self.egreedy_final = egreedy_final
        self.egreedy_decay = egreedy_decay

        # Set action decision function
        # Stochastic QLearning agent plays as a Q-Based policy sampling from Q-table
        # Q table updates following the stochastic scenario Bellman equation
        self.__action_decision_function__ = self.__EpsilonGreedyQLearning_decision_function__
        self.__update_function__ = self.__EpsilonGreedyQLearning_bellman_update__


    def __EpsilonGreedyQLearning_decision_function__(self, old_state: State):
        ''' Epsilon greedy implementation '''
        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > self.egreedy:
            # Explotation
            # Observe Q and exploit best action MAX Q (S', A')
            ''' MAX Q(S', A') '''
            random_values = self.Q[old_state.observation] + torch.rand(1, self.env.action_space.n) / 1000
            action = torch.argmax(random_values, dim=1).item()

        else:
            # Exploration
            # pseudo random move
            q_values = self.Q[old_state.observation]
            probabilities = torch.softmax(q_values, dim=0)
            action = torch.multinomial(probabilities, 1).item()

        # Simple epsilon greedy weight decay
        if self.egreedy and self.egreedy_final and self.egreedy_decay and self.egreedy > self.egreedy_final:
            self.egreedy *= self.egreedy_decay

        return action


    def __EpsilonGreedyQLearning_bellman_update__(self, old_state: State, new_observation : gym.Space, action, reward: float):
        '''Epsilon greedy Bellman equation update
           Current memory cell (left part of Stochastic Q-learning equation) '''
        current_memory = self.Q[old_state.observation, action]

        ''' Input gate cell (right part of Q-learning equation) '''
        input_gate = reward + self.gamma * torch.max(self.Q[new_observation])

        ''' Bellman equation for stochastic environment '''
        self.Q[old_state.observation, action] = (1 - self.learning_rate) * current_memory + self.learning_rate * input_gate


    def play(self, max_steps=5000, seed=None):
        super().__play__(max_steps)
