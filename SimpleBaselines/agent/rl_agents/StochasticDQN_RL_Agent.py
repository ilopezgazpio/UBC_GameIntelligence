from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
from SimpleBaselines.nn.NeuralNetwork import NeuralNetwork
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import sys

class StochasticDQN_RL_Agent(AbstractRLAgent):

    def __init__(self,
                 env:gym.Env,
                 seed=None,
                 gamma=0.99,
                 nn_learning_rate=0.01,
                 Q_learning_rate=0.95,
                 egreedy=0.9,
                 egreedy_final=0.02,
                 egreedy_decay=500,
                 hidden_layers_size=[64],
                 activation_fn=nn.Tanh,
                 dropout=0.0,
                 use_batch_norm=False,
                 loss_fn=nn.MSELoss,
                 optimizer=optim.Adam
                 ):

        super().__init__(env=env)

        # Set action decision function
        # DQN agent plays as a Q-Based policy sampling from an internal NN estimator
        # internal estimator NN updates following the stochastic scenario Bellman equation under specified loss function
        self.__action_decision_function__ = self.__DQN_decision_function__
        self.__update_function__ = self.__DQN_bellman_update__

        ''' Parameters for the DQN agent '''
        self.total_steps = 0
        self.gamma = gamma
        self.egreedy = egreedy
        self.egreedy_final = egreedy_final
        self.egreedy_decay = egreedy_decay
        self.Q_learning_rate = Q_learning_rate


        self.QNetwork = NeuralNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            hidden_layers_size,
            activation_fn,
            nn_learning_rate,
            dropout,
            use_batch_norm,
            loss_fn,
            optimizer,
            seed
        )


    def update_egreedy(self):
        # Epsilon greedy weight decay
        self.total_steps += 1
        if self.egreedy and self.egreedy_final and self.egreedy_decay and self.egreedy > self.egreedy_final:
            self.egreedy = self.egreedy_final + ( self.egreedy - self.egreedy_final ) * math.exp( -1. * self.total_steps / self.egreedy_decay )


    def __DQN_decision_function__(self, state: State):
        ''' Epsilon greedy implementation '''

        if random.random() > self.egreedy:
            # Explotation. Observe Q and exploit best action MAX Q (S', A') as estimation of internal NN
            ''' MAX Q(S', A') '''
            with torch.no_grad():
                state_tensor = self.QNetwork.toDevice(state.observation)
                q_values = self.QNetwork(state_tensor)
                action = torch.argmax(q_values).item()

        else:
            # Exploration / pseudo random move
             action = self.env.action_space.sample()

        return action


    def __DQN_bellman_update__(self, old_state: State, action, new_observation : gym.Space, reward: float, terminated, truncated):

        '''DQN Bellman equation update'''
        old_observation_tensor = self.QNetwork.toDevice(old_state.observation)
        new_observation_tensor = self.QNetwork.toDevice(new_observation)
        reward = self.QNetwork.toDevice([reward])

        if terminated or truncated:
            # Terminated episode has known reward
            target_value = reward
        else:
            # NN is used to predict Q table for a given state --> Q[state, actions]
            # detach is used to avoid gradients at this time, we want to train NN with state, not with next_observation
             with torch.no_grad():
                 ''''Stochastic Bellman equation update
                 Current memory cell (left part of Stochastic Q-learning equation) '''
                 current_memory = self.QNetwork(old_observation_tensor)[action]

                 ''' Input gate cell (right part of Q-learning equation) '''
                 input_gate = reward + self.gamma * torch.max( self.QNetwork(new_observation_tensor ))

                 ''' Bellman equation for stochastic environment '''
                 target_value = (1 - self.Q_learning_rate) * current_memory + self.Q_learning_rate * input_gate

        predicted_value = self.QNetwork(old_observation_tensor)[action]

        self.QNetwork.update_NN(predicted_value, target_value)



    def play(self, max_steps=5000, seed=None):
        super().__play__(max_steps)
        self.update_egreedy()
