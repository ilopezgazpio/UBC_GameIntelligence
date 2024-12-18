from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
from SimpleBaselines.agent.rl_agents.DeterministicDQN_RL_Agent import DeterministicDQN_RL_Agent
from SimpleBaselines.nn.ConvolutionalNetwork import ConvolutionalNetwork
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import sys

class DeterministicDQConvN_RL_Agent(DeterministicDQN_RL_Agent):

    def __init__(self,
                 env:gym.Env,
                 seed=None,
                 gamma=0.99,
                 nn_learning_rate=0.01,
                 egreedy=0.9,
                 egreedy_final=0.02,
                 egreedy_decay=500,
                 hidden_layers_size=[(32, 3, 1),(64,3, 1)],
                 activation_fn=nn.ReLU,
                 dropout=0.0,
                 use_batch_norm=False,
                 loss_fn=nn.MSELoss,
                 optimizer=optim.Adam
                 ):

        super().__init__(env=env,
                         seed=seed,
                         gamma=gamma,
                         nn_learning_rate=0.01,
                         egreedy=0.9,
                         egreedy_final=0.02,
                         egreedy_decay=500,
                         hidden_layers_size=[64],
                         activation_fn=nn.Tanh,
                         dropout=0.0,
                         use_batch_norm=False,
                         loss_fn=nn.MSELoss,
                         optimizer=optim.Adam
                         )

        self.QNetwork = self.__init_NN__(seed, hidden_layers_size, activation_fn, dropout, use_batch_norm, loss_fn, optimizer, nn_learning_rate)

        def __init_NN__(self, seed, hidden_layers_size, activation_fn, dropout, use_batch_norm, loss_fn, optimizer, nn_learning_rate):
            return ConvolutionalNetwork(
                self.env.observation_space.shape,
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
