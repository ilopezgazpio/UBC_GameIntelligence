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


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(4, 64)
        self.linear2 = nn.Linear(64, 2)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        return self.linear2(x)


class DeterministicDQN_RL_Agent(AbstractRLAgent):

    def __init__(self,
                 env:gym.Env,
                 seed=None,
                 gamma=0.99,
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
                 ):

        super().__init__(env=env)
        self.reset_env(seed=seed)
        self.total_steps = 0

        # Set action decision function
        # DQN agent plays as a Q-Based policy sampling from an internal NN estimator
        # internal estimator NN updates following the stochastic scenario Bellman equation under specified loss function
        self.__action_decision_function__ = self.__DQN_decision_function__
        self.__update_function__ = self.__DQN_bellman_update__

        ''' Parameters for the DQN internal network '''


        '''
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
        '''

        # Replicability
        torch.manual_seed(seed)
        random.seed(seed)

        # Set device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")

        self.learning_rate = nn_learning_rate
        self.QNetwork = NeuralNetwork().to(self.device)
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.QNetwork.parameters(), lr=nn_learning_rate)

        print(self.QNetwork)

        ''' Parameters for the DQN agent '''
        self.gamma = gamma
        self.egreedy = egreedy
        self.egreedy_final = egreedy_final
        self.egreedy_decay = egreedy_decay


    def update_egreedy(self):
        # Epsilon greedy weight decay
        self.total_steps += 1
        if self.egreedy and self.egreedy_final and self.egreedy_decay and self.egreedy > self.egreedy_final:
            self.egreedy = self.egreedy_final + (
                    self.egreedy - self.egreedy_final
            ) * math.exp( -1. * self.total_steps / self.egreedy_decay )


    def __DQN_decision_function__(self, state: State):
        ''' Epsilon greedy implementation '''

        if random.random() > self.egreedy:
            # Explotation. Observe Q and exploit best action MAX Q (S', A') as estimation of internal NN
            ''' MAX Q(S', A') '''
            with torch.no_grad():
                state_tensor = torch.tensor(state.observation, dtype=torch.float32).to(self.device)
                q_values = self.QNetwork(state_tensor)
                action = torch.argmax(q_values).item()

        else:
            # Exploration
            # pseudo random move
             action = self.env.action_space.sample()

        return action


    def __DQN_bellman_update__(self, old_state: State, action, new_observation : gym.Space, reward: float, terminated, truncated):
        '''DQN Bellman equation update'''

        old_observation_tensor = torch.tensor(old_state.observation, dtype=torch.float32).to(self.device)
        new_observation_tensor = torch.tensor(new_observation, dtype=torch.float32).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)

        if terminated or truncated:
            # Terminated episode has known reward
            target_value = reward
        else:
            # NN is used to predict Q table for a given state --> Q[state, actions]
            # detach is used to avoid gradients at this time, we want to train NN with state, not with next_observation
             with torch.no_grad():
                 ''' Bellman equation for deterministic environment '''
                 # Q[state, action] = reward + gamma * torch.max(Q[new_state])
                 q_values = self.QNetwork(new_observation_tensor)
                 target_value = reward + self.gamma * torch.max(q_values)

        predicted_value = self.QNetwork(old_observation_tensor)[action]
        loss = self.loss_func(predicted_value, target_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




    def play(self, max_steps=5000, seed=None):
        super().__play__(max_steps)
        self.total_steps += self.current_state.step
