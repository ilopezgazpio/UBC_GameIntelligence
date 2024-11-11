from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
from SimpleBaselines.nn.NeuralNetwork import NeuralNetwork
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import math


class DeterministicDQN_RL_Agent(AbstractRLAgent):

    def __init__(self,
                 env:gym.Env,
                 seed=None,
                 gamma=0.95,
                 nn_learning_rate=0.001,
                 egreedy=0.9,
                 egreedy_final=0.02,
                 egreedy_decay=0.9999,
                 hidden_layers_size=[64],
                 activation_fn=nn.Tanh,
                 dropout=0.0,
                 use_batch_norm=False,
                 loss_fn=nn.MSELoss,
                 optimizer=optim.Adam
                 ):

        super().__init__(env=env)
        self.reset_env(seed=seed)

        # Set action decision function
        # DQN agent plays as a Q-Based policy sampling from an internal NN estimator
        # internal estimator NN updates following the stochastic scenario Bellman equation under specified loss function
        self.__action_decision_function__ = self.__DQN_decision_function__
        self.__update_function__ = self.__DQN_bellman_update__

        ''' Parameters for the DQN internal network '''
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

        print(self.QNetwork)

        ''' Parameters for the DQN agent '''
        self.gamma = gamma
        self.egreedy = egreedy
        self.egreedy_final = egreedy_final
        self.egreedy_decay = egreedy_decay


    def __DQN_decision_function__(self, old_state: State):
        ''' Epsilon greedy implementation '''
        random_for_egreedy = torch.rand(1)[0]

        if random_for_egreedy > self.egreedy:
            # Explotation
            # Observe Q and exploit best action MAX Q (S', A') as estimation of internal NN
            ''' MAX Q(S', A') '''

            with torch.no_grad():
                q_values = self.QNetwork.forward(old_state.observation)
                q_values = q_values.view(1, -1).cpu()
                q_values += torch.rand(1, self.env.action_space.n) / 1000
                action = torch.argmax(q_values, dim=1).item()

        else:

            # Exploration
            # pseudo random move
            with torch.no_grad():
                q_values = self.QNetwork.forward(old_state.observation)
                q_values = q_values.view(1, -1).cpu()
                probabilities = torch.softmax(q_values, dim=0)
                action = torch.multinomial(probabilities, 1).item()

        # Epsilon greedy weight decay
        if self.egreedy and self.egreedy_final and self.egreedy_decay and self.egreedy > self.egreedy_final:
            self.egreedy *= self.egreedy_decay

        return action


    def __DQN_bellman_update__(self, old_state: State, new_observation : gym.Space, action, reward: float):
        '''DQN Bellman equation update'''

        if old_state.terminated or old_state.truncated:
            # Terminated episode has known reward
            target_value = reward
        else:
            # NN is used to predict Q table for a given state --> Q[state, actions]
            # detach is used to avoid gradients at this time, we want to train NN with state, not with next_observation
            new_state_values = self.QNetwork(new_observation).detach()

            ''' Bellman equation for deterministic environment '''
            # Q[state, action] = reward + gamma * torch.max(Q[new_state])
            target_value = reward + self.gamma * torch.max(new_state_values)

        predicted_value = self.QNetwork(old_state.observation)[action]
        self.QNetwork.update_NN(predicted_value, target_value)



    def play(self, max_steps=5000, seed=None):
        self.step = 0
        super().__play__(max_steps)
