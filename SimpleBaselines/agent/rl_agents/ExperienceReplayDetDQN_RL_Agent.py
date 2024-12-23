from SimpleBaselines.states.State import State
from SimpleBaselines.agent.rl_agents.DeterministicDQN_RL_Agent import DeterministicDQN_RL_Agent
from SimpleBaselines.memory.NaiveExperienceReplay import NaiveExperienceReplay

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


class ExperienceReplayDetDQN_RL_Agent(DeterministicDQN_RL_Agent):

    def __init__(self,
                 env:gym.Env,
                 seed=None,
                 gamma=0.99,
                 n_step=1,
                 nn_learning_rate=0.01,
                 egreedy=0.9,
                 egreedy_final=0.02,
                 egreedy_decay=500,
                 hidden_layers_size=[64],
                 activation_fn=nn.Tanh,
                 dropout=0.0,
                 use_batch_norm=False,
                 loss_fn=nn.MSELoss,
                 optimizer=optim.Adam,
                 memory_size=50000,
                 batch_size=32
                 ):

        super().__init__(env=env,
                         seed=seed,
                         gamma=gamma,
                         nn_learning_rate=nn_learning_rate,
                         egreedy=egreedy,
                         egreedy_final=egreedy_final,
                         egreedy_decay=egreedy_decay,
                         hidden_layers_size=hidden_layers_size,
                         activation_fn=activation_fn,
                         dropout=dropout,
                         use_batch_norm=use_batch_norm,
                         loss_fn=loss_fn,
                         optimizer=optimizer
                         )


        ''' Parameters for the Experience Replay DQN agent '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.n_step_buffer = []

        # Create memory buffer
        self.memory = NaiveExperienceReplay(memory_size=self.memory_size, batch_size=self.batch_size)

        # Set action decision function
        # Experience Replay Deterministic DQN agent plays as a standard Deterministic DQN
        # internal estimator NN updates following the deterministic scenario Bellman equation under specified loss function

        # action decision function is inherited from superclass
        # self.__action_decision_function__ = self.__DQN_decision_function__

        # we need to update the update function to use experience replay
        self.__update_function__ = self.__ER_DQN_bellman_update__


    def __ER_DQN_bellman_update__(self, old_state: State, action, new_observation : gym.Space, reward: float, terminated, truncated):

        # Store n-step transitions
        if len(self.n_step_buffer) < self.n_step:
            self.n_step_buffer.append((old_state, action, new_observation, reward, terminated, truncated))
            return
        else:
            exp = old_state, action, new_observation, reward, terminated, truncated
            # Add n-step reward and new_state to the oldest transition and add it to the memory
            old_state, action, _, reward, terminated, truncated = self.n_step_buffer.pop(0)
            # Update reward with n-step reward
            reward += sum([self.gamma ** i * r for i, (_, _, _, r, _, _) in enumerate(self.n_step_buffer, 1)])
            self.n_step_buffer.append(exp)

        # Store experience in memory
        if terminated or truncated:
            for j, transition in enumerate(self.n_step_buffer):
                old_state, action, _, reward, terminated, truncated = transition
                reward += sum([self.gamma ** i-j * r for i, (_, _, _, r, _, _) in enumerate(self.n_step_buffer, j+1)])
                self.memory.push(old_state.observation, action, new_observation, reward, terminated, truncated)
            self.n_step_buffer.clear()
        
        else: self.memory.push(old_state.observation, action, new_observation, reward, terminated, truncated)
        if len(self.memory) < self.batch_size:
            # not enough samples in memory
            return

        old_observation_batch, action_batch, new_observation_batch, reward_batch, terminated_batch, truncated_batch = self.memory.sample()

        '''DQN Bellman equation update'''
        old_observation_batch = self.QNetwork.toDevice(old_observation_batch)
        action_batch = self.QNetwork.toDevice(action_batch, dType=torch.int64)
        new_observation_batch = self.QNetwork.toDevice(new_observation_batch)
        reward_batch = self.QNetwork.toDevice(reward_batch)
        terminated_batch = self.QNetwork.toDevice(terminated_batch, dType=torch.uint8)
        truncated_batch = self.QNetwork.toDevice(truncated_batch, dType=torch.uint8)

        # NN is used to predict Q table for a given state --> Q[state, actions]
        # detach is used to avoid gradients at this time, we want to train NN with state, not with next_observation
        with torch.no_grad():
            ''' Bellman equation for deterministic environment '''
            # Q[state, action] = reward + gamma * torch.max(Q[new_state])
            q_values = self.QNetwork(new_observation_batch)

        # Automatic broadcasting is used below
        target_value = reward_batch + (1 - (terminated_batch | truncated_batch)) * self.gamma**self.n_step * torch.max(q_values, 1)[0]

        predicted_value = self.QNetwork(old_observation_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        self.QNetwork.update_NN(predicted_value, target_value)

