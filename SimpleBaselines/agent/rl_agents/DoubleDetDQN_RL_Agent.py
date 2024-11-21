from SimpleBaselines.states.State import State
from SimpleBaselines.agent.rl_agents.StableTargetNetDetDQN_RL_Agent import StableTargetNetDetDQN_RL_Agent

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


class DoubleDetDQN_RL_Agent(StableTargetNetDetDQN_RL_Agent):

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
                 optimizer=optim.Adam,
                 memory_size=50000,
                 batch_size=32,
                 target_net_update_steps=500,
                 clip_error=True
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
                         optimizer=optimizer,
                         memory_size=memory_size,
                         batch_size=batch_size,
                         target_net_update_steps=target_net_update_steps,
                         clip_error=clip_error
                         )

        ''' Parameters for the Double DQN agent '''
        # Set action decision function
        # Double DQN agent plays as a standard Deterministic DQN with experience replay with the following modification:
        # The target value is calculated using the QNetwork to select the best action and the Stable Target Net to evaluate it
        # i.e. Use learning net only to get indexes of actions (ignore the Q values)

        # internal estimator NN updates following the deterministic scenario Bellman equation under specified loss function
        # BUT training is more stable since parameters of training net are not always exposed at the time of optimizing

        # action decision function is inherited from superclass
        # self.__action_decision_function__ = self.__DQN_decision_function__

        # we need to update the update function to use experience replay
        self.__update_function__ = self.__Double_DQN_bellman_update__


    def __Double_DQN_bellman_update__(self, old_state: State, action, new_observation : gym.Space, reward: float, terminated, truncated):

        # Store experience in memory
        self.memory.push(old_state.observation, action, new_observation, reward, terminated, truncated)
        if len(self.memory) < self.batch_size:
            # not enough samples in memory
            return

        old_observation_batch, action_batch, new_observation_batch, reward_batch, terminated_batch, truncated_batch = self.memory.sample()

        '''DQN Bellman equation update USING STABLE NETWORK'''
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
            # Use learning net only to get indexes of actions (ignore the Q values)
            q_values_indexes = self.QNetwork(new_observation_batch)
            max_q_values_indexes = torch.max(q_values_indexes, 1)[1]

            # Produce Q values using stable net, but instead of taking max Q value
            # we use the index of the best actions from the learning net to get best Q values
            # This is expected to reduce overestimation of Q values
            q_values = self.stable_target_net(new_observation_batch)
            max_q_values = q_values.gather(1, max_q_values_indexes.unsqueeze(1)).squeeze(1)

        # Automatic broadcasting is used below
        target_value = reward_batch + (1 - (terminated_batch | truncated_batch)) * self.gamma * max_q_values

        predicted_value = self.QNetwork(old_observation_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        self.QNetwork.update_NN(predicted_value, target_value, self.clip_error)

        if self.update_target_counter % self.target_net_update_steps == 0:
            self.stable_target_net.network.load_state_dict(self.QNetwork.network.state_dict())

        self.update_target_counter += 1

