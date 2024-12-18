# Mixed experience DQN for Multi Discrete action space

from SimpleBaselines.agent.rl_agents.MixedExperienceDQN_RL_Agent import MixedExperienceDQN_RL_Agent
from SimpleBaselines.nn.MultiDiscreteNeuralNetwork import MultiDiscreteNeuralNetwork
from SimpleBaselines.states.State import State

import random
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

class MultiDQN(MixedExperienceDQN_RL_Agent):
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
                 data_size=10000,
                 batch_size=32,
                 target_net_update_steps=500,
                 clip_error=True,
                 data_amount_start = 1.0,
                 data_amount_end = 0.0,
                 data_amount_decay = 0.99
                 ):
                
                super().__init__(env=env,
                         seed=seed,
                         gamma=gamma,
                         nn_learning_rate=nn_learning_rate,
                         n_step=n_step,
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
                         data_size=data_size,
                         batch_size=batch_size,
                         target_net_update_steps=target_net_update_steps,
                         clip_error=clip_error,
                         data_amount_start = data_amount_start,
                         data_amount_end = data_amount_end,
                         data_amount_decay = data_amount_decay
                         )
                
                self.env = env

                self.__update_function__ = self.__Multi_DQN_Bellman_Update__
                self.__action_decision_function__ = self.__Multi_DQN_decision_function__



        def __init_NN__(self, seed, hidden_layers_size, activation_fn, dropout, use_batch_norm, loss_fn, optimizer, nn_learning_rate):
                return MultiDiscreteNeuralNetwork(
                    self.env.observation_space.shape[0],
                    self.env.action_space.nvec,
                    hidden_layers_size,
                    activation_fn,
                    nn_learning_rate,
                    dropout,
                    use_batch_norm,
                    loss_fn,
                    optimizer,
                    seed
                )
        
        def __Multi_DQN_decision_function__(self, state: State):
                ''' Epsilon greedy implementation '''

                if random.random() > self.egreedy:
                        # Explotation. Observe Q and exploit best action MAX Q (S', A') as estimation of internal NN
                        ''' MAX Q(S', A') '''
                        with torch.no_grad():
                                state_tensor = self.QNetwork.toDevice(state.observation)
                                q_values = self.QNetwork(state_tensor)
                                action = [torch.argmax(q_values[i]).item() for i in range(len(q_values))]

                else:
                # Exploration / pseudo random move
                        action = self.env.action_space.sample()

                return action

        

        def __Multi_DQN_Bellman_Update__(self, old_state: State, action, new_observation : gym.Space, reward: float, terminated, truncated):

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
                        # q_values_indexes.shape = (batch_size, n_vec, n_actions)
                        max_q_values_indexes = torch.argmax(q_values_indexes, dim=2)
                        # Produce Q values using stable net, but instead of taking max Q value
                        # we use the index of the best actions from the learning net to get best Q values
                        # This is expected to reduce overestimation of Q values
                        q_values = self.stable_target_net(new_observation_batch)
                        # q_values.shape = (batch_size, n_vec, n_actions)
                        max_q_values = torch.gather(q_values, 2, max_q_values_indexes.unsqueeze(2)).squeeze(2)

                # Automatic broadcasting is used below
                done_batch = (1- (terminated_batch | truncated_batch)).unsqueeze(1).repeat(1,4)
                target_value = reward_batch.unsqueeze(1).repeat(1,4) + done_batch * self.gamma * max_q_values
                old_q_values = self.QNetwork(old_observation_batch)
                predicted_value = torch.gather(old_q_values, 2, action_batch.unsqueeze(2)).squeeze(2)

                self.QNetwork.update_NN(predicted_value, target_value, self.clip_error)

                if self.update_target_counter % self.target_net_update_steps == 0:
                        self.stable_target_net.load_state_dict(self.QNetwork.state_dict())

                self.update_target_counter += 1

                