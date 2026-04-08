from SimpleBaselines.states.State import State
from SimpleBaselines.agent.rl_agents.StochasticDQN_RL_Agent import StochasticDQN_RL_Agent
from SimpleBaselines.memory.PrioritizedExperienceReplay import PrioritizedExperienceReplay

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

class PERStocDQN_RL_Agent(StochasticDQN_RL_Agent):

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
                 optimizer=optim.Adam,
                 memory_size=50000,
                 batch_size=32, 
                 per_alpha=0.3,
                 per_beta=0.4,
                 per_beta_increment=0.0005,
                 per_epsilon=0.01,
                 per_variant='proportional'
                 ):

        super().__init__(env=env,
                         seed=seed,
                         gamma=gamma,
                         nn_learning_rate=nn_learning_rate,
                         Q_learning_rate=Q_learning_rate,
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
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment
        self.per_epsilon = per_epsilon
        self.per_variant = per_variant

        # Create memory buffer
        self.memory = PrioritizedExperienceReplay(
            memory_size=self.memory_size, 
            batch_size=self.batch_size,
            alpha=self.per_alpha, 
            beta=self.per_beta,
            beta_increment=self.per_beta_increment,
            epsilon=self.per_epsilon,
            variant=self.per_variant
            )

        # Set action decision function
        # Experience Replay Stochastic DQN agent plays as a standard Stochastic DQN
        # internal estimator NN updates following the stochastic scenario Bellman equation under specified loss function

        # action decision function is inherited from superclass
        # self.__action_decision_function__ = self.__DQN_decision_function__
        
        # we need to update the update function to use experience replay
        self.__update_function__ = self.__ER_DQN_bellman_update__


    def __ER_DQN_bellman_update__(self, old_state: State, action, new_observation : gym.Space, reward: float, terminated, truncated):

        self.memory.push(old_state.observation, action, new_observation, reward, terminated, truncated)
                
        if len(self.memory) < self.batch_size:
            # not enough samples in memory
            return

        old_observation_batch, action_batch, new_observation_batch, reward_batch, terminated_batch, truncated_batch, weights, indices = self.memory.sample()

        '''DQN Bellman equation update'''
        old_observation_batch = self.QNetwork.toDevice(old_observation_batch)
        action_batch = self.QNetwork.toDevice(action_batch, dType=torch.int64)
        new_observation_batch = self.QNetwork.toDevice(new_observation_batch)
        reward_batch = self.QNetwork.toDevice(reward_batch)
        terminated_batch = self.QNetwork.toDevice(terminated_batch, dType=torch.uint8)
        truncated_batch = self.QNetwork.toDevice(truncated_batch, dType=torch.uint8)
        weights = self.QNetwork.toDevice(weights, dType=torch.float32)

        # NN is used to predict Q table for a given state --> Q[state, actions]
        # detach is used to avoid gradients at this time, we want to train NN with state, not with next_observation
        '''Stochastic Bellman equation update
        Current memory cell (left part of Stochastic Q-learning equation)'''
        current_memory = self.QNetwork(old_observation_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            '''Input gate cell (right part of Q-learning equation)'''
            q_next = self.QNetwork(new_observation_batch)
            input_gate = reward_batch + (1 - (terminated_batch | truncated_batch)) * self.gamma * torch.max(q_next, dim=1)[0]

            '''Bellman equation for stochastic environment'''
            target_value = (1 - self.Q_learning_rate) * current_memory + self.Q_learning_rate * input_gate        
        
        td_errors = torch.abs(target_value - current_memory).detach().cpu().numpy()

        self.memory.update_priorities(indices, td_errors)

        loss = (weights * (current_memory - target_value) ** 2).mean()

        self.QNetwork.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.QNetwork.parameters(), max_norm=10.0)
        self.QNetwork.optimizer.step()
