from SimpleBaselines.states.State import State
from SimpleBaselines.agent.rl_agents.PERStocDQN_RL_Agent import PERStocDQN_RL_Agent
from SimpleBaselines.nn.NeuralNetwork import NeuralNetwork

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


class STNStocDQN_RL_Agent(PERStocDQN_RL_Agent): # with PER memory

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
                 per_variant='proportional',
                 target_net_update_steps=500,
                 clip_error=True
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
                         optimizer=optimizer,
                         memory_size=memory_size,
                         batch_size=batch_size,
                         per_alpha=per_alpha,
                         per_beta=per_beta,
                         per_beta_increment=per_beta_increment,
                         per_epsilon=per_epsilon,
                         per_variant=per_variant
                         )


        ''' Parameters for the Stable Target Net DQN agent '''
        self.update_target_counter = 1
        self.target_net_update_steps = target_net_update_steps
        self.clip_error = clip_error

        # Create the stable network (must be a clone of the QNetwork)
        self.stable_target_net = self.__init_NN__(seed, hidden_layers_size, activation_fn, dropout, use_batch_norm, loss_fn, optimizer, nn_learning_rate)

        # Set action decision function
        # Stable Target Network DQN agent plays as a standard Deterministic DQN with experience replay
        # internal estimator NN updates following the deterministic scenario Bellman equation under specified loss function
        # BUT training is more stable since parameters of training net are not always exposed at the time of optimizing

        # action decision function is inherited from superclass
        # self.__action_decision_function__ = self.__DQN_decision_function__

        # we need to update the update function to use experience replay
        self.__update_function__ = self.__STN_DQN_bellman_update__


    def __STN_DQN_bellman_update__(self, old_state: State, action, new_observation : gym.Space, reward: float, terminated, truncated):

        # Store experience in memory
        self.memory.push(old_state.observation, action, new_observation, reward, terminated, truncated)
        
        if len(self.memory) < self.batch_size:
            # not enough samples in memory
            return

        old_observation_batch, action_batch, new_observation_batch, reward_batch, terminated_batch, truncated_batch, weights, indices = self.memory.sample()

        '''DQN Bellman equation update USING STABLE NETWORK'''
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
            '''Input cell (right part of Q-learning equation)'''
            q_next = self.stable_target_net(new_observation_batch)
            input_gate = reward_batch + (1 - (terminated_batch | truncated_batch)) * self.gamma * torch.max(q_next, dim=1)[0]

            '''Bellman equation for stochastic environment. 
            Target with TARGET NETWORK'''
            target_value = (1 - self.Q_learning_rate) * current_memory + self.Q_learning_rate * input_gate
        
        # Update PER memory
        td_errors = torch.abs(target_value - current_memory).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        loss = (weights * (current_memory - target_value) ** 2).mean()

        self.QNetwork.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.QNetwork.parameters(), max_norm=10.0)
        self.QNetwork.optimizer.step()

        # Update target net evey target_net_update_steps steps
        if self.update_target_counter % self.target_net_update_steps == 0:
            self.stable_target_net.network.load_state_dict(self.QNetwork.network.state_dict())

        self.update_target_counter += 1
