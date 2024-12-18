import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from SimpleBaselines.memory.MixedExperienceReplay import MixedExperienceReplay
from SimpleBaselines.agent.rl_agents.DoubleDetDQN_RL_Agent import DoubleDetDQN_RL_Agent

class MixedExperienceDQN_RL_Agent(DoubleDetDQN_RL_Agent):

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
                         batch_size=batch_size,
                         target_net_update_steps=target_net_update_steps,
                         clip_error=clip_error
                         )

        ''' Parameters for the Mixed Experience DQN agent '''
        self.memory = MixedExperienceReplay(memory_size, data_size, batch_size, data_amount_start, data_amount_end, data_amount_decay)
        
        # Set action decision function
        # Mixed Experience DQN agent plays as a standard Dueling DQN with experience replay with the following modification:
        # The memory buffer is a mix of experience replay and a set of expert demonstrations, with a specified ratio of data from each

        # internal estimator NN updates following the deterministic scenario Bellman equation under specified loss function

    def populate_data(self, data):
        """
        Populate the memory with data
        data: list of tuples (state, action, next_state, reward, terminated, truncated)
        """
        self.memory.populate_data(data)

