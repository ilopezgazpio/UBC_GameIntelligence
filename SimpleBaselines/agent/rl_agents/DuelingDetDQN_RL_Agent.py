from SimpleBaselines.states.State import State
from SimpleBaselines.agent.rl_agents.DoubleDetDQN_RL_Agent import DoubleDetDQN_RL_Agent
from SimpleBaselines.nn.DuelingNeuralNetwork import DuelingNeuralNetwork
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


class DuelingDetDQN_RL_Agent(DoubleDetDQN_RL_Agent):

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
                         per_variant=per_variant,
                         target_net_update_steps=target_net_update_steps,
                         clip_error=clip_error
                         )

        ''' Parameters for the Dueling DQN agent '''
        # Set action decision function
        # Dueling DQN agent plays as a standard Double DQN with experience replay with the following modification:
        # The neural networking architecture is modified to have two separate streams for state value and advantage value

        # internal estimator NN updates following the deterministic scenario Bellman equation under specified loss function
        # BUT training is more stable since parameters of training net are not always exposed at the time of optimizing

        # action decision function is inherited from superclass
        # self.__action_decision_function__ = self.__DQN_decision_function__

        # update function is inherited from superclass
        # self.__update_function__ = self.__Double_DQN_bellman_update__

        def __init_NN__(self, seed, hidden_layers_size, activation_fn, dropout, use_batch_norm, loss_fn, optimizer, nn_learning_rate):
            return DuelingNeuralNetwork(
                input_layer_size = self.env.observation_space.shape[0],
                output_layer_size = self.env.action_space.n,
                hidden_layers_size = hidden_layers_size,
                activation_fn = activation_fn,
                learning_rate = nn_learning_rate,
                dropout = dropout,
                use_batch_norm = use_batch_norm,
                loss_fn = loss_fn,
                optimizer = optimizer,
                seed = seed
            )
