import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class MultiDiscreteNeuralNetwork(nn.Module):

    def __init__(
            self,
            input_layer_size,
            output_layer_size,
            hidden_layers_size=[64],
            activation_fn=nn.Tanh,
            learning_rate=0.01,
            dropout=0.0,
            use_batch_norm=False,
            loss_fn=nn.MSELoss,
            optimizer=optim.Adam,
            seed=None
    ):

        super(MultiDiscreteNeuralNetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            random.seed(seed)

        # Build layer sizes list
        layer_sizes = [input_layer_size] + hidden_layers_size

        # NN structure
        self.base_network = self.build_layers(layer_sizes, activation_fn, dropout, use_batch_norm)
        self.action_networks = [nn.Linear(hidden_layers_size[-1], output_layer_size[i]) for i in range(len(output_layer_size))]

        # GPU capabilities
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        [action_network.to(self.device) for action_network in self.action_networks]

        # Parameters for optimization
        self.loss_fn = loss_fn()
        self.optimizer = optimizer(params=self.base_network.parameters(), lr=learning_rate)


    def build_layers(self, layer_sizes, activation_fn, dropout, use_batch_norm):

        # Build layers using nn.Sequential
        layers = []
        num_layers = len(layer_sizes) - 1

        for i in range(num_layers):

            # Add linear layer
            layers.append(nn.Linear(
                layer_sizes[i], layer_sizes[i + 1])
            )

            # Apply batch normalization if enabled
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))

            # Apply activation function
            if i < num_layers - 1 and activation_fn is not None:
                layers.append(activation_fn())

            # Apply dropout if enabled
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        # Return the sequential model
        return nn.Sequential(*layers)


    def toDevice(self, x : np.array, dType=torch.float32):
        # return torch.tensor(x, dtype=dType).to(self.device)
        x = np.array(x)
        return torch.from_numpy(x).type(dType).to(self.device)


    def forward(self, x):
        base_results = self.base_network(x)
        action_results = torch.stack([action_network(base_results) for action_network in self.action_networks], dim=-2)
        return action_results



    def update_NN(self, predicted_value, target_value, clip_error=False):
        'Update the weights of the NN'
        loss = self.loss_fn(predicted_value, target_value)
        self.optimizer.zero_grad()
        loss.backward()
        if clip_error:
            for param in self.base_network.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()