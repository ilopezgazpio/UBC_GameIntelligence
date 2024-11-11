import random

import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class NeuralNetwork(nn.Module):

    def __init__(
            self,
            input_layer_size,
            output_layer_size,
            hidden_layers_size=None,
            activation_fn=nn.Tanh,
            learning_rate=0.001,
            dropout=0.0,
            use_batch_norm=False,
            loss_fn=nn.MSELoss,
            optimizer=optim.Adam,
            seed=None,):

        super(NeuralNetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        # Build layer sizes list
        layer_sizes = [input_layer_size] + hidden_layers_size + [output_layer_size]

        # NN structure
        self.network = self.build_layers(layer_sizes, activation_fn, dropout, use_batch_norm)

        # Parameters for optimization
        self.loss_fn = loss_fn()
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate)


        # GPU capabilities
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)



    def build_layers(self, layer_sizes, activation_fn, dropout, use_batch_norm):

        # Build layers using nn.Sequential
        layers = []
        num_layers = len(layer_sizes) - 1

        for i in range(num_layers):

            # Add linear layer
            layers.append(nn.Linear(
                layer_sizes[i], layer_sizes[i + 1])
            )

            # Apply batch normalization if enabled and not on the output layer
            if use_batch_norm and i < num_layers - 1:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))

            # Apply activation function if not on the output layer
            if i < num_layers - 1 and activation_fn is not None:
                layers.append(activation_fn())

            # Apply dropout if enabled and not on the output layer
            if dropout > 0.0 and i < num_layers - 1:
                layers.append(nn.Dropout(dropout))

        # Return the sequential model
        return nn.Sequential(*layers)


    def forward(self, observation : np.array):
        observation = torch.Tensor(observation).to(self.device)
        return self.network(observation)


    def update_NN(self, predicted_value, target_value):
        'Update the weights of the NN'
        loss = self.loss_fn(predicted_value, target_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()