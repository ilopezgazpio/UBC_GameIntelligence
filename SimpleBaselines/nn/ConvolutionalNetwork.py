import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


class ConvolutionalNetwork(nn.Module):

    def __init__(
            self,
            input_layer_size, #(height, width, channels) triplet
            output_layer_size,
            hidden_layers_size=[], # triplets for nn.conv2d out_channels, kernel_size, stride
            activation_fn=nn.ReLU,
            learning_rate=0.01,
            dropout=0.0,
            use_batch_norm=False,
            loss_fn=nn.MSELoss,
            optimizer=optim.Adam,
            seed=None
    ):

        super(ConvolutionalNetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            random.seed(seed)


        # NN structure
        self.network = self.build_layers(
            input_layer_size,
            hidden_layers_size,
            output_layer_size,
            activation_fn,
            dropout,
            use_batch_norm
        )
        print(self.network)

        # GPU capabilities
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # Parameters for optimization
        self.loss_fn = loss_fn()
        self.optimizer = optimizer(params=self.network.parameters(), lr=learning_rate)



    def build_layers(self,
                     input_layer_size,
                     hidden_layers_size,
                     output_layer_size,
                     activation_fn,
                     dropout,
                     use_batch_norm):

        layers = []
        num_layers = len(hidden_layers_size)

        current_channels = input_layer_size[2]
        current_height = input_layer_size[0]
        current_width = input_layer_size[1]

        # Build convolutional layers
        for i in range(num_layers):
            out_channels = hidden_layers_size[i][0]
            kernel_size = hidden_layers_size[i][1]
            stride = hidden_layers_size[i][2]
            padding = 0

            # Add Conv2d layer
            layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )

            # Update current_channels
            current_channels = out_channels

            # Compute output dimensions
            current_height = (current_height + 2 * padding - kernel_size) // stride + 1
            current_width = (current_width + 2 * padding - kernel_size) // stride + 1

            # Apply batch normalization if enabled
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(current_channels))

            # Apply activation function
            if activation_fn is not None:
                layers.append(activation_fn())

            # Apply dropout if enabled
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))

        # Flatten the output
        layers.append(nn.Flatten(start_dim=0, end_dim=-1))

        # Compute flattened size
        flattened_size = current_channels * current_height * current_width

        # Add the output Linear layer
        layers.append(
            nn.Linear(flattened_size, output_layer_size)
        )

        return nn.Sequential(*layers)

    '''
    def build_layers(self, input_layer_size, hidden_layers_size, output_layer_size, activation_fn, dropout, use_batch_norm):

        # Build layers using nn.Sequential
        layers = []
        in_channels = input_layer_size[2]  # Assuming input_layer_size is (H, W, C)

        for i, layer_params in enumerate(hidden_layers_size):

            out_channels, kernel_size, stride = layer_params

            # Add Conv2d layer
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
            )

            # Optionally add BatchNorm2d
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))

            # Add activation function
            if activation_fn is not None:
                layers.append(activation_fn())

            # Optionally add Dropout
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            in_channels = out_channels  # Update for next layer

        # Add output layer
        layers.append(
            nn.Flatten()
        )

        # Determine the size after convolutional layers to connect to the output layer
        # This requires knowing the spatial dimensions after convolutions
        # For simplicity, we'll assume the input size remains the same due to paddingand compute the flattened size accordingly
        H, W, C = input_layer_size

        for layer_params in hidden_layers_size:
            _, kernel_size, stride = layer_params
            H = (H - kernel_size) // stride + 1
            W = (W - kernel_size) // stride + 1

        flattened_size = hidden_layers_size[-1][0] * H * W

        # Add a fully connected layer to the output
        layers.append(nn.Linear(flattened_size, output_layer_size))

        return nn.Sequential(*layers)
    '''

    def toDevice(self, x : np.array, dType=torch.float32):
        return torch.tensor(x, dtype=dType).to(self.device)


    def forward(self, x):
        if x.ndim >= 3 and x.shape[-1] == 3:  # Check if channels are in the last dimension
            x = x.permute(2, 0, 1)  # Rearrange to [batch_size, channels, height, width]
        return self.network(x)


    def update_NN(self, predicted_value, target_value):
        'Update the weights of the NN'
        loss = self.loss_fn(predicted_value, target_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
