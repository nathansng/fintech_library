import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TreNetCNN(nn.Module):
    def __init__(self, num_data, layers=None, num_filters=None, dropout=None, conv_size=3, pooling_size=3, output_size=2, device=None):
        """
        layers (int): Number of cnn stacks to create
        num_filters (list(int)): Number of CNN filters corresponding to same index stacks
        dropout (list(float)): Probability of dropout corresponing to same index stack
        convsize (int, list(int)): Size of filter sizes
        """
        super(TreNetCNN, self).__init__()
        self.num_data = num_data
        self.layers = layers
        self.num_filters = num_filters
        self.dropout = dropout
        self.conv_size = conv_size
        self.pooling_size = pooling_size
        self.output_size = output_size
        self.cnn_stack = self.create_cnn_stack()

    def create_cnn_stack(self):
        # Initialize default stack settings
        if not self.layers:
            self.layers = 2
        if not self.num_filters:
            self.num_filters = [128] * self.layers
        if not self.dropout:
            self.dropout = [0.0] * self.layers
        if type(self.conv_size) == int:
            self.conv_size = [self.conv_size] * self.layers

        # Create cnn stacks
        cnn_stacks = []
        num_channels = 1
        updated_data = self.num_data
        for i in range(self.layers):
            cnn_stack = nn.Sequential(
                nn.Conv1d(in_channels=num_channels, out_channels=self.num_filters[i], kernel_size=self.conv_size[i]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=self.pooling_size, stride=1),
                nn.Dropout(p=self.dropout[i])
            )
            num_channels = self.num_filters[i]
            cnn_stacks.append(cnn_stack)

            # Keep track of current size of data
            updated_data = updated_data - self.conv_size[i] + 1 - self.pooling_size + 1
            new_data_size = self.num_filters[i] * updated_data

        # Add fully connected layer at end to output trend duration and slope
        output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(new_data_size, self.output_size)
        )
        cnn_stacks.append(output_layer)

        # Combine cnn stacks
        return nn.Sequential(*cnn_stacks)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, -1))
        output = self.cnn_stack(x)
        return output