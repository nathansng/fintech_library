import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

class GRU(nn.Module):
    """Initializes a GRU model for time series forecast prediction.

    Args:
        input_dim (int): Number of input dimensions
        hidden_dim (int): Size of hidden layer
        num_layers (int): Number of layers in the LSTM model
        output_dim (int): Size of output layer
        device (Torch device): Device to store model on

    Returns:
        None
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim,  device=None):
        super(GRU, self).__init__()

        # Initialize hidden dimenision and layers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.device = device

        # Initialize deep learning models
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first = True).to(device)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):
        """Perform one forward pass of the GRU model

        Args:
            data (tensor): Time series data

        Returns:
            Tensor containing outputs of all input data
        """

        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        # Reshape data if needed
        if len(x.shape) != 3:
            x = x.reshape(x.shape[0], -1, self.input_dim).to(self.device)

        output, hidden = self.gru(x, hidden)
        output = self.fc(output[:, -1, :])
        return output

