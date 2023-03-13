import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LSTM(nn.Module):
    """Initializes an LSTM model for time series forecast prediction.

    Args:
        input_dim (int): Number of input dimensions
        hidden_dim (int): Size of hidden layer
        num_layers (int): Number of layers in the LSTM model
        output_dim (int): Size of output layer

    Returns:
        None
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device=None):
        super(LSTM, self).__init__()

        # Initialize hidden dimension and layers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim

        # Initialize deep learning models
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)

        self.device = device

    def forward(self, x):
        """Perform one forward pass of the LSTM model

        Args:
            data (tensor): Time series data

        Returns:
            Tensor containing outputs of all input data
        """

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # Reshape data if needed
        if len(x.shape) != 3:
            x = x.reshape(x.shape[0], -1, self.input_dim).to(self.device)

        # Run data through model
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out