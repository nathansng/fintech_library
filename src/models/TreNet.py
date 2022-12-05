import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from LSTM import LSTM
from CNN import TreNetCNN


class TreNet(nn.Module):
    def __init__(self, LSTM_params, CNN_params, feature_fusion, output_dim, device=None):
        super(TreNet, self).__init__()

        # Set number of parameters for feature fusion layer
        LSTM_params['output_dim'] = feature_fusion
        CNN_params['output_size'] = feature_fusion

        self.lstm = LSTM(**LSTM_params)
        self.cnn = TreNetCNN(**CNN_params)
        self.fusion = nn.Linear(feature_fusion, output_dim)
        self.cutoff = CNN_params['num_data']

        self.device = device

    def forward(self, data):
        trends, data = data

        # Run trends through LSTM
        lstm_out = self.lstm(trends)

        # Set cutoff for CNN stock prices
        cutoff_data = torch.zeros(data.shape[0], self.cutoff).to(self.device)
        for i in range(data.shape[0]):
            cutoff_data[i] = data[i, -self.cutoff:]

        # Run stock prices through CNN
        cnn_out = self.cnn(cutoff_data)

        # Concat outputs in feature fusion layer
        feature_in = torch.add(lstm_out, cnn_out)

        # Run outputs through feature fusion layer
        output = self.fusion(feature_in)
        return output