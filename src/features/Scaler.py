import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


"""
Scales data using Sklearn's MinMaxScaler
Generalized to accept tensors
"""
class Scaler():
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, data):
        # Check if data is tensor
        if type(data) == torch.Tensor:
            data = torch.Tensor.cpu(data).detach().numpy()

        # Check if data is dataframe
        if type(data) == pd.DataFrame or type(data) == pd.Series:
            data = data.values

        # Transform data
        if len(data.shape) == 1:
            scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        else:
            scaled_data = self.scaler.fit_transform(data)

        # Return tensor of scaled data
        return torch.tensor(scaled_data, dtype=torch.float)

    def inverse_transform(self, data):
        # Check if data is tensor
        if type(data) == torch.Tensor:
            data = torch.Tensor.cpu(data).detach().numpy()

        # Check if data is dataframe
        if type(data) == pd.DataFrame or type(data) == pd.Series:
            data = data.values

        inverse_data = self.scaler.inverse_transform(data)

        # Return tensor of inverse data
        return torch.tensor(inverse_data, dtype=torch.float)