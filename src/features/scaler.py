import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


"""
Scales multiple sources of data using Scaler class
"""
class MultiScaler(): 
    def __init__(self, num_sources): 
        self.Scalers = [Scaler() for i in range(num_sources)]
       
    def fit_transform(self, data):
        scale_data = []
        for i in range(len(data)): 
            if data[i] != None: 
                scale_data.append(self.Scalers[i].fit_transform(data[i]))
            else: 
                scale_data.append(None)
        return scale_data
    
    def inverse_transform(self, data): 
        unscale_data = []
        for i in range(len(data)): 
            if data[i] != None: 
                unscale_data.append(self.Scalers[i].inverse_transform(data[i]))
            else:
                unscale_data.append(None)
        return unscale_data


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