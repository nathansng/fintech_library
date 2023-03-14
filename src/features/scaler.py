import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


class MultiScaler():
    """ Scales multiple sources of data using the Scaler class. See :class:`~features.scaler.Scaler` for more details.

    Args:
        num_sources (int): Number of different sources to scale

    Returns:
        None
    """

    def __init__(self, num_sources):
        self.Scalers = [Scaler() for i in range(num_sources)]


    def fit_transform(self, data):
        """ Fits and scales all data sources. Use None to fill in missing data sources.

        Args:
            data (list[tensor or dataframe or series]): Data to fit and transform

        Returns:
            List of all scaled data
        """
        scale_data = []
        for i in range(len(data)):
            if data[i] != None:
                scale_data.append(self.Scalers[i].fit_transform(data[i]))
            else:
                scale_data.append(None)
        return scale_data


    def transform(self, data):
        """Transforms all data sources according to pre-trained scalers. Use None to fill in missing data sources.

        Args:
            data (list[tensor or dataframe or series]): Data to transform

        Returns:
            List of all scaled data
        """

        scale_data = []
        for i in range(len(data)):
            if data[i] != None:
                scale_data.append(self.Scalers[i].transform(data[i]))
            else:
                scale_data.append(None)
        return scale_data


    def inverse_transform(self, data):
        """ Inverse transform scaled data. Use None to fill in missing data sources.

        Args:
            data (list[tensor]): Data to revert back to original values

        Returns:
            Returns list of tensors of inversely transformed values
        """

        unscale_data = []
        for i in range(len(data)):
            if data[i] != None:
                unscale_data.append(self.Scalers[i].inverse_transform(data[i]))
            else:
                unscale_data.append(None)
        return unscale_data


class Scaler():
    """ Scales data using Sklearn's MinMaxScaler. Generalized to accept tensors.

    Returns:
        None
    """

    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, data):
        """ Fits and scales data

        Args:
            data (tensor or dataframe or series): Data to fit and transform

        Returns:
            Tensor of scaled data values
        """

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


    def transform(self, data):
        """ Scales data according to pretrained scaler

        Args:
            data (tensor or dataframe or series): Data to transform

        Returns:
            Tensor of scaled data values
        """

        # Check if data is tensor
        if type(data) == torch.Tensor:
            data = torch.Tensor.cpu(data).detach().numpy()

        # Check if data is dataframe
        if type(data) == pd.DataFrame or type(data) == pd.Series:
            data = data.values

        # Transform data
        if len(data.shape) == 1:
            scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        else:
            scaled_data = self.scaler.transform(data)

        # Return tensor of scaled data
        return torch.tensor(scaled_data, dtype=torch.float)


    def inverse_transform(self, data):
        """ Inverse transform scaled data

        Args:
            data (tensor or dataframe or series): Data to revert back to original values

        Returns:
            Tensor of inversely transformed values
        """
        # Check if data is tensor
        if type(data) == torch.Tensor:
            data = torch.Tensor.cpu(data).detach().numpy()

        # Check if data is dataframe
        if type(data) == pd.DataFrame or type(data) == pd.Series:
            data = data.values

        inverse_data = self.scaler.inverse_transform(data)

        # Return tensor of inverse data
        return torch.tensor(inverse_data, dtype=torch.float)
