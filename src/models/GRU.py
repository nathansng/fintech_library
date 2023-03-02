import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device=None):
        super(GRU, self).__init__()
        
        # Initialize hidden dimenision and layers
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = inpu_dim
        
        # Initialize deep learning models
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first = True).to(device)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)

        # Reshape data if needed
        if len(x.shape) != 3:
            x = x.reshape(x.shape[0], -1, self.input_dim).to(self.device)

        output, hidden = self.gru(x, hidden)
        output = self.fc(output[:, -1, :])
        return output

        